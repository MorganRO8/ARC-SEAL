import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from typing import Dict, Any, Callable, List, Optional, Union, Sequence, Mapping
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from tqdm import tqdm
import numpy as np
import torch
import datetime
import uuid
import math

from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .arc import Example, Grid, Task
from .representers import WordGridRepresenter

import itertools
from typing import List

import numpy as np

from .arc import Task
from .augmenters import (
    Augmenter,
    Chain,
    Concat,
    Flip,
    IdentityAugmenter,
    IncreaseHeight,
    IncreaseResolution,
    IncreaseWidth,
    PermuteColors,
    PermuteExamples,
    RandomTranslateXY,
    Reflect,
    Repeat,
    Rotate,
    Transpose,
)
from .messagers import MessageRepresenter


class TTT:
    def __init__(self, model_name: str, state_dict_path: Optional[str] = None, lora_config: Optional[LoraConfig] = None):
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Make sure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        if state_dict_path is not None:
            state_dict = torch.load(state_dict_path)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded state dict from {state_dict_path}")
        
        self.model = get_peft_model(model, lora_config) 

        # Store initial LoRA A parameter values
        self.initial_lora_A = {}
        for name, param in self.model.named_parameters():
            if "lora_A" in name:
                # Store a deep copy of the initial parameters
                self.initial_lora_A[name] = param.data.clone().detach()
    
    def update_model(
        self,
        task_text_list: Sequence[Union[str, Mapping[str, torch.Tensor]]],
        output_dir: str,
        batch_size: int,
        gradient_accumulation_steps: int,
        learning_rate: float,
        num_train_epochs: int,
        lr_scheduler_type: str,
        loss_on_all_tokens: bool,
    ) -> str:
        """
        Update the model using the provided task by:
        1. Reset LoRA weights
        2. Create training data from the task
        3. Train the model using SFT
        4. Save the LoRA weights
        
        Args:
            task: ARC task to use for training
            task_processor: Function to process tasks (like get_preprocessed_tasks_single)
            representer: Representer for formatting tasks (like GPTTextMessageRepresenterForBarc)
            output_dir: Directory to save model
            batch_size: Batch size for training
            gradient_accumulation_steps: Number of gradient accumulation steps
            learning_rate: Learning rate for training
            num_train_epochs: Number of epochs for training
            lr_scheduler_type: Learning rate scheduler type
            
        Returns:
            Path to the saved model
        """
        # Reset the LoRA weights
        self.reset_lora()
        
        if not task_text_list:
            print("No training samples provided; skipping fine-tuning.")
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            return output_dir

        first_sample = task_text_list[0]
        if isinstance(first_sample, Mapping):
            training_data = self._collate_tokenized_samples(task_text_list)  # type: ignore[arg-type]
        else:
            training_data = self._tokenize_and_process(task_text_list, loss_on_all_tokens)
        
        # clear cache
        torch.cuda.empty_cache()

        # Train the model
        self._train_model(
            training_data,
            output_dir=output_dir,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            lr_scheduler_type=lr_scheduler_type,
        )
        
        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
        return output_dir
        
    def reset_lora(self):
        """Reset all LoRA B parameters to zero and LoRA A parameters to their initial values."""
        for name, param in self.model.named_parameters():
            if "lora_B" in name:
                param.data.fill_(0.0)
            elif "lora_A" in name and name in self.initial_lora_A:
                # Restore the original values of LoRA A parameters
                param.data.copy_(self.initial_lora_A[name])
        
    def _tokenize_and_process(self, text_list: List[str], loss_on_all_tokens: bool):
        """
        Tokenize a list of texts and set up labels for instruction fine-tuning.
        Specifically looks for the second-to-last occurrence of the assistant header token sequence.
        Processes all texts in parallel for efficiency.
        
        Args:
            text_list: List of text strings to process
            
        Returns:
            Dictionary with tokenized inputs and labels
        """
        # Tokenize all texts in a batch
        outputs = self.tokenizer(
            text_list,
            truncation=True,
            max_length=8192,
            padding="longest",
            return_tensors="pt"
        )
        input_ids = outputs["input_ids"]
        
        # Process all samples in parallel
        batch_size = input_ids.shape[0]
        labels = input_ids.clone()

        if not loss_on_all_tokens:
            for i in range(batch_size):
                sample_input_ids = input_ids[i].tolist()
                special_indices = []
                for j in range(len(sample_input_ids) - 1):
                    if sample_input_ids[j] == 128007 and sample_input_ids[j + 1] == 271:
                        special_indices.append(j + 1)

                if not special_indices:
                    print(
                        f"Warning: Assistant header sequence not found in sample {i}; skipping loss masking."
                    )
                    labels[i, :] = -100
                    continue

                if len(special_indices) >= 2:
                    special_index = special_indices[-2]
                else:
                    special_index = special_indices[-1]

                labels[i, : special_index + 1] = -100

        outputs["labels"] = labels
        return outputs

    def _collate_tokenized_samples(
        self,
        samples: Sequence[Mapping[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        input_tensors: List[torch.Tensor] = []
        attention_tensors: List[torch.Tensor] = []
        label_tensors: List[torch.Tensor] = []

        for sample in samples:
            input_tensor = sample["input_ids"]
            attention_tensor = sample["attention_mask"]
            label_tensor = sample["labels"]

            if input_tensor.dim() > 1:
                input_tensor = input_tensor.squeeze(0)
            if attention_tensor.dim() > 1:
                attention_tensor = attention_tensor.squeeze(0)
            if label_tensor.dim() > 1:
                label_tensor = label_tensor.squeeze(0)

            input_tensors.append(input_tensor.cpu().to(torch.long))
            attention_tensors.append(attention_tensor.cpu().to(torch.long))
            label_tensors.append(label_tensor.cpu().to(torch.long))

        padded_input = pad_sequence(
            input_tensors,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        padded_attention = pad_sequence(attention_tensors, batch_first=True, padding_value=0)
        padded_labels = pad_sequence(label_tensors, batch_first=True, padding_value=-100)

        return {
            "input_ids": padded_input,
            "attention_mask": padded_attention,
            "labels": padded_labels,
        }
    
    def _train_model(
        self,
        training_data: Dict[str, Any],
        output_dir: str,
        batch_size: int,
        gradient_accumulation_steps: int,
        learning_rate: float,
        num_train_epochs: int,
        lr_scheduler_type: str,
    ):
        """
        Train the model using the provided text examples.
        
        Args:
            text_list: List of formatted training examples
            output_dir: Directory to save checkpoints
            batch_size: Batch size for training
            gradient_accumulation_steps: Number of gradient accumulation steps
            learning_rate: Learning rate for training
            num_train_epochs: Number of epochs for training
        
        Returns:
            Trained model and tokenizer
        """
        num_examples = int(training_data["input_ids"].shape[0])
        print(
            f"Training on {num_examples} examples for {num_train_epochs} epochs, lr: {learning_rate}"
        )
        ds = Dataset.from_dict(training_data)

        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        if getattr(self.model, "config", None) is not None:
            try:
                self.model.config.use_cache = False
            except AttributeError:
                pass

        target_batch_size = max(int(batch_size), 1)
        dataset_limited_batch = max(1, min(target_batch_size, num_examples))
        target_global_batch = max(1, target_batch_size * max(int(gradient_accumulation_steps), 1))
        attempted_batch_size = dataset_limited_batch

        def _is_cuda_oom(error: BaseException) -> bool:
            if isinstance(error, torch.cuda.OutOfMemoryError):
                return True
            message = str(error).lower()
            return "cuda out of memory" in message or "cuda error: out of memory" in message

        last_error: Optional[BaseException] = None

        while attempted_batch_size >= 1:
            effective_grad_accum = max(1, math.ceil(target_global_batch / attempted_batch_size))
            print(
                "Starting training attempt with per_device_train_batch_size="
                f"{attempted_batch_size}, gradient_accumulation_steps={effective_grad_accum}"
            )

            training_args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=attempted_batch_size,
                gradient_accumulation_steps=effective_grad_accum,
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                lr_scheduler_type=lr_scheduler_type,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                bf16=True,
                gradient_checkpointing=True,
                remove_unused_columns=False,
                optim="adamw_torch",
                warmup_steps=11,
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=ds,
            )

            try:
                print("Starting training...")
                trainer.train()
                print("Training complete.")
                break
            except Exception as error:  # noqa: BLE001
                if _is_cuda_oom(error) and attempted_batch_size > 1:
                    last_error = error
                    attempted_batch_size = max(attempted_batch_size // 2, 1)
                    print(
                        "Encountered CUDA OOM; retrying with per_device_train_batch_size="
                        f"{attempted_batch_size}."
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.reset_lora()
                    continue
                raise
        else:
            if last_error is not None:
                raise last_error

        return self.model, self.tokenizer
