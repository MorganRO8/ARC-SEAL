import os
import re
import json
import glob
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Optional
from datetime import datetime
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from builtins import input
import argparse

from peft import LoraConfig
import arclib
from arclib.arc import Example, Task
from arclib.arc import (
    make_submission,
    read_tasks_from_single_file,
    to_list,
    to_tuple,
)
from arclib.representers import (
    CompositeRepresenter,
    ConnectedComponentRepresenter,
    DelimitedGridRepresenter,
    DiffExampleRepresenter,
    GridRepresenter,
    ImageTaskRepresenter,
    PythonListGridRepresenter,
    TaskRepresenter,
    TextTaskRepresenter,
    TextExampleRepresenter,
    WordGridRepresenter,
)
from arclib.messagers import (
    GPTTextMessageRepresenterForBarc,
    GPTTextMessageRepresenterV2,
    PythonSolverMessageRepresenter,
)
from arclib.update_model import TTT
from inference.preprocess import get_preprocessed_tasks_single

from arclib.voting import vote
from arclib.eval import evaluate
from inference.engine_vllm import get_sampling_params, initialize_engine, process_requests
from inference.preprocess import get_preprocessed_tasks

import itertools
from typing import List

import numpy as np

from arclib.arc import Task
from arclib.augmenters import (
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
from arclib.messagers import MessageRepresenter

from vllm import LLM, SamplingParams

from utils.prompts import self_edit_prompt, system_message
from utils.python_executor import extract_solver_code, run_solver


def mode_array(array_list):
    """Return the most common array from a list of arrays."""
    tuple_shape_list = [(tuple(arr.flatten()), arr.shape) for arr in array_list]
    most_common_tuple_shape, _ = Counter(tuple_shape_list).most_common(1)[0]
    most_common_tuple, original_shape = most_common_tuple_shape
    mode_arr = np.array(most_common_tuple).reshape(original_shape)
    return mode_arr


def read_tasks_from_folder(task_folder: str, test: bool = False) -> List[Task]:
    """Read tasks from a folder of JSON files."""
    all_tasks = []
    for file in glob.glob(f"{task_folder}/*.json"):
        basename = os.path.basename(file)
        idx = basename.replace(".json", "")
        tasks = read_tasks_from_file(file, test=test)
        for i, task in enumerate(tasks):
            task.name = idx + "-" + str(i)
        all_tasks += tasks
    return all_tasks


def read_tasks_from_single_file(
    challenge_file: str, test: bool = False, solution_file: Optional[str] = None
) -> List[Task]:
    """Read tasks from a single JSON file with optional solutions."""
    with open(challenge_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if solution_file is not None:
        test = False
        with open(solution_file, "r", encoding="utf-8") as handle:
            solutions = json.load(handle)
            for key, value in solutions.items():
                for idx, solution in enumerate(value):
                    data[key]["test"][idx]["output"] = solution

    all_tasks = []
    for task_name, subtasks in data.items():
        parsed_tasks = Task.read_tasks_from_dict(subtasks, test=test)
        for i, task in enumerate(parsed_tasks):
            task.name = task_name + "-" + str(i)
            all_tasks.append(task)

    return all_tasks


def read_tasks_from_file(task_file: str, test: bool = False) -> List[Task]:
    """Read tasks from a JSON file."""
    with open(task_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return Task.read_tasks_from_dict(data, test=test)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


def get_augmenters(
    include_basic: bool = True,
    include_size: bool = True,
    include_chain: bool = True,
    include_repeat: bool = True,
    include_concat: bool = False,
) -> List[Augmenter]:
    basic_augmenters_to_apply = (
        [
            Rotate(90),
            Rotate(270),
            Rotate(180),
            Flip(0),
            Flip(1),
            Reflect(0, reverse=True),
            Reflect(1, reverse=True),
            Reflect(0, reverse=False),
            Reflect(1, reverse=False),
            RandomTranslateXY(),
            Transpose(),
        ]
        if include_basic
        else []
    )

    size_augmenters_to_apply = (
        [
            IncreaseResolution(2),
            IncreaseHeight(2),
            IncreaseWidth(2),
        ]
        if include_size
        else []
    )

    concat_augmenters_to_apply = (
        [
            Concat((IdentityAugmenter(), Rotate(180)), axis=0),
            Concat((IdentityAugmenter(), Rotate(180)), axis=1),
        ]
        if include_concat
        else []
    )

    chain_augmenters_to_apply = (
        [
            Chain([Rotate(90), IncreaseResolution(2)]),
            Chain([Rotate(270), IncreaseResolution(2)]),
            Chain([Rotate(180), IncreaseResolution(2)]),
            Chain([Flip(0), IncreaseResolution(2)]),
            Chain([Flip(1), IncreaseResolution(2)]),
            Chain([Transpose(), IncreaseResolution(2)]),
        ]
        if include_chain
        else []
    )

    repeat_augmenters_to_apply = (
        [
            Repeat(0, 2),
            Repeat(1, 2),
            Repeat(2, 2),
        ]
        if include_repeat
        else []
    )

    augmenters_to_apply = (
        basic_augmenters_to_apply
        + size_augmenters_to_apply
        + concat_augmenters_to_apply
        + chain_augmenters_to_apply
        + repeat_augmenters_to_apply
    )

    #print("Augmenters to apply: ", augmenters_to_apply, "len: ", len(augmenters_to_apply))
    return augmenters_to_apply

def _tokenize_and_process(
    text: str,
    tokenizer,
    *,
    loss_on_all_tokens: bool = False,
):
    """Tokenize a chat transcript and mask non-response tokens."""

    outputs = tokenizer(
        text,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = outputs["input_ids"].squeeze(0).to(torch.long).cpu()
    attention_mask = outputs["attention_mask"].squeeze(0).to(torch.long).cpu()
    labels = input_ids.clone()

    if not loss_on_all_tokens:
        ids_list = input_ids.tolist()
        special_indices = []
        for i in range(len(ids_list) - 1):
            if ids_list[i] == 128007 and ids_list[i + 1] == 271:
                special_indices.append(i + 1)

        if not special_indices:
            raise ValueError("Assistant header token sequence not found in sample.")

        if len(special_indices) >= 2:
            special_index = special_indices[-2]
        else:
            special_index = special_indices[-1]

        labels[: special_index + 1] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def format_and_filter(
    formatter,
    tokenizer,
    task,
    train_on_input: bool = False,
    *,
    code_mode: bool = False,
):
    if code_mode:
        chat_messages = task.get("chat_messages")
        if chat_messages is None:
            raise ValueError("Code-mode transcripts must include 'chat_messages'.")

        assistant_message = chat_messages[-1]
        prompt_messages = chat_messages[:-1]

        chat_text = task.get("full_text")
        if chat_text is None:
            chat_text = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

        tokenized = task.get("tokenized")
        if tokenized is None:
            try:
                tokenized = _tokenize_and_process(
                    chat_text,
                    tokenizer,
                    loss_on_all_tokens=train_on_input,
                )
            except ValueError as error:
                print(f"Skipping transcript due to tokenization error: {error}")
                return None

        total_tokens = int(tokenized["attention_mask"].sum().item())
        return {
            "input": prompt_messages,
            "output": assistant_message,
            "total_tokens": total_tokens,
            "full_text": chat_text,
            "tokenized": tokenized,
        }

    encoded_messages = formatter.encode(task)
    data = {"input": encoded_messages[0], "output": encoded_messages[1]}
    chat_messages = data["input"] + [data["output"]]
    task_text = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    try:
        tokenized = _tokenize_and_process(
            task_text,
            tokenizer,
            loss_on_all_tokens=train_on_input,
        )
    except ValueError as error:
        print(f"Skipping formatted task due to tokenization error: {error}")
        return None

    data["total_tokens"] = int(tokenized["attention_mask"].sum().item())
    data["full_text"] = task_text
    data["tokenized"] = tokenized
    return data


def get_test_time_train_data(
    original_task: Task, augmenters: List[Augmenter], n: int = 1, permute_n: int = 1, seed: int = 0
) -> List[Task]:
    rng = np.random.RandomState(seed)
    train_examples = original_task.train_examples.copy()
    initial_tasks = []
    N = len(train_examples)
    for i in range(len(train_examples)):
        examples = train_examples.copy()
        indices = set(range(N)) - {i}
        # we already remove i, so we need to remove n-1 more
        combs = list(itertools.combinations(indices, n - 1))
        combs = [indices - set(comb) for comb in combs]
        for comb in combs:
            initial_tasks.append(
                Task(name="", train_examples=[examples[j] for j in comb], test_example=examples[i])
            )

    augmented_tasks = []
    for augmenter in augmenters:
        for task in initial_tasks:
            task = augmenter.apply_to_task(task, to_input=True, to_output=True, rng=rng)
            # some augmentations increase shapes
            if not (task.max_height() <= 30 and task.max_width() <= 30):
                continue
            augmented_tasks.append(task)

    augmented_tasks = list(set(augmented_tasks + initial_tasks))

    color_and_permute_augmented_tasks = []

    for _ in range(permute_n):
        for task in augmented_tasks:
            if len(augmenters) != 0:
                new_task = PermuteColors().apply_to_task(task, to_input=True, to_output=True, rng=rng)
            else:
                new_task = task
            new_task = PermuteExamples().apply_to_task(
                new_task, rng=rng, to_input=True, to_output=True
            )
            color_and_permute_augmented_tasks.append(new_task)

    augmented_tasks = color_and_permute_augmented_tasks + augmented_tasks

    augmented_tasks = list(set(augmented_tasks))

    return augmented_tasks


def get_formatted_data(
    task: Task,
    augmenters: List[Augmenter],
    formatter: MessageRepresenter,
    tokenizer,
    leave_n: int = 1,
    permute_n: int = 1,
    seed: int = 0,
    max_tokens: int = 8192,
):

    train_data = get_test_time_train_data(
        task, augmenters, n=leave_n, permute_n=permute_n, seed=seed
    )

    formatted_data = []
    for task in train_data:
        formatted = format_and_filter(
            formatter,
            tokenizer,
            task,
            train_on_input=False,
        )
        if formatted is not None and formatted["total_tokens"] < max_tokens:
            formatted_data.append(formatted)

    return formatted_data


def process_task(
    task: Task,
    augmenters: List[Augmenter],
    formatter: MessageRepresenter,
    tokenizer,
    leave_n: List[int],
    permute_n: int = 1,
    Nmax: int = 250,
    seed: int = 0,
    *,
    code_mode: bool = False,
    transcripts: Optional[List[dict]] = None,
    max_tokens: int = 8192,
):
    rng = np.random.RandomState(seed)

    if code_mode:
        formatted_samples = []
        if transcripts is None:
            return formatted_samples

        for sample in transcripts:
            formatted = format_and_filter(
                formatter,
                tokenizer,
                sample,
                train_on_input=False,
                code_mode=True,
            )
            if formatted is not None and formatted["total_tokens"] < max_tokens:
                formatted_samples.append(formatted)

        if len(formatted_samples) > Nmax:
            rng.shuffle(formatted_samples)
            formatted_samples = formatted_samples[:Nmax]

        return formatted_samples

    train = []
    # Generate training data for each n in leave_n
    for n in leave_n:
        leave_n_train_data = get_formatted_data(
            task, augmenters, formatter, tokenizer, leave_n=n, permute_n=permute_n, seed=seed
        )
        train.extend(leave_n_train_data)

    # Shuffle and limit the total number of examples if needed
    if len(train) > Nmax:
        rng.shuffle(train)
        train = train[:Nmax]

    return train

def get_prompt(task: Task, system_message: str, self_edit_prompt: str):
    train_examples = task.serialize()['train']
    formatted_examples = ""

    for example in train_examples:
        # Format input grid
        input_grid = example['input']
        input_str = "Input:\n"
        for row in input_grid:
            input_str += " ".join(map(str, row)) + "\n"
        
        # Format output grid
        output_grid = example['output']
        output_str = "\nOutput:\n"
        for row in output_grid:
            output_str += " ".join(map(str, row)) + "\n"
        
        # Combine with separator
        formatted_examples += input_str + output_str + "\n"

    user_message = formatted_examples
    user_message = user_message + "------\n\n" + self_edit_prompt
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

def main(
    experiment_name,
    skip_repeated_configs,
    challenge_file,
    solution_file,
    model_name,
    n_tasks,
    n_self_edits_per_task,
    code_mode: bool = False,
    vllm_dtype: str = "float16",
    vllm_max_model_len: int = 4096,
    vllm_max_num_batched_tokens: int = 4096,
    vllm_gpu_memory_utilization: float = 0.6,
    vllm_enforce_eager: bool = True,
    vllm_tensor_parallel_size: int = 1,
):
    # lora config
    lora_config = LoraConfig(
        r=128, 
        lora_alpha=16,
        lora_dropout=0.00,
        bias="none",
        task_type="CAUSAL_LM", 
        target_modules=["q_proj", "v_proj", "gate_proj", "down_proj", "up_proj"]
    )

    # training config
    batch_size = 2
    gradient_accumulation_steps = 1
    lr_scheduler_type = "cosine"
    code_mode_learning_rate = 1e-4
    code_mode_epochs = 1

    if code_mode:
        representer = PythonSolverMessageRepresenter()
    else:
        standard_formatter = TextTaskRepresenter(
            example_representer=TextExampleRepresenter(
                io_sep=" -> ",
                input_header="",
                output_header="",
                output_footer="#",
                grid_representer=PythonListGridRepresenter(),
            )
        )

        representer = GPTTextMessageRepresenterV2(task_representer=standard_formatter)

    # Load tasks
    tasks = read_tasks_from_single_file(
        challenge_file=challenge_file, 
        solution_file=solution_file
    )

    # Setup tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Phase 1: Generate configs using self-edit model
    print("Phase 1: Generating configs using self-edit model...")
    llm_kwargs = {}
    if vllm_dtype:
        llm_kwargs["dtype"] = vllm_dtype
    if vllm_max_model_len and vllm_max_model_len > 0:
        llm_kwargs["max_model_len"] = vllm_max_model_len
    if vllm_max_num_batched_tokens and vllm_max_num_batched_tokens > 0:
        llm_kwargs["max_num_batched_tokens"] = vllm_max_num_batched_tokens
    if vllm_gpu_memory_utilization and vllm_gpu_memory_utilization > 0:
        llm_kwargs["gpu_memory_utilization"] = vllm_gpu_memory_utilization
    if vllm_tensor_parallel_size and vllm_tensor_parallel_size > 0:
        llm_kwargs["tensor_parallel_size"] = vllm_tensor_parallel_size
    llm_kwargs["enforce_eager"] = vllm_enforce_eager

    print(f"Initializing vLLM with kwargs: {llm_kwargs}")
    self_edit_model = LLM(model=model_name, **llm_kwargs)
    sampling_params = SamplingParams(
        max_tokens=128,
        temperature=0.8,
    )

    # Dictionary to store explored configs per task
    explored_configs = {}
    task_configs = {}  # Store full configs for each task

    for i in range(n_tasks):
        task = tasks[i]

        # Get the base task name (without -0 or -1 suffix) skip if it has -1 suffix
        base_task_name = task.name
        if base_task_name.endswith("-0"):
            base_task_name = base_task_name[:-2]
        if base_task_name.endswith("-1"):
            continue

        if code_mode:
            prompt_messages, _ = representer.encode(task)
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = get_prompt(task, system_message, self_edit_prompt)

        # Initialize config/program tracking for this task
        if base_task_name not in explored_configs:
            explored_configs[base_task_name] = set()
            task_configs[base_task_name] = []

        expected_output = None
        if getattr(task.test_example, "output", None) is not None:
            expected_output = np.array(task.test_example.output)

        while len(task_configs[base_task_name]) < n_self_edits_per_task:
            response = self_edit_model.generate(prompt_text, sampling_params=sampling_params)
            output = response[0].outputs[0]

            if code_mode:
                raw_response = output.text
                code = extract_solver_code(raw_response)

                if skip_repeated_configs and code and code in explored_configs[base_task_name]:
                    print(f"Skipping already explored program for task {base_task_name}")
                    continue

                if code:
                    explored_configs[base_task_name].add(code)

                execution_payload = {
                    "success": False,
                    "error_type": None,
                    "message": None,
                    "stdout": "",
                    "stderr": "",
                }

                reward = 0.0
                success = False
                predicted_output = None

                if code:
                    exec_result = run_solver(
                        code,
                        train_examples=task.train_examples,
                        test_input=task.test_example.input,
                    )
                    execution_payload.update(
                        {
                            "success": exec_result.success,
                            "error_type": exec_result.error_type,
                            "message": exec_result.message,
                            "stdout": exec_result.stdout,
                            "stderr": exec_result.stderr,
                        }
                    )
                    if exec_result.success and exec_result.output is not None:
                        predicted_output = exec_result.output.tolist()
                        execution_payload["output"] = predicted_output
                        if expected_output is not None and np.array_equal(
                            exec_result.output, expected_output
                        ):
                            success = True
                            reward = 1.0
                else:
                    execution_payload.update(
                        {
                            "error_type": "CodeExtractionError",
                            "message": "No executable code block found in response.",
                        }
                    )

                prompt_messages_copy = [dict(message) for message in prompt_messages]
                assistant_content = code if code else raw_response
                chat_messages = prompt_messages_copy + [
                    {"role": "assistant", "content": assistant_content}
                ]

                attempt_entry = {
                    "prompt_messages": prompt_messages_copy,
                    "prompt_text": prompt_text,
                    "raw_response": raw_response,
                    "code": code,
                    "token_ids": output.token_ids,
                    "execution": execution_payload,
                    "reward": reward,
                    "success": success,
                    "chat_messages": chat_messages,
                }

                if predicted_output is not None:
                    attempt_entry["predicted_output"] = predicted_output
                if expected_output is not None:
                    attempt_entry["target_output"] = expected_output.tolist()

                if success and code:
                    chat_text = tokenizer.apply_chat_template(
                        chat_messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    attempt_entry["full_text"] = chat_text
                    try:
                        tokenized = _tokenize_and_process(chat_text, tokenizer)
                    except ValueError as error:
                        print(
                            f"Failed to tokenize successful program for {base_task_name}: {error}"
                        )
                        attempt_entry["tokenized"] = None
                    else:
                        attempt_entry["tokenized"] = tokenized
                        attempt_entry["total_tokens"] = int(
                            tokenized["attention_mask"].sum().item()
                        )
                else:
                    attempt_entry["tokenized"] = None

                task_configs[base_task_name].append(attempt_entry)
                print(
                    f"New program for task {base_task_name}: success={success}, reward={reward}"
                )
            else:
                try:
                    config = json.loads(output.text)
                except json.JSONDecodeError:
                    continue

                config_key = (
                    ("data_generation", tuple(sorted(config["data_generation"].items()))),
                    ("training", tuple(sorted(config["training"].items()))),
                )

                if skip_repeated_configs and config_key in explored_configs[base_task_name]:
                    print(f"Skipping already explored config for task {base_task_name}")
                    continue

                explored_configs[base_task_name].add(config_key)
                task_configs[base_task_name].append(
                    {
                        "config": config,
                        "prompt": prompt_text,
                        "response": output.text,
                        "token_ids": output.token_ids,
                    }
                )
                print(f"New config for task {base_task_name}:", config)

    # Delete self-edit model to free memory
    del self_edit_model
    print("Phase 1 complete.")

    # Phase 2: Train models using generated configs
    print("\nPhase 2: Training models using generated configs...")
    
    # setup ttt 
    ttt = TTT(
        model_name=model_name,
        lora_config=lora_config
    )

    final_configs_and_indices = {}
    # Train models for each task using its configs
    for base_task_name, configs in task_configs.items():
        task = next(t for t in tasks if t.name.startswith(base_task_name))
        task_ttt = 0
        curr_task_configs = {}

        if code_mode:
            positive_attempts = [
                attempt
                for attempt in configs
                if attempt.get("success") and attempt.get("tokenized") is not None
            ]

            if not positive_attempts:
                print(f"No successful programs for {base_task_name}; skipping fine-tuning.")
                final_configs_and_indices[base_task_name] = {}
                continue

            positive_attempts.sort(key=lambda attempt: attempt.get("reward", 0), reverse=True)

            training_samples = [attempt["tokenized"] for attempt in positive_attempts]

            adapter_path = ttt.update_model(
                task_text_list=training_samples,
                output_dir=f"loras/self-edit/{experiment_name}/{base_task_name}/{task_ttt}",
                batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=code_mode_learning_rate,
                num_train_epochs=code_mode_epochs,
                lr_scheduler_type=lr_scheduler_type,
                loss_on_all_tokens=False,
            )

            sanitized_attempts = []
            for attempt in positive_attempts:
                attempt_summary = {
                    "reward": attempt.get("reward", 0),
                    "code": attempt.get("code"),
                    "prompt_messages": attempt.get("prompt_messages"),
                    "prompt_text": attempt.get("prompt_text"),
                    "raw_response": attempt.get("raw_response"),
                    "execution": attempt.get("execution"),
                    "chat_messages": attempt.get("chat_messages"),
                }
                if "predicted_output" in attempt:
                    attempt_summary["predicted_output"] = attempt["predicted_output"]
                if "target_output" in attempt:
                    attempt_summary["target_output"] = attempt["target_output"]
                if attempt.get("total_tokens") is not None:
                    attempt_summary["total_tokens"] = attempt["total_tokens"]
                sanitized_attempts.append(attempt_summary)

            curr_task_configs[task_ttt] = {
                "rewarded_programs": sanitized_attempts,
                "adapter_path": adapter_path,
                "learning_rate": code_mode_learning_rate,
                "num_train_epochs": code_mode_epochs,
            }
            final_configs_and_indices[base_task_name] = curr_task_configs
            task_ttt += 1
            continue

        for config_data in configs:
            config = config_data["config"]
            try:
                augmenters_to_apply = get_augmenters(
                    include_basic=config["data_generation"]["use_basic_augmentations"],
                    include_size=config["data_generation"]["use_size_augmentations"],
                    include_chain=config["data_generation"]["use_chain_augmentations"],
                    include_repeat=config["data_generation"]["use_repeat_augmentations"]
                )
            except Exception as e:
                print(f"Error getting augmenters for task {base_task_name}: {e}")
                augmenters_to_apply = get_augmenters(
                    include_basic=False,
                    include_size=False,
                    include_chain=False,
                    include_repeat=False
                )
                config["training"]["num_train_epochs"] = 0

            train_data = process_task(
                task=task,
                augmenters=augmenters_to_apply,
                formatter=representer,
                tokenizer=tokenizer,
                leave_n=[1,2],
                permute_n=1,
                Nmax=250,
                seed=0
            )

            if len(train_data) == 0:
                continue

            task_text_list = [data["full_text"] for data in train_data]

            # check if the keys "strategy" and "num_train_epochs" , "learning_rate" are in the config
            if "strategy" not in config["training"] or "num_train_epochs" not in config["training"] or "learning_rate" not in config["training"]:
                print(f"Skipping training for task {base_task_name} because the training config is not valid")
                config["training"]["num_train_epochs"] = 0
                config["training"]["learning_rate"] = 0
                config["training"]["strategy"] = "train_using_all_tokens"

            if config["training"]["strategy"] not in ["train_using_all_tokens", "train_using_output_tokens"]:
                print(f"Skipping training for task {base_task_name} because the training strategy is not valid")
                config["training"]["num_train_epochs"] = 0

            # if the number of steps is greater than 250, then we create a dummy lora
            if config["training"]["num_train_epochs"] * len(train_data) // 2 > 375:
                print(f"Skipping training for task {base_task_name} because the number of steps is greater than 375")
                config["training"]["num_train_epochs"] = 0

            adapter_path = ttt.update_model(
                task_text_list=task_text_list,
                output_dir=f"loras/self-edit/{experiment_name}/{base_task_name}/{task_ttt}",
                batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=config["training"]["learning_rate"],
                num_train_epochs=config["training"]["num_train_epochs"],
                lr_scheduler_type=lr_scheduler_type,
                loss_on_all_tokens=config["training"]["strategy"] == "train_using_all_tokens"
            )

            curr_task_configs[task_ttt] = config_data
            task_ttt += 1

        final_configs_and_indices[base_task_name] = curr_task_configs
    # Delete ttt to free memory
    del ttt

    # Save final configs and indices to file
    configs_file = os.path.join(f"loras/self-edit/{experiment_name}", "final_configs_and_indices.json")
    os.makedirs(os.path.dirname(configs_file), exist_ok=True)
    with open(configs_file, "w") as f:
        json.dump(final_configs_and_indices, f)
    
    print("Training complete. Final configs and indices saved to:", configs_file)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run self-edit training with specified parameters')
    parser.add_argument('--experiment_name', type=str, required=True,
                      help='Name of the experiment')
    parser.add_argument('--skip_repeated_configs', action='store_true',
                      help='Whether to skip repeated configs')
    parser.add_argument('--challenge_file', type=str, required=True,
                      help='Path to the challenge file')
    parser.add_argument('--solution_file', type=str, required=True,
                      help='Path to the solution file')
    parser.add_argument('--model_name', type=str, required=True,
                      help='Name of the model to use')
    parser.add_argument('--n_tasks', type=int, required=True,
                      help='Number of tasks to process')
    parser.add_argument('--n_self_edits_per_task', type=int, required=True,
                      help='Number of self-edits per task')
    parser.add_argument('--code_mode', action='store_true',
                      help='Format prompts for Python solver generation instead of grid outputs')
    parser.add_argument('--vllm_dtype', type=str, default='float16', choices=['auto', 'float16', 'bfloat16'],
                      help='Precision to use for vLLM weights (default: float16 for lower memory use).')
    parser.add_argument('--vllm_max_model_len', type=int, default=4096,
                      help='Upper bound on sequence length handed to vLLM (default: 4096 tokens).')
    parser.add_argument('--vllm_max_num_batched_tokens', type=int, default=4096,
                      help='Cap on the total tokens processed per batch to limit KV cache size.')
    parser.add_argument('--vllm_gpu_memory_utilization', type=float, default=0.6,
                      help='Fraction of GPU memory vLLM may reserve (default: 0.6).')
    parser.add_argument('--vllm_tensor_parallel_size', type=int, default=1,
                      help='Tensor parallel world size for vLLM (default: 1).')
    parser.add_argument('--vllm_enforce_eager', dest='vllm_enforce_eager', action='store_true',
                      help='Force eager execution to avoid torch.compile capture (default).')
    parser.add_argument('--no_vllm_enforce_eager', dest='vllm_enforce_eager', action='store_false',
                      help='Disable eager enforcement if you prefer torch.compile graphs.')
    parser.set_defaults(vllm_enforce_eager=True)

    args = parser.parse_args()

    main(
        experiment_name=args.experiment_name,
        skip_repeated_configs=args.skip_repeated_configs,
        challenge_file=args.challenge_file,
        solution_file=args.solution_file,
        model_name=args.model_name,
        n_tasks=args.n_tasks,
        n_self_edits_per_task=args.n_self_edits_per_task,
        code_mode=args.code_mode,
        vllm_dtype=args.vllm_dtype,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_max_num_batched_tokens=args.vllm_max_num_batched_tokens,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enforce_eager=args.vllm_enforce_eager,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
    )
    
   
 
    
