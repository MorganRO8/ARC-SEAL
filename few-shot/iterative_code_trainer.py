"""Iterative ARC self-improvement loop for code-generating solvers.

This script orchestrates the following steps:

1. Load a local instruction-tuned code LLM and format ARC tasks with the
   :class:`PythonSolverMessageRepresenter` prompts.
2. Sample multiple candidate programs per task and execute them safely via the
   sandboxed :func:`run_solver` helper.
3. Record every attempt in a replay buffer for later offline analysis.
4. Accumulate successful transcripts and periodically fine-tune a LoRA adapter
   using :class:`arclib.update_model.TTT` before swapping it back into the
   inference engine.

The implementation is intentionally modular so experiments can adjust the
sampling, evaluation, and training cadence through CLI arguments.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from peft import LoraConfig

from arclib.arc import Task, read_tasks_from_file, read_tasks_from_folder, read_tasks_from_single_file
from arclib.messagers import PythonSolverMessageRepresenter
from arclib.update_model import TTT
from utils.python_executor import extract_solver_code, run_solver
from utils.chat_template import detect_thinking_support


DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]


@dataclass
class AttemptLog:
    """Serializable structure describing a single program attempt."""

    task_name: str
    prompt_messages: Sequence[Mapping[str, str]]
    prompt_text: str
    raw_response: str
    code: str
    success: bool
    reward: float
    execution: Dict[str, Optional[str]]
    predicted_output: Optional[List[List[int]]]
    target_output: Optional[List[List[int]]]

    def to_json(self) -> str:
        payload = asdict(self)
        payload["prompt_messages"] = list(self.prompt_messages)
        return json.dumps(payload, ensure_ascii=False)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative ARC self-improvement loop")
    parser.add_argument("--model", required=True, help="Base instruction-tuned code model path")
    parser.add_argument(
        "--tasks",
        required=True,
        help="Path to ARC tasks (folder or JSON file)",
    )
    parser.add_argument(
        "--solutions",
        default=None,
        help="Optional JSON file containing ground-truth solutions for challenge splits.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory used to persist adapters and replay buffers.",
    )
    parser.add_argument(
        "--replay-buffer",
        default=None,
        help="Path to a JSONL replay buffer file (defaults to <output-dir>/replay.jsonl).",
    )
    parser.add_argument(
        "--tasks-per-iteration",
        type=int,
        default=8,
        help="Number of ARC tasks sampled per iteration.",
    )
    parser.add_argument(
        "--generations-per-task",
        type=int,
        default=4,
        help="Number of program samples generated for each task in an iteration.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum training iterations to run (defaults to full task list).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for program generation.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling value.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate per attempt.",
    )
    parser.add_argument(
        "--min-successes-for-update",
        type=int,
        default=4,
        help="Number of successful programs required before triggering a LoRA update.",
    )
    parser.add_argument(
        "--update-frequency",
        type=int,
        default=1,
        help="Perform a LoRA update every N iterations once enough successes accumulate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="LoRA fine-tuning batch size.",
    )
    parser.add_argument(
        "--grad-accumulation",
        type=int,
        default=1,
        help="Gradient accumulation steps for LoRA updates.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate used for LoRA fine-tuning on code transcripts.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of SFT epochs for each LoRA update.",
    )
    parser.add_argument(
        "--lr-scheduler",
        default="cosine",
        help="Scheduler type passed to the Trainer (default: cosine).",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=128,
        help="LoRA rank (r parameter).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling factor.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="LoRA dropout probability.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for task shuffling.",
    )
    return parser.parse_args(argv)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_tasks(path: str, solutions: Optional[str] = None) -> List[Task]:
    path_obj = Path(path)
    if path_obj.is_dir():
        return read_tasks_from_folder(str(path_obj))
    if path_obj.suffix == ".json" and solutions:
        return read_tasks_from_single_file(str(path_obj), solution_file=solutions)
    if path_obj.suffix == ".json":
        return read_tasks_from_file(str(path_obj))
    raise ValueError(f"Unsupported task path: {path}")


def format_prompt(
    tokenizer,
    representer: PythonSolverMessageRepresenter,
    task: Task,
    *,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Mapping[str, str]], str]:
    prompt_messages, _ = representer.encode(task)
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        **dict(chat_template_kwargs or {}),
    )
    return prompt_messages, prompt_text


def decode_model_outputs(
    tokenizer,
    prompt_text: str,
    sequences: torch.Tensor,
) -> List[str]:
    prompt_length = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[-1]
    decoded: List[str] = []
    for sequence in sequences:
        continuation = sequence[prompt_length:].tolist()
        decoded.append(tokenizer.decode(continuation, skip_special_tokens=True).strip())
    return decoded


def generate_programs(
    model,
    tokenizer,
    prompt_text: str,
    *,
    generations_per_task: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    device = next(model.parameters()).device
    encoded = tokenizer(prompt_text, return_tensors="pt").to(device)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature,
        "top_p": top_p,
        "num_return_sequences": generations_per_task,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }

    with torch.no_grad():
        generations = model.generate(**encoded, **generation_kwargs)

    sequences = generations.sequences if hasattr(generations, "sequences") else generations
    return decode_model_outputs(tokenizer, prompt_text, sequences)


def compute_reward(result, expected: Optional[np.ndarray]) -> Tuple[bool, float, Optional[List[List[int]]]]:
    if not result.success or expected is None:
        return False, 0.0, None

    predicted = result.output
    if predicted is None:
        return False, 0.0, None

    success = np.array_equal(predicted, expected)
    reward = 1.0 if success else 0.0
    return success, reward, predicted.tolist()


def _tokenize_chat_transcript(text: str, tokenizer) -> Mapping[str, torch.Tensor]:
    outputs = tokenizer(
        text,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = outputs["input_ids"].squeeze(0).to(torch.long).cpu()
    attention_mask = outputs["attention_mask"].squeeze(0).to(torch.long).cpu()
    labels = input_ids.clone()

    ids_list = input_ids.tolist()
    special_indices = []
    for idx in range(len(ids_list) - 1):
        if ids_list[idx] == 128007 and ids_list[idx + 1] == 271:
            special_indices.append(idx + 1)

    if not special_indices:
        raise ValueError("Assistant header token sequence not found in chat transcript.")

    if len(special_indices) >= 2:
        boundary = special_indices[-2]
    else:
        boundary = special_indices[-1]

    labels[: boundary + 1] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def initialize_ttt(args: argparse.Namespace) -> TTT:
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=DEFAULT_LORA_TARGETS,
    )
    return TTT(model_name=args.model, lora_config=lora_config)


def maybe_run_update(
    ttt: TTT,
    positive_samples: List[Mapping[str, torch.Tensor]],
    args: argparse.Namespace,
    iteration: int,
    output_root: Path,
) -> Optional[Path]:
    if len(positive_samples) < args.min_successes_for_update:
        return None

    if (iteration + 1) % args.update_frequency != 0:
        return None

    adapter_dir = output_root / "adapters" / f"iter-{iteration+1:04d}"
    ensure_directory(adapter_dir)

    adapter_path = ttt.update_model(
        task_text_list=positive_samples,
        output_dir=str(adapter_dir),
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        lr_scheduler_type=args.lr_scheduler,
        loss_on_all_tokens=False,
    )

    adapter_name = f"iter_{iteration+1:04d}"
    if adapter_name in ttt.model.peft_config:
        ttt.model.delete_adapter(adapter_name)
    ttt.model.load_adapter(adapter_path, adapter_name=adapter_name, is_trainable=False)
    ttt.model.set_adapter(adapter_name)
    ttt.model.eval()

    return Path(adapter_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)

    output_root = Path(args.output_dir)
    ensure_directory(output_root)

    replay_path = Path(args.replay_buffer) if args.replay_buffer else output_root / "replay.jsonl"
    replay_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(args.tasks, solutions=args.solutions)
    if not tasks:
        raise ValueError("No tasks found for the provided path.")

    random.shuffle(tasks)

    ttt = initialize_ttt(args)
    tokenizer = ttt.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    thinking_config = detect_thinking_support(tokenizer)
    chat_template_kwargs = dict(thinking_config.apply_chat_kwargs)
    if thinking_config.enabled:
        reason = thinking_config.reason or "detected thinking tokens"
        print(f"Enabling thinking support in chat template ({reason}).")
    else:
        chat_template_kwargs = {}

    representer = PythonSolverMessageRepresenter()

    total_attempts = 0
    positive_samples: List[Mapping[str, torch.Tensor]] = []
    adapters: List[Path] = []

    max_iterations = args.max_iterations or math.ceil(len(tasks) / max(1, args.tasks_per_iteration))

    with replay_path.open("a", encoding="utf-8") as replay_file:
        for iteration in range(max_iterations):
            start_idx = iteration * args.tasks_per_iteration
            end_idx = start_idx + args.tasks_per_iteration
            if start_idx >= len(tasks):
                break

            batch_tasks = tasks[start_idx:end_idx]
            tqdm_desc = f"Iteration {iteration+1}/{max_iterations}"
            for task in tqdm(batch_tasks, desc=tqdm_desc):
                prompt_messages, prompt_text = format_prompt(
                    tokenizer,
                    representer,
                    task,
                    chat_template_kwargs=chat_template_kwargs,
                )

                try:
                    generations = generate_programs(
                        ttt.model,
                        tokenizer,
                        prompt_text,
                        generations_per_task=args.generations_per_task,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                except RuntimeError as error:
                    print(f"Generation failed for task {task.name}: {error}")
                    continue

                expected_output = None
                if getattr(task.test_example, "output", None) is not None:
                    expected_output = np.array(task.test_example.output)

                for raw_response in generations:
                    total_attempts += 1
                    code = extract_solver_code(raw_response)
                    exec_payload: Dict[str, Optional[str]] = {
                        "success": None,
                        "error_type": None,
                        "message": None,
                        "stdout": None,
                        "stderr": None,
                    }

                    success = False
                    reward = 0.0
                    predicted_output: Optional[List[List[int]]] = None

                    if code:
                        result = run_solver(
                            code,
                            train_examples=task.train_examples,
                            test_input=task.test_example.input,
                        )
                        exec_payload.update(
                            {
                                "success": str(result.success),
                                "error_type": result.error_type,
                                "message": result.message,
                                "stdout": result.stdout,
                                "stderr": result.stderr,
                            }
                        )

                        success, reward, predicted_output = compute_reward(result, expected_output)

                    attempt_record = AttemptLog(
                        task_name=task.name,
                        prompt_messages=prompt_messages,
                        prompt_text=prompt_text,
                        raw_response=raw_response,
                        code=code,
                        success=success,
                        reward=reward,
                        execution=exec_payload,
                        predicted_output=predicted_output,
                        target_output=expected_output.tolist() if expected_output is not None else None,
                    )
                    replay_file.write(attempt_record.to_json() + "\n")
                    replay_file.flush()

                    if success and code:
                        chat_messages = list(prompt_messages) + [
                            {"role": "assistant", "content": code}
                        ]
                        chat_text = tokenizer.apply_chat_template(
                            chat_messages,
                            tokenize=False,
                            add_generation_prompt=False,
                            **dict(chat_template_kwargs or {}),
                        )
                        try:
                            tokenized = _tokenize_chat_transcript(chat_text, tokenizer)
                        except ValueError as error:
                            print(f"Tokenization failed for {task.name}: {error}")
                        else:
                            positive_samples.append(tokenized)

            adapter_path = maybe_run_update(
                ttt,
                positive_samples,
                args,
                iteration,
                output_root,
            )
            if adapter_path is not None:
                adapters.append(adapter_path)
                print(f"Updated adapter saved to {adapter_path}")

    print("\nSummary:")
    print(f"Total tasks processed: {min(len(tasks), max_iterations * args.tasks_per_iteration)}")
    print(f"Total attempts: {total_attempts}")
    print(f"Successful programs collected: {len(positive_samples)}")
    if adapters:
        print("Adapters saved:")
        for adapter in adapters:
            print(f" - {adapter}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main(sys.argv[1:])

