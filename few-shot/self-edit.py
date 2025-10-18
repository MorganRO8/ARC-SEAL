import argparse
import glob
import json
import os
import re
import textwrap
from collections import Counter
import hashlib
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from builtins import input
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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

from utils.code_formatting import FormattingResult, try_fix_indentation
from utils.output_size_inference import infer_output_shape
from utils.prompts import self_edit_prompt, system_message
from utils.chat_template import detect_thinking_support
from utils.python_executor import SolverResult, extract_solver_code, run_solver
from utils.solver_helpers import HelperLibrary, get_helper_library


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


def _pad_or_crop_to_shape(
    array: np.ndarray, target_shape: Tuple[int, int]
) -> np.ndarray:
    """Pad or crop ``array`` to match ``target_shape`` using the mode color."""

    target_rows, target_cols = target_shape
    if array.size == 0:
        fill_value = 0
    else:
        flat = array.flatten()
        try:
            counts = np.bincount(flat)
            fill_value = int(np.argmax(counts))
        except ValueError:
            fill_value = int(flat[0])

    result = np.full((target_rows, target_cols), fill_value, dtype=array.dtype)
    rows = min(array.shape[0], target_rows)
    cols = min(array.shape[1], target_cols)
    result[:rows, :cols] = array[:rows, :cols]
    return result


def score_grid_prediction(
    predicted_output: Optional[np.ndarray],
    expected_output: Optional[np.ndarray],
    *,
    shape_hint: Optional[Tuple[int, int]] = None,
    reference_input: Optional[np.ndarray] = None,
) -> Tuple[
    float,
    Optional[int],
    Optional[int],
    Optional[str],
    Optional[np.ndarray],
    Dict[str, Any],
]:
    """Compute a normalized partial-credit reward for a predicted grid.

    Returns ``(reward, correct_cells, total_cells, reason, normalized_output,
    details)``. ``normalized_output`` is the grid actually compared against the
    target after any padding/cropping. ``reason`` is ``None`` when a comparison
    succeeded. ``details`` contains metadata about the chosen reference grid and
    intermediate statistics used to compute the reward.
    """

    if expected_output is None:
        return 0.0, None, None, "missing_target", None, {}
    if predicted_output is None:
        return 0.0, None, None, "missing_prediction", None, {}

    try:
        predicted_array = np.asarray(predicted_output, dtype=int)
    except (TypeError, ValueError):
        return 0.0, None, None, "non_integer_prediction", None, {}

    try:
        expected_array = np.asarray(expected_output, dtype=int)
    except (TypeError, ValueError):
        return 0.0, None, None, "invalid_target", None, {}

    target_shape = expected_array.shape
    normalized_array = predicted_array
    adjustment_reason: Optional[str] = None

    if predicted_array.shape != target_shape:
        if (
            shape_hint is not None
            and len(target_shape) >= 2
            and (int(target_shape[0]), int(target_shape[1])) == tuple(shape_hint)
        ):
            normalized_array = _pad_or_crop_to_shape(predicted_array, tuple(shape_hint))
            adjustment_reason = (
                "shape_adjusted "
                f"{predicted_array.shape[:2]}->{tuple(shape_hint)}"
            )
        else:
            return 0.0, 0, int(expected_array.size), "shape_mismatch", predicted_array, {}

    total_cells = int(expected_array.size)
    if total_cells == 0:
        return 0.0, 0, 0, "empty_target", normalized_array, {}

    # Build candidate reference grids: blank grid and (optionally) the input grid.
    reference_candidates: List[Tuple[str, np.ndarray, int, Dict[str, Any]]] = []
    blank_reference = np.zeros(target_shape, dtype=int)
    blank_diff = int(np.count_nonzero(blank_reference != expected_array))
    reference_candidates.append(("blank", blank_reference, blank_diff, {"adjusted": False}))

    if reference_input is not None:
        try:
            input_array = np.asarray(reference_input, dtype=int)
        except (TypeError, ValueError):
            input_array = None
        if input_array is not None:
            input_adjusted = False
            if input_array.shape != target_shape:
                input_array = _pad_or_crop_to_shape(input_array, target_shape)
                input_adjusted = True
            input_diff = int(np.count_nonzero(input_array != expected_array))
            reference_candidates.append(
                (
                    "input",
                    input_array,
                    input_diff,
                    {"adjusted": input_adjusted},
                )
            )

    # Prefer the candidate with the fewest differing cells; break ties in favour
    # of the input grid because it conveys more structure than a blank canvas.
    reference_name, reference_array, reference_diff, reference_meta = min(
        reference_candidates,
        key=lambda item: (item[2], 0 if item[0] == "input" else 1),
    )

    change_mask = expected_array != reference_array
    changed_cells = int(np.count_nonzero(change_mask))

    matches = normalized_array == expected_array
    overall_matches = int(np.count_nonzero(matches))
    exact_match = overall_matches == total_cells

    mismatched_unchanged = int(np.count_nonzero(~change_mask & ~matches))
    correct_changed = int(np.count_nonzero(change_mask & matches))
    incorrect_cells = int(total_cells - overall_matches)

    if changed_cells > 0:
        reward = (
            float(correct_changed) / float(changed_cells)
            - float(incorrect_cells) / float(total_cells)
        )
    else:
        # No cells differ from the reference; fall back to overall accuracy.
        reward = 1.0 - float(incorrect_cells) / float(total_cells)

    reward = max(0.0, min(1.0, reward))

    # For reporting we still highlight the cells that required intervention.
    correct_cells = correct_changed if changed_cells > 0 else overall_matches
    total_considered = changed_cells if changed_cells > 0 else total_cells

    details: Dict[str, Any] = {
        "reference_type": reference_name,
        "reference_difference_cells": int(reference_diff),
        "reference_input_adjusted": bool(reference_meta.get("adjusted", False)),
        "cells_requiring_change": int(changed_cells),
        "cells_unchanged": int(total_cells - changed_cells),
        "correct_changed_cells": int(correct_changed),
        "mismatched_unchanged_cells": int(mismatched_unchanged),
        "overall_matches": int(overall_matches),
        "incorrect_cells": int(incorrect_cells),
        "reward_changed_component": (
            float(correct_changed) / float(changed_cells)
            if changed_cells > 0
            else None
        ),
        "reward_penalty_component": float(incorrect_cells) / float(total_cells),
        "exact_match": bool(exact_match),
    }

    if exact_match:
        reward = 1.0
        details["reward_penalty_component"] = 0.0
        if changed_cells > 0:
            details["reward_changed_component"] = 1.0

    return reward, correct_cells, total_considered, adjustment_reason, normalized_array, details


_SUSPICIOUS_LOOP_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bwhile\s+True\b"), "an unconditional `while True` loop"),
    (re.compile(r"\bwhile\s+1\b"), "an unconditional `while 1` loop"),
    (
        re.compile(r"itertools\s*\.\s*count\s*\("),
        "`itertools.count` which produces an infinite iterator",
    ),
]


def detect_unbounded_control_flow(code: str) -> Optional[str]:
    """Return a human-readable reason if ``code`` contains obvious infinite loops."""

    for pattern, description in _SUSPICIOUS_LOOP_PATTERNS:
        if pattern.search(code):
            return description
    return None


_INDENTATION_PHRASES = (
    "indentationerror",
    "unexpected indent",
    "expected an indented block",
    "indentation error",
    "unindent does not match any outer indentation level",
)


def _needs_indentation_fix(result: Optional[SolverResult]) -> bool:
    if result is None or result.success:
        return False

    error_type = (result.error_type or "").lower()
    message = (result.message or "").lower()

    if "indent" in error_type:
        return True

    return any(phrase in message for phrase in _INDENTATION_PHRASES)


def _truncate_block(value: str, limit: int = 1200) -> str:
    if len(value) <= limit:
        return value
    ellipsis = "\n… (truncated)"
    return value[: max(0, limit - len(ellipsis))] + ellipsis


def _format_labeled_block(label: str, content: str) -> str:
    if not content:
        return ""
    indented = textwrap.indent(content.strip(), prefix="    ")
    return f"{label}:\n{indented}"


def build_error_feedback(
    exec_result: Optional[SolverResult],
    *,
    reward_reason: Optional[str] = None,
    formatting_result: Optional[FormattingResult] = None,
    retries_remaining: int = 0,
) -> str:
    """Create a user-turn message summarizing an execution failure."""

    lines: List[str] = [
        "The previous Python program failed to produce a valid solution.",
    ]

    if exec_result is not None:
        if exec_result.error_type:
            lines.append(f"Error type: {exec_result.error_type}")
        if exec_result.message:
            lines.append(f"Error message: {exec_result.message}")
        if exec_result.exit_code is not None:
            lines.append(f"Exit code: {exec_result.exit_code}")

        if exec_result.stdout:
            lines.append(
                _format_labeled_block(
                    "Stdout",
                    _truncate_block(exec_result.stdout.strip()),
                )
            )
        if exec_result.stderr:
            lines.append(
                _format_labeled_block(
                    "Stderr",
                    _truncate_block(exec_result.stderr.strip()),
                )
            )

    if reward_reason:
        lines.append(f"Grid evaluation issue: {reward_reason}")

    if formatting_result is not None:
        if formatting_result.error:
            lines.append(
                f"Auto-formatting with {formatting_result.tool} failed: {formatting_result.error}."
            )
        elif formatting_result.changed:
            lines.append(
                f"Auto-formatting with {formatting_result.tool} adjusted the indentation before rerunning, but the solver still failed."
            )

    guidance_lines = [
        "Please fix the program and output only the corrected Python source code defining solve().",
        "Ensure the indentation is valid and the function returns the required output grid.",
    ]

    if retries_remaining <= 0:
        guidance_lines.append(
            "This is the final opportunity for this task, so double-check the entire solution before replying."
        )

    lines.append("")
    lines.extend(guidance_lines)

    return "\n".join(filter(None, lines)).strip()


def _serialize_solver_result(result: Optional[SolverResult]) -> Dict[str, Any]:
    if result is None:
        return {}

    payload: Dict[str, Any] = {
        "success": result.success,
        "error_type": result.error_type,
        "message": result.message,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    if result.exit_code is not None:
        payload["exit_code"] = result.exit_code
    if result.output is not None:
        if isinstance(result.output, np.ndarray):
            payload["output"] = result.output.tolist()
        else:
            payload["output"] = result.output
    payload["helper_error"] = bool(result.helper_error)
    if result.helper_exception_type is not None:
        payload["helper_exception_type"] = result.helper_exception_type
    if result.helper_function is not None:
        payload["helper_function"] = result.helper_function
    return payload


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
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
):
    if code_mode:
        chat_messages = task.get("chat_messages")
        if chat_messages is None:
            raise ValueError("Code-mode transcripts must include 'chat_messages'.")

        assistant_message = chat_messages[-1]
        prompt_messages = chat_messages[:-1]

        extra_kwargs = dict(chat_template_kwargs or {})

        chat_text = task.get("full_text")
        if chat_text is None:
            chat_text = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=False,
                **extra_kwargs,
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
    extra_kwargs = dict(chat_template_kwargs or {})
    task_text = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
        **extra_kwargs,
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
    *,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
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
            chat_template_kwargs=chat_template_kwargs,
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
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
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
                chat_template_kwargs=chat_template_kwargs,
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
            task,
            augmenters,
            formatter,
            tokenizer,
            leave_n=n,
            permute_n=permute_n,
            seed=seed,
            chat_template_kwargs=chat_template_kwargs,
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
    solver_timeout: float = 5.0,
    solver_cpu_time_limit: Optional[float] = 5.0,
    solver_memory_limit_mb: Optional[int] = 512,
    code_mode_retry_attempts: int = 2,
    vllm_dtype: str = "float16",
    vllm_max_model_len: int = 4096,
    vllm_max_num_batched_tokens: int = 4096,
    vllm_gpu_memory_utilization: float = 0.6,
    vllm_enforce_eager: bool = True,
    vllm_tensor_parallel_size: int = 1,
    include_solver_helpers: bool = False,
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

    helper_library: Optional[HelperLibrary] = None
    helper_prompt_overview: Optional[str] = None
    helper_prompt_api: Optional[str] = None
    helper_prompt_source: Optional[str] = None
    helper_namespace: str = "ARC_HELPERS"
    helper_source_digest: Optional[str] = None

    if code_mode and include_solver_helpers:
        helper_library = get_helper_library()
        helper_prompt_overview = helper_library.prompt_overview
        helper_prompt_api = helper_library.api_reference
        helper_prompt_source = helper_library.source
        helper_namespace = helper_library.namespace
        helper_source_digest = hashlib.sha256(helper_library.source.encode("utf-8")).hexdigest()
        print(
            "Including solver helper library under namespace",
            helper_namespace,
        )

    solver_timeout = max(float(solver_timeout), 0.1)
    if solver_cpu_time_limit is None:
        solver_cpu_time_limit_s: Optional[int] = None
    elif solver_cpu_time_limit <= 0:
        solver_cpu_time_limit_s = None
    else:
        solver_cpu_time_limit_s = max(int(round(solver_cpu_time_limit)), 1)

    if solver_memory_limit_mb is None or solver_memory_limit_mb <= 0:
        solver_memory_limit_value: Optional[int] = None
    else:
        solver_memory_limit_value = int(solver_memory_limit_mb)

    retry_budget = max(int(code_mode_retry_attempts), 0)

    execution_limits_payload = {
        "timeout": solver_timeout,
        "cpu_time_limit": solver_cpu_time_limit_s,
        "memory_limit_mb": solver_memory_limit_value,
        "retry_budget": retry_budget if code_mode else 0,
    }

    # Load tasks
    tasks = read_tasks_from_single_file(
        challenge_file=challenge_file,
        solution_file=solution_file
    )

    # Setup tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    thinking_config = detect_thinking_support(tokenizer)
    chat_template_kwargs = dict(thinking_config.apply_chat_kwargs)
    if thinking_config.enabled:
        reason = thinking_config.reason or "detected thinking tokens"
        print(f"Enabling thinking support in chat template ({reason}).")
    else:
        chat_template_kwargs = {}

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

    total_tasks = min(n_tasks, len(tasks))
    if n_tasks > len(tasks):
        print(
            f"Requested {n_tasks} tasks but only {len(tasks)} available; limiting to {total_tasks}."
        )

    progress_bar = tqdm(
        total=total_tasks,
        desc="Self-edit tasks",
        unit="task",
        position=2,
        leave=True,
        dynamic_ncols=True,
    )
    progress_bar.refresh()
    try:
        for i in range(total_tasks):
            task = tasks[i]

            # Get the base task name (without -0 or -1 suffix) skip if it has -1 suffix
            base_task_name = task.name
            if base_task_name.endswith("-0"):
                base_task_name = base_task_name[:-2]
            if base_task_name.endswith("-1"):
                progress_bar.update(1)
                continue

            context_budget: Optional[int] = None
            max_model_len: Optional[int] = None
            size_hint: Optional[Tuple[int, int]] = None
            size_metadata: Optional[Dict[str, Any]] = None
            base_prompt_messages: Optional[List[Dict[str, Any]]] = None

            if code_mode:
                size_hint, size_metadata = infer_output_shape(task)
                if size_hint is not None:
                    print(
                        "Inferred output size for",
                        f"{base_task_name}: {size_hint[0]}×{size_hint[1]}",
                        f"(method={size_metadata.get('method') if size_metadata else 'unknown'})",
                    )
                encoded_messages, _ = representer.encode(
                    task,
                    size_inference=(size_hint, size_metadata),
                    execution_limits=execution_limits_payload,
                    helper_overview=helper_prompt_overview,
                    helper_api_reference=helper_prompt_api,
                    helper_source=helper_prompt_source,
                    helper_namespace=helper_namespace,
                )
                base_prompt_messages = [deepcopy(message) for message in encoded_messages]

                def _prepare_prompt(messages: List[Dict[str, Any]]) -> Tuple[str, int]:
                    prompt_text_local = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        **chat_template_kwargs,
                    )
                    prompt_tokens = tokenizer(
                        prompt_text_local,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )
                    prompt_length = int(prompt_tokens["input_ids"].shape[-1])
                    return prompt_text_local, prompt_length

                try:
                    base_prompt_text, initial_prompt_length = _prepare_prompt(base_prompt_messages)
                except ValueError as error:
                    print(
                        "Failed to tokenize prompt for task"
                        f" {base_task_name}: {error}"
                    )
                    progress_bar.update(1)
                    continue

                if vllm_max_model_len and vllm_max_model_len > 0:
                    max_model_len = vllm_max_model_len
                else:
                    max_model_len = getattr(
                        getattr(self_edit_model, "llm_engine", None),
                        "max_model_len",
                        None,
                    )
                    if max_model_len is None and getattr(
                        getattr(self_edit_model, "llm_engine", None),
                        "model_config",
                        None,
                    ) is not None:
                        max_model_len = getattr(
                            self_edit_model.llm_engine.model_config,
                            "max_model_len",
                            None,
                        )

                if max_model_len is not None:
                    context_budget = max_model_len - (sampling_params.max_tokens or 0)
                    if context_budget <= 0:
                        print(
                            "No room for prompt tokens with current generation "
                            f"budget (max_model_len={max_model_len}, "
                            f"max_tokens={sampling_params.max_tokens})."
                        )
                        progress_bar.update(1)
                        continue

                    if initial_prompt_length > context_budget:
                        print(
                            f"Skipping task {base_task_name}: prompt requires "
                            f"{initial_prompt_length} tokens but only "
                            f"{context_budget} are available."
                        )
                        progress_bar.update(1)
                        continue

                def _run_code_mode_attempt() -> Tuple[Optional[Dict[str, Any]], bool, bool]:
                    """Run a single code-mode attempt with feedback-driven retries.

                    Returns ``(attempt_entry, skipped, abort_task)`` where ``attempt_entry``
                    is populated when a program was executed, ``skipped`` is ``True`` when the
                    generation was discarded (e.g., duplicate program), and ``abort_task`` is
                    ``True`` when the task should stop generating further attempts entirely.
                    """

                    if base_prompt_messages is None:
                        return None, False, True

                    conversation_messages = [
                        deepcopy(message) for message in base_prompt_messages
                    ]
                    retries_remaining = retry_budget
                    attempt_retry_history: List[Dict[str, Any]] = []

                    local_reward: float = 0.0
                    local_success = False
                    local_correct_cells: Optional[int] = None
                    local_total_cells: Optional[int] = None
                    local_reward_reason: Optional[str] = None
                    local_predicted_output = None
                    local_normalized_output = None
                    local_rejection_reason: Optional[str] = None
                    local_execution_payload: Dict[str, Any] = {}
                    local_formatting_result: Optional[FormattingResult] = None
                    local_exec_result: Optional[SolverResult] = None
                    local_final_code: Optional[str] = None
                    local_raw_code: Optional[str] = None
                    local_raw_response: str = ""
                    local_token_ids = None

                    abort_task = False
                    skip_attempt = False

                    while True:
                        try:
                            prompt_text_current, prompt_token_length = _prepare_prompt(
                                conversation_messages
                            )
                        except ValueError as error:
                            print(
                                "Failed to tokenize prompt for task",
                                f" {base_task_name}: {error}",
                            )
                            abort_task = True
                            break

                        if (
                            context_budget is not None
                            and prompt_token_length > context_budget
                        ):
                            print(
                                f"Skipping task {base_task_name}: prompt requires "
                                f"{prompt_token_length} tokens but only "
                                f"{context_budget} are available."
                            )
                            abort_task = True
                            break

                        try:
                            response = self_edit_model.generate(
                                prompt_text_current, sampling_params=sampling_params
                            )
                        except ValueError as error:
                            if "maximum model length" in str(error).lower():
                                print(
                                    f"Skipping task {base_task_name} due to context "
                                    f"overflow: {error}"
                                )
                                abort_task = True
                                break
                            raise

                        output = response[0].outputs[0]
                        local_token_ids = output.token_ids
                        raw_response = output.text or ""
                        local_raw_response = raw_response
                        code = extract_solver_code(raw_response)
                        local_raw_code = code

                        if (
                            skip_repeated_configs
                            and code
                            and code in explored_configs[base_task_name]
                        ):
                            print(
                                f"Skipping already explored program for task {base_task_name}"
                            )
                            skip_attempt = True
                            break

                        if code:
                            explored_configs[base_task_name].add(code)

                        assistant_content = code if code else raw_response
                        conversation_messages.append(
                            {"role": "assistant", "content": assistant_content}
                        )

                        execution_payload: Dict[str, Any] = {
                            "success": False,
                            "error_type": None,
                            "message": None,
                            "stdout": "",
                            "stderr": "",
                            "exit_code": None,
                            "timeout_s": solver_timeout,
                            "cpu_time_limit_s": solver_cpu_time_limit_s,
                            "memory_limit_mb": solver_memory_limit_value,
                        }
                        helper_info_payload: Dict[str, Any] = {
                            "included": helper_library is not None,
                            "helper_error": False,
                            "helper_exception_type": None,
                            "helper_function": None,
                        }
                        if helper_library is not None:
                            helper_info_payload.update(
                                {
                                    "namespace": helper_library.namespace,
                                    "source_sha256": helper_source_digest,
                                }
                            )
                        execution_payload["solver_helpers"] = helper_info_payload
                        execution_attempts: List[Dict[str, Any]] = []

                        local_reward = 0.0
                        local_success = False
                        local_correct_cells = None
                        local_total_cells = None
                        local_reward_reason = None
                        local_reward_details: Dict[str, Any] = {}
                        local_predicted_output = None
                        local_normalized_output = None
                        local_rejection_reason = None
                        local_exec_result = None
                        local_formatting_result = None
                        local_final_code = code

                        if code:
                            local_rejection_reason = detect_unbounded_control_flow(code)

                        if code and local_rejection_reason is None:
                            local_exec_result = run_solver(
                                local_final_code or "",
                                train_examples=task.train_examples,
                                test_input=task.test_example.input,
                                timeout=solver_timeout,
                                memory_limit_mb=solver_memory_limit_value,
                                cpu_time_limit_s=solver_cpu_time_limit_s,
                                helper_library=helper_library,
                            )
                            execution_attempts.append(
                                {
                                    "origin": "model",
                                    "code": local_final_code,
                                    "result": _serialize_solver_result(local_exec_result),
                                }
                            )
                            if local_exec_result.exit_code is not None:
                                execution_payload["exit_code"] = local_exec_result.exit_code
                            if helper_library is not None:
                                helper_info_payload.update(
                                    {
                                        "helper_error": bool(local_exec_result.helper_error),
                                        "helper_exception_type": local_exec_result.helper_exception_type,
                                        "helper_function": local_exec_result.helper_function,
                                    }
                                )

                            if _needs_indentation_fix(local_exec_result):
                                local_formatting_result = try_fix_indentation(
                                    local_final_code or ""
                                )
                                formatted_code = local_formatting_result.formatted_code
                                if (
                                    formatted_code
                                    and formatted_code != local_final_code
                                ):
                                    local_final_code = formatted_code
                                    local_exec_result = run_solver(
                                        local_final_code,
                                        train_examples=task.train_examples,
                                        test_input=task.test_example.input,
                                        timeout=solver_timeout,
                                        memory_limit_mb=solver_memory_limit_value,
                                        cpu_time_limit_s=solver_cpu_time_limit_s,
                                        helper_library=helper_library,
                                    )
                                    execution_attempts.append(
                                        {
                                            "origin": "autoformatted",
                                            "code": local_final_code,
                                            "result": _serialize_solver_result(
                                                local_exec_result
                                            ),
                                        }
                                    )
                                    if local_exec_result.exit_code is not None:
                                        execution_payload["exit_code"] = (
                                            local_exec_result.exit_code
                                        )
                                    if helper_library is not None:
                                        helper_info_payload.update(
                                            {
                                                "helper_error": bool(local_exec_result.helper_error),
                                                "helper_exception_type": local_exec_result.helper_exception_type,
                                                "helper_function": local_exec_result.helper_function,
                                            }
                                        )
                        elif code and local_rejection_reason is not None:
                            local_reward_reason = "rejected_unbounded_loop"
                            execution_payload.update(
                                {
                                    "error_type": "RejectedPattern",
                                    "message": (
                                        "Skipped execution because the program contains "
                                        f"{local_rejection_reason}."
                                    ),
                                    "rejection_reason": local_rejection_reason,
                                }
                            )
                        else:
                            execution_payload.update(
                                {
                                    "error_type": "CodeExtractionError",
                                    "message": "No executable code block found in response.",
                                }
                            )

                        if local_exec_result is not None:
                            payload_dict = _serialize_solver_result(local_exec_result)
                            execution_payload.update(payload_dict)

                            if local_exec_result.output is not None:
                                predicted_array = local_exec_result.output
                                local_predicted_output = (
                                    predicted_array.tolist()
                                    if isinstance(predicted_array, np.ndarray)
                                    else predicted_array
                                )
                                execution_payload["output"] = local_predicted_output

                                (
                                    local_reward,
                                    local_correct_cells,
                                    local_total_cells,
                                    local_reward_reason,
                                    normalized_array,
                                    local_reward_details,
                                ) = score_grid_prediction(
                                    predicted_array,
                                    expected_output,
                                    shape_hint=size_hint,
                                    reference_input=task.test_example.input,
                                )

                                if normalized_array is not None:
                                    if isinstance(normalized_array, np.ndarray):
                                        local_normalized_output = normalized_array.tolist()
                                    else:
                                        local_normalized_output = np.asarray(
                                            normalized_array
                                        ).tolist()

                                if expected_output is None:
                                    local_success = local_exec_result.success
                                else:
                                    local_success = bool(
                                        local_reward_details.get("exact_match", False)
                                    )
                            else:
                                if local_exec_result.success and expected_output is None:
                                    local_success = True
                                local_reward_reason = local_reward_reason or (
                                    local_exec_result.error_type or "execution_failed"
                                )
                        else:
                            local_reward = 0.0

                        if local_reward_reason is not None:
                            execution_payload["grid_evaluation_reason"] = local_reward_reason
                        if local_rejection_reason is not None:
                            execution_payload["rejection_reason"] = local_rejection_reason
                        if local_reward_details:
                            execution_payload["grid_reward_details"] = dict(
                                local_reward_details
                            )

                        local_execution_payload = execution_payload

                        step_record: Dict[str, Any] = {
                            "code": local_final_code,
                            "raw_code": local_raw_code,
                            "raw_response": local_raw_response,
                            "execution": execution_payload.copy(),
                            "executions": execution_attempts,
                        }
                        if local_formatting_result is not None:
                            step_record["formatting"] = {
                                "tool": local_formatting_result.tool,
                                "changed": local_formatting_result.changed,
                                "error": local_formatting_result.error,
                            }

                        attempt_retry_history.append(step_record)

                        if local_success or local_reward > 0.0 or retries_remaining <= 0:
                            break

                        feedback_message = build_error_feedback(
                            local_exec_result,
                            reward_reason=local_reward_reason,
                            formatting_result=local_formatting_result,
                            retries_remaining=retries_remaining - 1,
                        )
                        step_record["feedback"] = feedback_message
                        conversation_messages.append(
                            {"role": "user", "content": feedback_message}
                        )
                        retries_remaining -= 1

                    if skip_attempt:
                        return None, True, False

                    if abort_task:
                        return None, False, True

                    if not attempt_retry_history:
                        return None, False, True

                    prompt_messages_copy = [
                        deepcopy(message) for message in base_prompt_messages
                    ]
                    chat_messages = [
                        deepcopy(message) for message in conversation_messages
                    ]

                    attempt_entry: Dict[str, Any] = {
                        "prompt_messages": prompt_messages_copy,
                        "prompt_text": base_prompt_text,
                        "raw_response": local_raw_response,
                        "code": local_final_code,
                        "raw_code": local_raw_code,
                        "token_ids": local_token_ids,
                        "execution": local_execution_payload,
                        "reward": float(local_reward),
                        "success": bool(local_success),
                        "chat_messages": chat_messages,
                        "retry_history": attempt_retry_history,
                    }

                    solver_helpers_entry = dict(execution_payload.get("solver_helpers", {}))
                    if helper_library is not None:
                        solver_helpers_entry.setdefault("namespace", helper_library.namespace)
                        solver_helpers_entry.setdefault("source_sha256", helper_source_digest)
                        if helper_prompt_overview is not None:
                            solver_helpers_entry.setdefault("prompt_overview", helper_prompt_overview)
                        if helper_prompt_api is not None:
                            solver_helpers_entry.setdefault("api_reference", helper_prompt_api)
                    attempt_entry["solver_helpers"] = solver_helpers_entry

                    if local_predicted_output is not None:
                        attempt_entry["predicted_output"] = local_predicted_output
                    if local_normalized_output is not None:
                        attempt_entry["normalized_output"] = local_normalized_output
                    if local_reward_details:
                        attempt_entry["grid_reward_details"] = dict(local_reward_details)
                    if expected_output is not None:
                        attempt_entry["target_output"] = expected_output.tolist()
                    if size_hint is not None:
                        attempt_entry["output_size_hint"] = [
                            int(size_hint[0]),
                            int(size_hint[1]),
                        ]
                    if size_metadata is not None:
                        attempt_entry["output_size_inference"] = size_metadata
                    if local_correct_cells is not None:
                        attempt_entry["correct_cells"] = local_correct_cells
                    if local_total_cells is not None:
                        attempt_entry["total_cells"] = local_total_cells
                    if local_rejection_reason is not None:
                        attempt_entry["rejection_reason"] = local_rejection_reason
                    if local_reward_reason is not None:
                        attempt_entry["grid_evaluation_reason"] = local_reward_reason
                    if local_formatting_result is not None:
                        attempt_entry["formatting"] = {
                            "tool": local_formatting_result.tool,
                            "changed": local_formatting_result.changed,
                            "error": local_formatting_result.error,
                        }

                    if (
                        attempt_entry.get("reward", 0) > 0
                        and attempt_entry.get("code")
                    ):
                        chat_text = tokenizer.apply_chat_template(
                            chat_messages,
                            tokenize=False,
                            add_generation_prompt=False,
                            **dict(chat_template_kwargs or {}),
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

                    return attempt_entry, False, False
            else:
                prompt_text = get_prompt(task, system_message, self_edit_prompt)
                size_hint = None
                size_metadata = None

            # Initialize config/program tracking for this task
            if base_task_name not in explored_configs:
                explored_configs[base_task_name] = set()
                task_configs[base_task_name] = []

            expected_output = None
            if getattr(task.test_example, "output", None) is not None:
                expected_output = np.array(task.test_example.output)

            while len(task_configs[base_task_name]) < n_self_edits_per_task:
                if code_mode:
                    attempt_entry, skipped, abort_task = _run_code_mode_attempt()
                    if skipped:
                        continue
                    if attempt_entry is None:
                        if abort_task:
                            break
                        continue

                    task_configs[base_task_name].append(attempt_entry)

                    reward = attempt_entry.get("reward", 0.0)
                    reward_str = (
                        f"{reward:.3f}"
                        if isinstance(reward, (int, float))
                        else reward
                    )
                    correct_cells = attempt_entry.get("correct_cells")
                    total_cells = attempt_entry.get("total_cells")
                    reward_reason = attempt_entry.get("grid_evaluation_reason")

                    if correct_cells is not None and total_cells is not None:
                        progress_str = f" ({correct_cells}/{total_cells} cells)"
                    else:
                        progress_str = ""

                    if reward_reason:
                        detail = reward_reason
                        if (
                            reward_reason == "shape_mismatch"
                            and expected_output is not None
                            and attempt_entry.get("predicted_output") is not None
                        ):
                            try:
                                predicted_shape = tuple(
                                    np.asarray(attempt_entry.get("predicted_output")).shape
                                )
                            except Exception:
                                predicted_shape = "unknown"
                            detail = (
                                "shape_mismatch "
                                f"{predicted_shape}!={expected_output.shape}"
                            )
                        progress_str = (
                            f"{progress_str} [{detail}]"
                            if progress_str
                            else f"[{detail}]"
                        )

                    print(
                        f"New program for task {base_task_name}: success={attempt_entry.get('success', False)}, "
                        f"reward={reward_str}{progress_str}"
                    )
                    continue

                try:
                    response = self_edit_model.generate(
                        prompt_text, sampling_params=sampling_params
                    )
                except ValueError as error:
                    if "maximum model length" in str(error).lower():
                        print(
                            f"Skipping task {base_task_name} due to context "
                            f"overflow: {error}"
                        )
                        break
                    else:
                        raise
                output = response[0].outputs[0]

                try:
                    config = json.loads(output.text)
                except json.JSONDecodeError:
                    continue

                if not isinstance(config, dict):
                    print(
                        "Skipping generated config: expected a JSON object but received "
                        f"{type(config).__name__}."
                    )
                    continue

                data_generation_cfg = config.get("data_generation")
                training_cfg = config.get("training")

                if not isinstance(data_generation_cfg, dict) or not isinstance(
                    training_cfg, dict
                ):
                    print(
                        "Skipping generated config: missing or malformed "
                        "'data_generation'/'training' sections."
                    )
                    continue

                config_key = (
                    ("data_generation", tuple(sorted(data_generation_cfg.items()))),
                    ("training", tuple(sorted(training_cfg.items()))),
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

            progress_bar.update(1)
    finally:
        progress_bar.close()

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
                if attempt.get("reward", 0) > 0 and attempt.get("tokenized") is not None
            ]

            if not positive_attempts:
                print(
                    f"No rewarded programs for {base_task_name}; skipping fine-tuning."
                )
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
                    "raw_code": attempt.get("raw_code"),
                    "prompt_messages": attempt.get("prompt_messages"),
                    "prompt_text": attempt.get("prompt_text"),
                    "raw_response": attempt.get("raw_response"),
                    "execution": attempt.get("execution"),
                    "chat_messages": attempt.get("chat_messages"),
                    "retry_history": attempt.get("retry_history"),
                    "formatting": attempt.get("formatting"),
                }
                if "predicted_output" in attempt:
                    attempt_summary["predicted_output"] = attempt["predicted_output"]
                if "target_output" in attempt:
                    attempt_summary["target_output"] = attempt["target_output"]
                if "correct_cells" in attempt:
                    attempt_summary["correct_cells"] = attempt["correct_cells"]
                if "total_cells" in attempt:
                    attempt_summary["total_cells"] = attempt["total_cells"]
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
                seed=0,
                chat_template_kwargs=chat_template_kwargs,
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
    parser.add_argument('--code_mode_retry_attempts', type=int, default=2,
                      help='Maximum number of feedback-guided retries per task when code_mode is enabled.')
    parser.add_argument('--solver_timeout', type=float, default=5.0,
                      help='Wall-clock timeout (in seconds) for executing generated programs.')
    parser.add_argument('--solver_cpu_time_limit', type=float, default=5.0,
                      help='CPU time limit in seconds before the sandbox sends SIGXCPU.')
    parser.add_argument('--solver_memory_limit_mb', type=int, default=512,
                      help='Approximate memory limit in MiB for solver subprocesses.')
    parser.add_argument('--solver_disable_cpu_limit', action='store_true',
                      help='Disable the CPU time RLIMIT for sandboxed solver execution.')
    parser.add_argument('--solver_disable_memory_limit', action='store_true',
                      help='Disable the memory RLIMIT for sandboxed solver execution.')
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
    parser.add_argument('--include_solver_helpers', action='store_true',
                      help='Expose the curated ARC solver helper library to code-mode runs and describe it in prompts.')
    parser.set_defaults(vllm_enforce_eager=True)

    args = parser.parse_args()

    solver_cpu_time_limit = None if args.solver_disable_cpu_limit else args.solver_cpu_time_limit
    solver_memory_limit_mb = (
        None if args.solver_disable_memory_limit else args.solver_memory_limit_mb
    )

    main(
        experiment_name=args.experiment_name,
        skip_repeated_configs=args.skip_repeated_configs,
        challenge_file=args.challenge_file,
        solution_file=args.solution_file,
        model_name=args.model_name,
        n_tasks=args.n_tasks,
        n_self_edits_per_task=args.n_self_edits_per_task,
        code_mode=args.code_mode,
        solver_timeout=args.solver_timeout,
        solver_cpu_time_limit=solver_cpu_time_limit,
        solver_memory_limit_mb=solver_memory_limit_mb,
        code_mode_retry_attempts=args.code_mode_retry_attempts,
        vllm_dtype=args.vllm_dtype,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_max_num_batched_tokens=args.vllm_max_num_batched_tokens,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enforce_eager=args.vllm_enforce_eager,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        include_solver_helpers=args.include_solver_helpers,
    )
    
   
 
    
