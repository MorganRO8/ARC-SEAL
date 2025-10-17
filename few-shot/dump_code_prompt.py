"""Utility script to inspect Python solver prompts for ARC tasks.

This script selects a task from a challenge file and renders the exact
code-generation prompt that ``self-edit.py`` would feed to the model when
``--code_mode`` is enabled.  The output is written to a user-provided text
file so prompts can be reviewed and iterated on offline.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from transformers import AutoTokenizer

from arclib.arc import Task, read_tasks_from_single_file
from arclib.messagers import PythonSolverMessageRepresenter
from utils.chat_template import detect_thinking_support
from utils.output_size_inference import infer_output_shape
from utils.solver_helpers import HelperLibrary, get_helper_library


@dataclass
class PromptRenderResult:
    """Container describing a rendered code-mode prompt."""

    task: Task
    prompt_text: str
    messages: List[Dict[str, Any]]
    size_hint: Optional[Tuple[int, int]]
    size_metadata: Optional[Dict[str, Any]]
    helper_library: Optional[HelperLibrary]


def _select_task(tasks: Sequence[Task], *, rng: random.Random, name: Optional[str]) -> Task:
    """Return either the named task or a random choice from ``tasks``."""

    if not tasks:
        raise ValueError("Challenge file did not yield any tasks.")

    if name:
        for task in tasks:
            if task.name == name:
                return task
        raise ValueError(f"Task named '{name}' was not found in the provided file.")

    return rng.choice(list(tasks))


def build_code_mode_prompt_for_task(
    task: Task,
    tokenizer,
    *,
    include_helpers: bool = False,
    execution_limits: Optional[Dict[str, Any]] = None,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> PromptRenderResult:
    """Construct the self-edit code-mode prompt for ``task``."""

    if execution_limits is None:
        execution_limits = {}
    if chat_template_kwargs is None:
        chat_template_kwargs = {}

    representer = PythonSolverMessageRepresenter()

    helper_library: Optional[HelperLibrary] = None
    helper_kwargs: Dict[str, Any] = {}
    if include_helpers:
        helper_library = get_helper_library()
        helper_kwargs = {
            "helper_overview": helper_library.prompt_overview,
            "helper_api_reference": helper_library.api_reference,
            "helper_source": helper_library.source,
            "helper_namespace": helper_library.namespace,
        }

    size_hint, size_metadata = infer_output_shape(task)

    messages, _ = representer.encode(
        task,
        size_inference=(size_hint, size_metadata),
        execution_limits=execution_limits,
        **helper_kwargs,
    )

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )

    return PromptRenderResult(
        task=task,
        prompt_text=prompt_text,
        messages=messages,
        size_hint=size_hint,
        size_metadata=size_metadata,
        helper_library=helper_library,
    )


def _format_metadata_lines(result: PromptRenderResult, *, model_name: str) -> Iterable[str]:
    """Yield human-readable metadata strings for the rendered prompt."""

    size_hint = result.size_hint
    size_line = "unknown"
    if size_hint is not None:
        size_line = f"{int(size_hint[0])}x{int(size_hint[1])}"
    helper_status = "disabled"
    helper_namespace = ""
    if result.helper_library is not None:
        helper_status = "enabled"
        helper_namespace = f" ({result.helper_library.namespace})"

    yield f"Model: {model_name}"
    yield f"Task: {result.task.name or '(unnamed task)'}"
    yield f"Output size hint: {size_line}"
    yield f"Helpers: {helper_status}{helper_namespace}"

    if result.size_metadata:
        method = result.size_metadata.get("method")
        explanation = result.size_metadata.get("explanation")
        if method:
            yield f"Size inference method: {method}"
        if explanation:
            yield f"Size inference notes: {explanation}"


def render_prompt_to_file(
    result: PromptRenderResult,
    *,
    output_path: Path,
    model_name: str,
) -> None:
    """Write the rendered prompt and metadata to ``output_path``."""

    metadata_lines = list(_format_metadata_lines(result, model_name=model_name))

    sections = ["# " + line for line in metadata_lines]
    sections.append("#")
    sections.append("# Prompt follows below. Copy everything after this comment for generation.")
    sections.append("")
    sections.append(result.prompt_text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Python solver prompt for a random ARC task.",
    )
    parser.add_argument(
        "--challenge_file",
        required=True,
        help="Path to the ARC challenges JSON file.",
    )
    parser.add_argument(
        "--solution_file",
        help="Optional ARC solutions JSON file for tasks that require it.",
    )
    parser.add_argument(
        "--model_name",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model/tokenizer name used to format the chat prompt.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination text file for the rendered prompt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for selecting a task.",
    )
    parser.add_argument(
        "--task_name",
        help="Optional specific task name to render instead of choosing randomly.",
    )
    parser.add_argument(
        "--include_solver_helpers",
        action="store_true",
        help="Include the optional solver helper library in the prompt and sandbox scope.",
    )
    parser.add_argument(
        "--solver_timeout",
        type=float,
        default=5.0,
        help="Wall-clock timeout (seconds) conveyed in the prompt metadata.",
    )
    parser.add_argument(
        "--solver_cpu_time_limit",
        type=float,
        default=5.0,
        help="CPU time limit (seconds) described to the model; set <=0 to disable.",
    )
    parser.add_argument(
        "--solver_memory_limit_mb",
        type=int,
        default=512,
        help="Approximate memory limit in MiB communicated to the model; set <=0 to disable.",
    )
    parser.add_argument(
        "--solver_disable_cpu_limit",
        action="store_true",
        help="Indicate that the sandbox CPU limit is disabled when rendering the prompt.",
    )
    parser.add_argument(
        "--solver_disable_memory_limit",
        action="store_true",
        help="Indicate that the sandbox memory limit is disabled when rendering the prompt.",
    )
    return parser.parse_args()


def _build_execution_limits(args: argparse.Namespace) -> Dict[str, Any]:
    cpu_limit: Optional[int]
    if args.solver_disable_cpu_limit or args.solver_cpu_time_limit is None:
        cpu_limit = None
    elif args.solver_cpu_time_limit <= 0:
        cpu_limit = None
    else:
        cpu_limit = int(round(args.solver_cpu_time_limit))

    memory_limit: Optional[int]
    if args.solver_disable_memory_limit or args.solver_memory_limit_mb is None:
        memory_limit = None
    elif args.solver_memory_limit_mb <= 0:
        memory_limit = None
    else:
        memory_limit = int(args.solver_memory_limit_mb)

    timeout = max(float(args.solver_timeout), 0.0)

    return {
        "timeout": timeout,
        "cpu_time_limit": cpu_limit,
        "memory_limit_mb": memory_limit,
    }


def main() -> None:
    args = parse_args()

    rng = random.Random(args.seed)

    tasks = read_tasks_from_single_file(
        args.challenge_file,
        solution_file=args.solution_file,
    )

    # Exclude duplicate "-1" variants by default to align with self-edit.py.
    filtered_tasks: List[Task] = []
    for task in tasks:
        if task.name.endswith("-1"):
            continue
        filtered_tasks.append(task)

    selected_task = _select_task(filtered_tasks or tasks, rng=rng, name=args.task_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    thinking_config = detect_thinking_support(tokenizer)
    chat_template_kwargs = dict(thinking_config.apply_chat_kwargs)

    if thinking_config.enabled:
        reason = thinking_config.reason or "detected thinking support"
        print(f"Enabling thinking mode for chat template ({reason}).")

    execution_limits = _build_execution_limits(args)

    result = build_code_mode_prompt_for_task(
        selected_task,
        tokenizer,
        include_helpers=args.include_solver_helpers,
        execution_limits=execution_limits,
        chat_template_kwargs=chat_template_kwargs,
    )

    output_path = Path(args.output)
    render_prompt_to_file(result, output_path=output_path, model_name=args.model_name)

    print(f"Wrote prompt for task '{selected_task.name}' to {output_path.resolve()}")


if __name__ == "__main__":
    main()

