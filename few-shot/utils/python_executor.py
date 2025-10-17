"""Sandboxed execution utilities for Python ARC solvers."""

from __future__ import annotations

import builtins
import importlib
import io
import multiprocessing
import queue
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - resource is not available on all platforms
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from arclib.arc import Example

GridLike = Sequence[Sequence[int]]
ExampleLike = Union["Example", Mapping[str, GridLike]]


SAFE_BUILTINS = {
    "abs",
    "all",
    "any",
    "bool",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "hash",
    "hex",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "object",
    "pow",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "type",
    "zip",
    # Exceptions and helpers commonly used in solutions.
    "Exception",
    "ValueError",
    "TypeError",
    "RuntimeError",
    "KeyError",
    "IndexError",
    "StopIteration",
    "NotImplementedError",
    "super",
}

ALLOWED_MODULES = {
    "collections",
    "functools",
    "heapq",
    "itertools",
    "math",
    "statistics",
    "typing",
    "copy",
}


@dataclass
class SolverResult:
    """Container for the outcome of executing a solver."""

    success: bool
    output: Optional[np.ndarray] = None
    error_type: Optional[str] = None
    message: Optional[str] = None
    stdout: str = ""
    stderr: str = ""


def extract_solver_code(response: str) -> str:
    """Extract the Python source code from a model response.

    The function prefers fenced code blocks and falls back to returning the
    stripped response. If multiple fenced blocks are present it chooses the one
    containing a ``def solve`` declaration when available.
    """

    if not response:
        return ""

    import re

    code_block_pattern = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    candidates = []
    for match in code_block_pattern.finditer(response):
        snippet = match.group(1).strip()
        if snippet.lower().startswith("python"):
            snippet = snippet[len("python") :].strip()
        candidates.append(snippet)

    if candidates:
        for candidate in candidates:
            if "def solve" in candidate:
                return candidate.strip()
        return candidates[0].strip()

    stripped = response.strip()
    if stripped.startswith("```"):
        import re

        stripped = re.sub(r"^```(?:python)?", "", stripped, flags=re.IGNORECASE).lstrip()
        stripped = re.sub(r"```$", "", stripped).rstrip()
    return stripped


def _serialize_examples(examples: Iterable[ExampleLike]) -> List[Mapping[str, List[List[int]]]]:
    serialized: List[Mapping[str, List[List[int]]]] = []
    for example in examples:
        if hasattr(example, "input") and hasattr(example, "output"):
            input_grid = np.array(getattr(example, "input")).tolist()
            output_grid = np.array(getattr(example, "output")).tolist()
        elif isinstance(example, Mapping):
            input_grid = np.array(example["input"]).tolist()
            output_grid = np.array(example.get("output")).tolist()
        else:
            raise TypeError(f"Unsupported example type: {type(example)!r}")
        serialized.append({"input": input_grid, "output": output_grid})
    return serialized


def _apply_limits(memory_limit_bytes: Optional[int], cpu_time_limit_s: Optional[int]) -> None:
    if resource is None:
        return
    if memory_limit_bytes is not None:
        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
        except Exception:
            pass
    if cpu_time_limit_s is not None:
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_time_limit_s, cpu_time_limit_s))
        except Exception:
            pass


def _worker(
    code: str,
    train_examples: List[Mapping[str, List[List[int]]]],
    test_input: List[List[int]],
    result_queue: multiprocessing.Queue,
    memory_limit_bytes: Optional[int],
    cpu_time_limit_s: Optional[int],
) -> None:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    try:
        _apply_limits(memory_limit_bytes, cpu_time_limit_s)

        safe_builtins = {name: getattr(builtins, name) for name in SAFE_BUILTINS if hasattr(builtins, name)}

        def safe_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
            if level != 0:
                raise ImportError("Relative imports are not allowed")
            root = name.split(".")[0]
            if root not in ALLOWED_MODULES:
                raise ImportError(f"Import of '{name}' is not permitted")
            return importlib.import_module(name)

        safe_builtins["__build_class__"] = builtins.__build_class__
        safe_builtins["__import__"] = safe_import

        globals_dict = {
            "__builtins__": safe_builtins,
            "TRAIN_EXAMPLES": train_examples,
            "TEST_INPUT": test_input,
        }

        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exec(code, globals_dict)
            solver = globals_dict.get("solve")
            if not callable(solver):
                raise ValueError("No callable solve() function found in generated code.")
            result = solver(test_input)

        try:
            output_array = np.array(result, dtype=int)
        except Exception as exc:  # pragma: no cover - dtype failure path
            raise ValueError(f"solve() must return a 2D array-like of ints: {exc}") from exc

        if output_array.ndim != 2:
            raise ValueError("solve() must return a 2D array-like of ints")

        result_queue.put(
            {
                "status": "ok",
                "output": output_array.tolist(),
                "stdout": stdout_buffer.getvalue(),
                "stderr": stderr_buffer.getvalue(),
            }
        )
    except Exception as exc:  # pragma: no cover - error paths covered via unit tests
        result_queue.put(
            {
                "status": "error",
                "error_type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "stdout": stdout_buffer.getvalue(),
                "stderr": stderr_buffer.getvalue(),
            }
        )
    finally:
        stdout_buffer.close()
        stderr_buffer.close()


def run_solver(
    code_str: str,
    train_examples: Iterable[ExampleLike],
    test_input: Union[GridLike, np.ndarray],
    *,
    timeout: float = 5.0,
    memory_limit_mb: Optional[int] = 512,
    cpu_time_limit_s: Optional[int] = 5,
) -> SolverResult:
    """Execute ``code_str`` safely and return the resulting grid or error."""

    if not code_str.strip():
        return SolverResult(
            success=False,
            error_type="EmptySource",
            message="No source code provided for execution.",
        )

    if "def solve" not in code_str:
        return SolverResult(
            success=False,
            error_type="MissingEntryPoint",
            message="Generated code does not define a solve() function.",
        )

    serialized_examples = _serialize_examples(train_examples)
    test_grid = np.array(test_input).tolist()

    ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue = ctx.Queue()
    memory_limit_bytes = None if memory_limit_mb is None else memory_limit_mb * 1024 * 1024

    process = ctx.Process(
        target=_worker,
        args=(code_str, serialized_examples, test_grid, result_queue, memory_limit_bytes, cpu_time_limit_s),
    )
    process.start()

    exit_code: Optional[int] = None
    try:
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join()
            return SolverResult(
                success=False,
                error_type="Timeout",
                message=f"Execution timed out after {timeout} seconds.",
            )
        exit_code = process.exitcode
    finally:
        # Drain any lingering data if the process exited early to avoid deadlocks.
        try:
            process.close()
        except AttributeError:
            pass

    payload: Optional[dict] = None
    try:
        payload = result_queue.get(timeout=0.5)
    except queue.Empty:
        payload = None

    result_queue.close()
    try:
        result_queue.join_thread()
    except (AttributeError, ValueError):  # pragma: no cover - platform specific
        pass

    if payload is None:
        if exit_code is None:
            exit_code = process.exitcode
        if exit_code not in (0, None):
            return SolverResult(
                success=False,
                error_type="ProcessExit",
                message=f"Solver exited with status {exit_code} without reporting a result.",
            )

        return SolverResult(
            success=False,
            error_type="NoResult",
            message="Solver process exited without returning a result.",
        )

    if payload.get("status") == "ok":
        output_array = np.array(payload["output"], dtype=int)
        return SolverResult(
            success=True,
            output=output_array,
            stdout=payload.get("stdout", ""),
            stderr=payload.get("stderr", ""),
        )

    return SolverResult(
        success=False,
        error_type=payload.get("error_type", "ExecutionError"),
        message=payload.get("message"),
        stdout=payload.get("stdout", ""),
        stderr=payload.get("stderr", ""),
    )


__all__ = ["SolverResult", "extract_solver_code", "run_solver"]
