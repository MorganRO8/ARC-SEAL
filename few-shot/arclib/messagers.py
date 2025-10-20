"""
This module contains classes for representing tasks and examples as messages for chat-based interfaces.
"""
from abc import ABC, abstractmethod
from html import escape
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from arclib.representers import GridRepresenter

from .arc import Example, Task
from .representers import (
    CompositeRepresenter,
    ConnectedComponentRepresenter,
    DelimitedGridRepresenter,
    DiffExampleRepresenter,
    ImageTaskRepresenter,
    PythonListGridRepresenter,
    TaskRepresenter,
    TextTaskRepresenter,
    TextExampleRepresenter,
    WordGridRepresenter,
)

from utils.prompts import (
    format_runtime_constraints,
    python_solver_function_docstring,
    python_solver_function_signature,
    python_solver_response_stub,
    python_solver_system_prompt,
    python_solver_user_template,
)

from utils.output_size_inference import infer_output_shape


MESSAGE = Dict[str, Union[str, Dict]]
MESSAGES = List[MESSAGE]


def _grid_to_string(grid: Sequence[Sequence[int]]) -> str:
    """Convert a 2D grid into a space-separated string representation."""

    if not grid:
        return "(empty grid)"

    rows = []
    for row in grid:
        if not row:
            rows.append("(empty row)")
            continue
        rows.append(" ".join(str(int(cell)) for cell in row))
    return "\n".join(rows)


def _encode_grid_for_prompt(
    grid: Sequence[Sequence[int]],
    grid_representer: Optional[GridRepresenter],
) -> str:
    """Return a prompt-friendly string for ``grid``."""

    array = np.asarray(grid)

    if grid_representer is None:
        return _grid_to_string(array.tolist())

    if isinstance(grid_representer, PythonListGridRepresenter):
        return _grid_to_string(array.tolist())

    encoded = grid_representer.encode(array)

    if isinstance(grid_representer, CompositeRepresenter):
        # ``CompositeRepresenter`` already stitches multiple views together with
        # explicit headers. Avoid stripping internal whitespace that might
        # collapse those sections, but do remove a trailing newline if one was
        # added while composing the views.
        return encoded.rstrip("\n")

    return encoded


def _format_train_examples_for_prompt(
    examples: Sequence[Example],
    grid_representer: Optional[GridRepresenter] = None,
) -> str:
    """Render training examples for inclusion in the Python solver prompt."""

    if not examples:
        return "No training examples were provided for this task."

    blocks: List[str] = []
    for idx, example in enumerate(examples, start=1):
        input_grid = _encode_grid_for_prompt(example.input, grid_representer)
        output_grid = _encode_grid_for_prompt(example.output, grid_representer)
        blocks.append(
            f"Input {idx}:\n{input_grid}\n\nOutput:\n{output_grid}"
        )
    return "\n\n".join(blocks)


def _format_test_example_for_prompt(
    example: Example,
    grid_representer: Optional[GridRepresenter] = None,
) -> str:
    """Render the test example grid(s) for the Python solver prompt."""

    parts = [
        f"Test input:\n{_encode_grid_for_prompt(example.input, grid_representer)}"
    ]

    if getattr(example, "output", None) is not None:
        parts.append(
            "Expected output:\n"
            + _encode_grid_for_prompt(example.output, grid_representer)
        )

    return "\n\n".join(parts)


def _format_grid_dimensions(grid) -> Optional[str]:
    """Return a human-readable rows × cols string for a grid-like object."""

    if grid is None:
        return None

    try:
        array = np.asarray(grid)
    except Exception:
        return None

    if array.ndim < 2:
        if array.ndim == 1:
            return f"{int(array.shape[0])}×1"
        return None

    rows, cols = array.shape[:2]
    try:
        rows_int = int(rows)
        cols_int = int(cols)
    except (TypeError, ValueError):
        return None

    return f"{rows_int}×{cols_int}"


def _summarize_task_grid_dimensions(
    task: Task, inferred_shape: Optional[Tuple[int, int]] = None
) -> str:
    """Build a bullet-list summary of train/test grid dimensions for a task."""

    lines: List[str] = []
    for idx, example in enumerate(task.train_examples, start=1):
        input_dims = _format_grid_dimensions(example.input)
        output_dims = _format_grid_dimensions(example.output)
        segments: List[str] = []
        if input_dims:
            segments.append(f"input {input_dims}")
        if output_dims:
            segments.append(f"output {output_dims}")
        if segments:
            lines.append(f"Train pair {idx}: " + ", ".join(segments))

    test_input_dims = _format_grid_dimensions(task.test_example.input)
    if test_input_dims:
        lines.append(f"Test input: {test_input_dims}")

    test_output = getattr(task.test_example, "output", None)
    if test_output is not None:
        test_output_dims = _format_grid_dimensions(test_output)
        if test_output_dims:
            lines.append(f"Test expected output: {test_output_dims}")

    if inferred_shape is not None:
        lines.append(
            "Inferred output size guidance: "
            f"{int(inferred_shape[0])}×{int(inferred_shape[1])}"
        )

    if not lines:
        return "(no grid dimension metadata available)"

    return "\n".join(f"- {line}" for line in lines)


def _format_size_guidance(
    task: Task,
    inferred_shape: Optional[Tuple[int, int]],
    inference_details: Optional[Dict[str, object]],
) -> str:
    if inference_details is None:
        inference_details = {}

    explanation = inference_details.get("explanation")
    method = inference_details.get("method")
    evidence = inference_details.get("evidence") or {}

    def _dims(shape: Optional[Tuple[int, int]]) -> Optional[str]:
        if shape is None:
            return None
        return f"{int(shape[0])}×{int(shape[1])}"

    def _shape_of(array) -> Optional[Tuple[int, int]]:
        try:
            values = np.asarray(array)
        except Exception:
            return None
        if values.ndim < 2:
            return None
        rows, cols = values.shape[:2]
        try:
            return int(rows), int(cols)
        except (TypeError, ValueError):
            return None

    test_input_shape = _shape_of(task.test_example.input)

    def _format_param(value: Optional[float]) -> str:
        if value is None:
            return "?"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, float) and abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return str(value)

    if inferred_shape is None:
        if explanation:
            return explanation
        return (
            "No automatic output-size heuristic matched this task. Use the training examples"
            " to deduce the correct output dimensions and implement them explicitly."
        )

    size_text = _dims(inferred_shape)
    method_display = method.replace("_", " ") if isinstance(method, str) else "heuristic"

    summary_parts: List[str] = [
        f"The expected test output grid size is {size_text} based on the {method_display} rule."
    ]

    if method == "constant_match":
        test_dims = _dims(test_input_shape)
        if test_dims:
            summary_parts.append(
                "All training inputs and outputs share these dimensions, and the test input"
                f" already has size {test_dims}. Treat this as an identity size rule."
            )
        else:
            summary_parts.append(
                "All training inputs and outputs share these dimensions. Treat the test case as"
                " using the same size."
            )
    elif method == "arithmetic_addition":
        row_param = evidence.get("row_param")
        col_param = evidence.get("col_param")
        if row_param == 0 and col_param == 0:
            summary_parts.append(
                "Training pairs show that outputs always match their corresponding inputs, so"
                " keep the test output the same size as its input."
            )
        else:
            summary_parts.append(
                "Training pairs change the grid size by adding"
                f" {_format_param(row_param)} rows and {_format_param(col_param)} columns. Apply the same adjustment to the"
                " test input before producing the result."
            )
    elif method and method.startswith("arithmetic_"):
        op = method.split("_", 1)[1]
        row_param = evidence.get("row_param")
        col_param = evidence.get("col_param")
        summary_parts.append(
            "Training grids map inputs to outputs via {op} by {row_param} on rows and"
            " {col_param} on columns; use that rule for the test grid.".format(
                op=op,
                row_param=_format_param(row_param),
                col_param=_format_param(col_param),
            )
        )

    if explanation:
        summary_parts.append(explanation)

    if test_input_shape is not None:
        if inferred_shape == test_input_shape:
            summary_parts.append(
                "Ensure your solver always returns grids with the same dimensions as the test input."
            )
        else:
            summary_parts.append(
                "The test input is {_dims(test_input_shape)}; make sure your code constructs a"
                f" {size_text} grid even when intermediate computations use other shapes."
            )
    else:
        summary_parts.append(
            f"Ensure your solver always returns a {size_text} grid."
        )

    return " ".join(part.strip() for part in summary_parts if part)


def display_messages(messages: MESSAGES):
    html_output = """<!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">
    <title>Chat View</title>
    <style>
    /* CSS styling for chat interface */
    body {
    font-family: Arial, sans-serif;
    background-color: #f5f5f5;
    }
    .chat-container {
    width: 80%;
    max-width: 800px;
    margin: 0 auto;
    margin-top: 50px;
    }
    .message {
    display: block;
    clear: both;
    margin-bottom: 15px;
    }
    .message.user {
    text-align: right;
    }
    .message.assistant {
    text-align: left;
    }
    .message.system {
    text-align: left;
    }
    .message .bubble {
    display: inline-block;
    padding: 10px 15px;
    border-radius: 15px;
    max-width: 70%;
    position: relative;
    }
    .message.user .bubble {
    background-color: #0084ff;
    color: white;
    }
    .message.assistant .bubble {
    background-color: #e5e5ea;
    color: black;
    }
    .message.system .bubble {
    background-color: #e5e5ea;
    color: black;
    }
    .message .bubble img {
    max-width: 100%;
    border-radius: 10px;
    }
    .message .role {
    font-size: 0.8em;
    color: black;
    margin-bottom: 5px;
    }
    </style>
    </head>
    <body>
    <div class="chat-container">
    """

    # Loop through messages
    for message in messages:
        role = message.get("role", "user")
        content_list = message.get("content", [])
        if not content_list:
            continue  # Skip if no content
        if isinstance(content_list, str):
            content_list = [{"type": "text", "text": content_list}]

        # Start message div
        html_output += f'<div class="message {role}">\n'
        # Start bubble div
        html_output += '<div class="bubble">\n'
        # Add role label inside the bubble
        html_output += f'<div class="role">{role.capitalize()}</div>\n'

        # Process content items
        for content in content_list:
            content_type = content.get("type")
            if content_type == "text":
                text = content.get("text", "")
                # Escape HTML entities in text
                safe_text = escape(text)
                # Replace newlines with <br>
                safe_text = safe_text.replace("\n", "<br>")
                html_output += f"<p>{safe_text}</p>\n"
            elif content_type == "image_url":
                image_url = content["image_url"].get("url", {})
                if image_url:
                    html_output += f'<img src="{image_url}" alt="Image">\n'
            else:
                # Handle other content types if necessary
                pass

        # Close bubble and message divs
        html_output += "</div>\n</div>\n"

    # Close chat-container and body tags
    html_output += """
</div>
</body>
</html>"""

    return html_output


class MessageRepresenter(ABC):
    task_representer: TaskRepresenter

    @abstractmethod
    def encode(self, task: Task, **kwargs) -> Tuple[MESSAGES, MESSAGE]:
        pass

    def display(self, messages: MESSAGES):
        return display_messages(messages)


# =============== MESSAGE REPRESENTATION ===============


class GPTTextMessagerepresenter(MessageRepresenter):
    def __init__(
        self,
        prompt: Optional[
            str
        ] = "Figure out the pattern in the following examples and apply it to the test case. {description}Your answer must follow the format of the examples. \n",
        task_representer: TaskRepresenter = TextTaskRepresenter(),
    ):
        self.prompt = prompt
        self.task_representer = task_representer

    def encode(self, task: Task, **kwargs) -> Tuple[MESSAGES, MESSAGE]:
        input_data = []

        if hasattr(task, "description"):
            desciption = "Here is a description of the task: \n\n{description}\n"
            description = desciption.format(description=task.description)
            prompt = self.prompt.format(description=description)
        else:
            prompt = self.prompt.format(description="")

        input_data.append({"role": "system", "content": prompt})

        for example in task.train_examples:
            query, output = self.task_representer.example_representer.encode(example, **kwargs)
            input_data.append({"role": "system", "content": query + output})

        query, output = self.task_representer.example_representer.encode(
            task.test_example, **kwargs
        )

        input_data.append({"role": "user", "content": query})

        output_data = {"role": "assistant", "content": output}

        return input_data, output_data

    def decode(self, input_data: MESSAGES, output_data: MESSAGE, **kwargs) -> Task:
        raise NotImplementedError("Decoding for GPTTextMessagerepresenter is not implemented.")


class GPTTextMessageRepresenterV2(MessageRepresenter):
    def __init__(
        self,
        prompt: Optional[
            str
        ] = "Figure out the underlying transformation in the following examples and apply it to the test case. {description}Here are some examples from this transformation, your answer must follow the format.\n",
        task_representer: TaskRepresenter = TextTaskRepresenter(),
    ):
        self.prompt = prompt
        self.task_representer = task_representer
        # if example_representer is not None:
        #     self.task_representer.example_representer = example_representer(
        #                 io_sep=" -> ",
        #                 input_header="",
        #                 output_header="",
        #                 grid_representer=PythonListGridRepresenter
        #             )

    def encode(self, task: Task, **kwargs) -> Tuple[MESSAGES, MESSAGE]:
        input_data = []

        if hasattr(task, "description"):
            description = task.description
            description = f"\n\n A possible description of the transformation: \n\n{description}\n"
            prompt = self.prompt.format(description=description)
        else:
            prompt = self.prompt.format(description="")

        if isinstance(self.task_representer.example_representer, DiffExampleRepresenter):
            if self.task_representer.example_representer.use_output:
                prompt += "The input-diff-output grids are provided as python arrays where the diff is simply the output minus input:\n"
            else:
                prompt += "The input-diff grids are provided as python arrays:\n"
        elif isinstance(
            self.task_representer.example_representer.grid_representer,
            ConnectedComponentRepresenter,
        ):
            connected_component = kwargs.get(
                "connected_component",
                self.task_representer.example_representer.grid_representer.connected_component,
            )
            connected_component = (
                "including diagonals" if connected_component == 8 else "excluding diagonals"
            )
            prompt += f"The input-output grids are provided with indices of connected shapes ({connected_component}) of the same color:\n"
        elif isinstance(
            self.task_representer.example_representer.grid_representer, PythonListGridRepresenter
        ):
            prompt += "The input-output grids are provided as python arrays:\n"
        elif isinstance(
            self.task_representer.example_representer.grid_representer, CompositeRepresenter
        ):
            composite = self.task_representer.example_representer.grid_representer
            if any(
                isinstance(representer, ConnectedComponentRepresenter)
                for representer in getattr(composite, "representers", [])
            ):
                connected_component = kwargs.get(
                    "connected_component", composite.connected_component
                )
                connected_component = (
                    "including diagonals"
                    if connected_component == 8
                    else "excluding diagonals"
                )
                prompt += (
                    "The input-output grids are provided as both python arrays and "
                    f"indices of connected shapes ({connected_component}) of the same color:\n"
                )
            else:
                prompt += (
                    "The input-output grids are provided with multiple textual views, "
                    "including rotations and diagonals in both directions, alongside the base python array representation:\n"
                )

        for example in task.train_examples:
            query, output = self.task_representer.example_representer.encode(example, **kwargs)
            if query is None or output is None:
                return None, None
            prompt += query + output + "\n"

        input_data.append({"role": "system", "content": prompt})

        query, output = self.task_representer.example_representer.encode(
            task.test_example, **kwargs
        )
        if query is None or output is None:
            return None, None

        input_data.append({"role": "user", "content": query})

        output_data = {"role": "assistant", "content": output}

        return input_data, output_data

    def decode(self, input_data: MESSAGES, output_data: MESSAGE, **kwargs) -> Task:
        raise NotImplementedError("Decoding for GPTTextMessageRepresenterV2 is not implemented.")

    def __repr__(self) -> str:
        return f"GPTTextMessageRepresenterV2(prompt={self.prompt!r}, task_representer={repr(self.task_representer)})"


class PythonSolverMessageRepresenter(MessageRepresenter):
    def __init__(
        self,
        system_prompt: str = python_solver_system_prompt,
        user_prompt_template: str = python_solver_user_template,
        function_signature: str = python_solver_function_signature,
        function_docstring: str = python_solver_function_docstring,
        task_representer: TaskRepresenter = TextTaskRepresenter(
            example_representer=TextExampleRepresenter(
                io_sep="",
                input_header="",
                output_header="",
                grid_representer=PythonListGridRepresenter(),
            )
        ),
        response_stub: str = python_solver_response_stub,
    ):
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.function_signature = function_signature
        self.function_docstring = function_docstring
        self.task_representer = task_representer
        self.response_stub = response_stub

    def encode(self, task: Task, **kwargs) -> Tuple[MESSAGES, MESSAGE]:
        size_inference = kwargs.get("size_inference")
        if size_inference is None:
            inferred_shape, inference_details = infer_output_shape(task)
        else:
            inferred_shape, inference_details = size_inference

        execution_limits = kwargs.get("execution_limits") or {}
        runtime_guidance = format_runtime_constraints(
            execution_limits.get("timeout"),
            execution_limits.get("cpu_time_limit"),
            execution_limits.get("memory_limit_mb"),
        )

        description = ""
        if getattr(task, "description", None):
            description = f"Task description:\n{task.description.strip()}\n\n"

        grid_representer = getattr(
            getattr(self.task_representer, "example_representer", None),
            "grid_representer",
            None,
        )

        train_examples_text = _format_train_examples_for_prompt(
            task.train_examples, grid_representer
        )
        test_example_text = _format_test_example_for_prompt(
            task.test_example, grid_representer
        )

        grid_stats = _summarize_task_grid_dimensions(task, inferred_shape)
        size_guidance = _format_size_guidance(task, inferred_shape, inference_details)

        helper_summary = kwargs.get("helper_summary")
        helper_groups = kwargs.get("helper_groups") or []
        helper_namespace = kwargs.get("helper_namespace", "ARC_HELPERS")

        helper_sections: List[str] = []
        if helper_summary or helper_groups:
            helper_sections.append(
                "The runtime preloads ARC utility functions under the name ``%s``."
                " Call them directly in your solver." % helper_namespace
            )
            if helper_summary:
                helper_sections.append(helper_summary)
            if helper_groups:
                helper_sections.extend(helper_groups)
        else:
            helper_sections.append("No ARC utility functions are preloaded for this run.")

        helper_section = "\n\n".join(helper_sections)

        user_content = self.user_prompt_template.format(
            description=description,
            train_examples=train_examples_text,
            test_example=test_example_text,
            function_signature=self.function_signature,
            function_docstring=self.function_docstring,
            grid_stats=grid_stats,
            size_guidance=size_guidance,
            runtime_guidance=runtime_guidance,
            helper_section=helper_section,
        )

        input_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        assistant_message = {"role": "assistant", "content": self.response_stub}

        return input_messages, assistant_message

    def decode(self, input_data: MESSAGES, output_data: MESSAGE, **kwargs) -> Task:
        raise NotImplementedError("Decoding for PythonSolverMessageRepresenter is not implemented.")

    def __repr__(self) -> str:
        return (
            "PythonSolverMessageRepresenter("
            f"system_prompt={self.system_prompt!r}, "
            f"user_prompt_template={self.user_prompt_template!r}, "
            f"function_signature={self.function_signature!r}, "
            f"function_docstring={self.function_docstring!r}, "
            f"task_representer={repr(self.task_representer)})"
        )


class GPTTextMessageRepresenterV2CoT(MessageRepresenter):
    def __init__(
        self,
        prompt: Optional[str] = None,
        task_representer: TaskRepresenter = TextTaskRepresenter(),
    ):
        if prompt:
            self.prompt = prompt
        else:
            self.prompt = "Figure out the underlying transformation in the following examples and apply it to the test case. {description}Here are some examples from this transformation, your answer must follow the format.\n"

        self.task_representer = task_representer

    def encode(self, task: Task) -> Tuple[MESSAGES, MESSAGE]:
        input_data = []

        if hasattr(task, "description"):
            description = task.description
            description = f"\n\n A possible description of the transformation: \n\n{description}\n"
            prompt = self.prompt.format(description=description)
        else:
            prompt = self.prompt.format(description="")

        if isinstance(
            self.task_representer.example_representer.grid_representer,
            ConnectedComponentRepresenter,
        ):
            prompt += "The input-output grids are provided with indices of connected shapes of the same color:\n"
        elif isinstance(
            self.task_representer.example_representer.grid_representer, PythonListGridRepresenter
        ):
            prompt += "The input-output grids are provided as python arrays:\n"
        elif isinstance(
            self.task_representer.example_representer.grid_representer, CompositeRepresenter
        ):
            prompt += "The input-output grids are provided as both python arrays and indices of connected shapes of the same color:\n"

        for example in task.train_examples:
            query, output = self.task_representer.example_representer.encode(example)
            prompt += query + output + "\n"

        input_data.append({"role": "system", "content": prompt})

        query, output = self.task_representer.example_representer.encode(task.test_example)

        input_data.append({"role": "user", "content": query + ". Let's think step by step:"})

        cot_strs = ""
        for i, cot in enumerate(task.test_example.cot[:-1]):
            if -1 in cot:
                cot = np.where(cot == -1, 0, cot)
            cot_str = self.task_representer.example_representer.grid_representer.encode(cot)
            cot_str = "Step-" + str(i + 1) + ":\n" + cot_str
            cot_strs += cot_str + "\n"

        cot_strs += (
            "Final Step:\n"
            + self.task_representer.example_representer.grid_representer.encode(
                task.test_example.cot[-1]
            )
        )

        output_data = {"role": "assistant", "content": cot_strs}

        return input_data, output_data

    def decode(self, input_data: MESSAGES, output_data: MESSAGE, **kwargs) -> Task:
        raise NotImplementedError("Decoding for GPTTextMessageRepresenterV2CoT is not implemented.")


class DataToCodeTextrepresenter(MessageRepresenter):
    def __init__(
        self,
        task_representer: TaskRepresenter = TextTaskRepresenter(),
        prompt: Optional[
            str
        ] = "Figure out the underlying code that produces the following input-output grids:\n",
    ):
        self.prompt = prompt
        self.task_representer = task_representer

    def encode(self, task: Task, code: str) -> Tuple[MESSAGES, MESSAGE]:
        input_data = []

        prompt = self.prompt

        input_data.append({"role": "system", "content": prompt})

        data_points = ""

        for example in task.train_examples:
            query, output = self.task_representer.example_representer.encode(example)
            data_points += query + output + "\n"

        input_data.append({"role": "user", "content": data_points})

        output_data = {"role": "assistant", "content": code}

        return input_data, output_data

    def decode(self, input_data: MESSAGES, output_data: MESSAGE, **kwargs) -> Task:
        # Decoding logic for DataToCodeTextrepresenter is complex and depends on the specific encoding format.
        # This is a placeholder for the actual implementation.
        raise NotImplementedError("Decoding for DataToCodeTextrepresenter is not implemented.")


class GPTTextMessageRepresenterFewShot(MessageRepresenter):
    def __init__(
        self,
        task_representer: TaskRepresenter = TextTaskRepresenter(),
        prompt: Optional[
            str
        ] = "Figure out the underlying transformations in each task and complete the examples. You must follow the format.\n\n",
    ):
        self.prompt = prompt
        self.task_representer = task_representer

    def encode(
        self, task: Task, examples: List[Task], num_demonstrations: List[int]
    ) -> Tuple[MESSAGES, MESSAGE]:
        input_data = []

        prompts = []
        for i, demo_task in enumerate(examples):
            k = num_demonstrations[i]
            if k >= 3:
                prompt = "== START OF TASK ==\n\n"
                demonstrations = demo_task.train_examples + [demo_task.test_example]
                for j in range(k):
                    example = demonstrations[j]
                    query, output = self.task_representer.example_representer.encode(example)
                    prompt += query + output + "\n\n"
                prompt += "== END OF TASK ==\n\n"
                prompts.append(prompt)

        prompts = "".join(prompts)

        input_data.append({"role": "system", "content": self.prompt + prompts})

        prompt = ""
        for example in task.train_examples:
            query, output = self.task_representer.example_representer.encode(example)
            prompt += query + output + "\n\n"

        query, output = self.task_representer.example_representer.encode(task.test_example)

        input_data.append({"role": "user", "content": prompt + query})

        output_data = {"role": "assistant", "content": output}

        return input_data, output_data

    def decode(self, input_data: MESSAGES, output_data: MESSAGE, **kwargs) -> Task:
        # Decoding logic for GPTTextMessagerepresenterFewShot is complex and depends on the specific encoding format.
        # This is a placeholder for the actual implementation.
        raise NotImplementedError(
            "Decoding for GPTTextMessagerepresenterFewShot is not implemented."
        )


class GPTTextImageMessagerepresenter(MessageRepresenter):
    def __init__(
        self,
        text_representer: TextTaskRepresenter = TextTaskRepresenter(),
        image_representer: ImageTaskRepresenter = ImageTaskRepresenter(),
        prompt: Optional[
            str
        ] = "Figure out the underlying transformation in the following examples and apply it to the test case. {description}Here are some examples from this transformation, your answer must follow the format.\n",
    ):
        self.prompt = prompt
        self.text_representer = text_representer
        self.image_representer = image_representer

    def encode(self, task: Task, **kwargs) -> Tuple[MESSAGES, MESSAGE]:
        input_data = []

        if hasattr(task, "description"):
            description = task.description
            description = f"\n\n A possible description of the transformation: \n\n{description}\n"
            prompt = self.prompt.format(description=description)
        else:
            prompt = self.prompt.format(description="")

        if isinstance(
            self.text_representer.example_representer.grid_representer,
            ConnectedComponentRepresenter,
        ):
            connected_component = kwargs.get(
                "connected_component",
                self.text_representer.example_representer.grid_representer.connected_component,
            )
            connected_component = (
                "including diagonals" if connected_component == 8 else "excluding diagonals"
            )
            prompt += f"The input-output grids are provided with both as image and as indices of connected shapes ({connected_component}) of the same color."
        elif isinstance(
            self.text_representer.example_representer.grid_representer, PythonListGridRepresenter
        ):
            prompt += "The input-output grids are provided both as image and as python arrays:\n"
        elif isinstance(
            self.text_representer.example_representer.grid_representer, CompositeRepresenter
        ):
            connected_component = kwargs.get(
                "connected_component",
                self.text_representer.example_representer.grid_representer.connected_component,
            )
            connected_component = (
                "including diagonals" if connected_component == 8 else "excluding diagonals"
            )
            prompt += f"The input-output grids are provided as both python arrays and as indices of connected shapes ({connected_component}) of the same color."

        input_data.append({"role": "system", "content": prompt})

        for j, example in enumerate(task.train_examples + [task.test_example]):
            content = []
            query, output = self.text_representer.example_representer.encode(example, **kwargs)

            content.append(
                {
                    "type": "text",
                    "text": query.replace("\nOUTPUT:\n", ""),
                }
            )

            input_image = self.image_representer.example_representer.grid_representer.encode(
                example.input, **kwargs
            )
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{input_image}"}}
            )
            if j != len(task.train_examples):
                output_image = self.image_representer.example_representer.grid_representer.encode(
                    example.output, **kwargs
                )
                content.append({"type": "text", "text": "\nOUTPUT:\n" + output})

                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{output_image}"},
                    }
                )
            else:
                test_content = []
                output_image = self.image_representer.example_representer.grid_representer.encode(
                    example.output, **kwargs
                )
                test_content.append({"type": "text", "text": "\nOUTPUT:\n" + output})

            input_data.append({"role": "user", "content": content})

        output_data = {
            "role": "assistant",
            "content": test_content,
        }

        return input_data, output_data

    def decode(self, input_data: MESSAGES, output_data: MESSAGE, **kwargs) -> Task:
        raise NotImplementedError("Decoding for GPTTextMessageRepresenterV2 is not implemented.")


class GPTTextImageMessageRepresenterFewShot(MessageRepresenter):
    def __init__(
        self,
        text_representer: TextTaskRepresenter = TextTaskRepresenter(),
        image_representer: ImageTaskRepresenter = ImageTaskRepresenter(),
        diff_representer: Optional[GridRepresenter] = DelimitedGridRepresenter(),
        prompt: Optional[
            str
        ] = "Figure out the underlying transformations in each task and complete the examples. You must follow the format.\n\n",
        disable_image: Optional[bool] = False,
        disable_text: Optional[bool] = False,
    ):
        self.prompt = prompt
        self.disable_image = disable_image
        self.disable_text = disable_text
        self.text_representer = text_representer
        self.image_representer = image_representer
        self.diff_representer = diff_representer

    def encode(self, task: Task, examples: List[Tuple[Task, str]]) -> Tuple[MESSAGES, MESSAGE]:
        input_data = []

        # if hasattr(task, "description"):
        #     description = task.description
        #     description = f"\n\n A possible description of the transformation: \n\n{description}\n"
        #     prompt = self.prompt.format(description=description)
        # else:
        #     prompt = self.prompt.format(description="")

        # if isinstance(self.text_representer.example_representer.grid_representer, ConnectedComponentRepresenter):
        #     connected_component = kwargs.get("connected_component", self.text_representer.example_representer.grid_representer.connected_component)
        #     connected_component = "including diagonals" if connected_component == 8 else "excluding diagonals"
        #     prompt += f"The input-output grids are provided with both as image and as indices of connected shapes ({connected_component}) of the same color."
        # elif isinstance(self.text_representer.example_representer.grid_representer, PythonListGridRepresenter):
        #     prompt += "The input-output grids are provided both as image and as python arrays:\n"
        # elif isinstance(self.text_representer.example_representer.grid_representer, CompositeRepresenter):
        #     connected_component = kwargs.get("connected_component", self.text_representer.example_representer.grid_representer.connected_component)
        #     connected_component = "including diagonals" if connected_component == 8 else "excluding diagonals"
        #     prompt += f"The input-output grids are provided as both python arrays and as indices of connected shapes ({connected_component}) of the same color."
        prompt = self.prompt
        input_data.append({"role": "system", "content": prompt})
        # Iterate over the examples provided for few-shot learning
        for example_task, example_output in examples:
            content = []
            for j, example in enumerate(example_task.train_examples + [example_task.test_example]):
                query, output = self.text_representer.example_representer.encode(example)
                if not self.disable_text:
                    content.append(
                        {
                            "type": "text",
                            "text": query.replace("\nOUTPUT:\n", ""),
                        }
                    )
                else:
                    content.append(
                        {
                            "type": "text",
                            "text": "\nINPUT:\n",
                        }
                    )

                if not self.disable_image:
                    input_image = (
                        self.image_representer.example_representer.grid_representer.encode(
                            example.input
                        )
                    )
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{input_image}"},
                        }
                    )
                if j != len(example_task.train_examples):
                    if not self.disable_text:
                        content.append({"type": "text", "text": "\nOUTPUT:\n" + output})
                    else:
                        content.append({"type": "text", "text": "\nOUTPUT:\n"})
                    if not self.disable_image:
                        output_image = (
                            self.image_representer.example_representer.grid_representer.encode(
                                example.output
                            )
                        )
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{output_image}"},
                            }
                        )

                    if np.shape(example.input) == np.shape(example.output):
                        diff = example.output - example.input
                        diff = np.where(diff != 0, example.output, diff)
                        encoded_diff = self.diff_representer.encode(diff)
                        if not self.disable_text:
                            content.append({"type": "text", "text": "\nDIFF:\n" + encoded_diff})
                        else:
                            content.append({"type": "text", "text": "\nDIFF:\n"})

                        if not self.disable_image:
                            diff_image = (
                                self.image_representer.example_representer.grid_representer.encode(
                                    diff
                                )
                            )
                            content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{diff_image}"},
                                }
                            )

            input_data.append({"role": "user", "content": content})
            # reasoning
            input_data.append({"role": "assistant", "content": example_output})

        content = []
        for j, example in enumerate(task.train_examples + [task.test_example]):
            query, output = self.text_representer.example_representer.encode(example)
            if not self.disable_text:
                content.append(
                    {
                        "type": "text",
                        "text": query.replace("\nOUTPUT:\n", ""),
                    }
                )
            else:
                content.append(
                    {
                        "type": "text",
                        "text": "\nINPUT:\n",
                    }
                )
            if not self.disable_image:
                input_image = self.image_representer.example_representer.grid_representer.encode(
                    example.input
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{input_image}"},
                    }
                )

            if j != len(task.train_examples):
                if not self.disable_text:
                    content.append({"type": "text", "text": "\nOUTPUT:\n" + output})
                else:
                    content.append({"type": "text", "text": "\nOUTPUT:\n"})

                if not self.disable_image:
                    output_image = (
                        self.image_representer.example_representer.grid_representer.encode(
                            example.output
                        )
                    )

                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{output_image}"},
                        }
                    )

                if np.shape(example.input) == np.shape(example.output):
                    diff = example.output - example.input
                    diff = np.where(diff != 0, example.output, diff)
                    encoded_diff = self.diff_representer.encode(diff)
                    if not self.disable_text:
                        content.append({"type": "text", "text": "\nDIFF:\n" + encoded_diff})
                    else:
                        content.append({"type": "text", "text": "\nDIFF:\n"})

                    if not self.disable_image:
                        diff_image = (
                            self.image_representer.example_representer.grid_representer.encode(diff)
                        )
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{diff_image}"},
                            }
                        )

        input_data.append({"role": "user", "content": content})

        output_data = [{}]

        return input_data, output_data


class TextMessageRepresenterFewShot(MessageRepresenter):
    def __init__(
        self,
        text_representer: TextTaskRepresenter = TextTaskRepresenter(),
        image_representer: ImageTaskRepresenter = ImageTaskRepresenter(),
        prompt: Optional[
            str
        ] = "Figure out the underlying transformations in each task and complete the examples. You must follow the format.\n\n",
    ):
        self.prompt = prompt
        self.text_representer = text_representer
        self.image_representer = image_representer

    def encode(
        self, task: Task, examples: List[Tuple[Task, str]], **kwargs
    ) -> Tuple[MESSAGES, MESSAGE]:
        input_data = []

        if hasattr(task, "description"):
            description = task.description
            description = f"\n\n A possible description of the transformation: \n\n{description}\n"
            prompt = self.prompt.format(description=description)
        else:
            prompt = self.prompt.format(description="")

        if isinstance(
            self.text_representer.example_representer.grid_representer,
            ConnectedComponentRepresenter,
        ):
            connected_component = kwargs.get(
                "connected_component",
                self.text_representer.example_representer.grid_representer.connected_component,
            )
            connected_component = (
                "including diagonals" if connected_component == 8 else "excluding diagonals"
            )
            prompt += f"The input-output grids are provided with both as image and as indices of connected shapes ({connected_component}) of the same color."
        elif isinstance(
            self.text_representer.example_representer.grid_representer, PythonListGridRepresenter
        ):
            prompt += "The input-output grids are provided both as image and as python arrays:\n"
        elif isinstance(
            self.text_representer.example_representer.grid_representer, CompositeRepresenter
        ):
            connected_component = kwargs.get(
                "connected_component",
                self.text_representer.example_representer.grid_representer.connected_component,
            )
            connected_component = (
                "including diagonals" if connected_component == 8 else "excluding diagonals"
            )
            prompt += f"The input-output grids are provided as both python arrays and as indices of connected shapes ({connected_component}) of the same color."

        input_data.append({"role": "system", "content": prompt})
        # Iterate over the examples provided for few-shot learning
        for example_task, example_output in examples:
            content = []
            for j, example in enumerate(example_task.train_examples + [example_task.test_example]):
                query, output = self.text_representer.example_representer.encode(example)

                content.append(
                    {
                        "type": "text",
                        "text": query.replace("\nOUTPUT:\n", ""),
                    }
                )

                if j != len(example_task.train_examples):
                    content.append({"type": "text", "text": "\nOUTPUT:\n" + output})

            input_data.append({"role": "user", "content": content})
            # reasoning
            input_data.append({"role": "assistant", "content": example_output})

        content = []
        for j, example in enumerate(task.train_examples + [task.test_example]):
            query, output = self.text_representer.example_representer.encode(example)

            content.append(
                {
                    "type": "text",
                    "text": query.replace("\nOUTPUT:\n", ""),
                }
            )

            if j != len(task.train_examples):
                content.append({"type": "text", "text": "\nOUTPUT:\n" + output})

        input_data.append({"role": "user", "content": content})

        output_data = [{}]

        return input_data, output_data


class GPTImageMessageRepresenterFewShot(MessageRepresenter):

    def __init__(
        self,
        text_representer: TextTaskRepresenter = TextTaskRepresenter(),
        image_representer: ImageTaskRepresenter = ImageTaskRepresenter(),
        prompt: Optional[
            str
        ] = "Figure out the underlying transformations in each task and complete the examples. You must follow the format.\n\n",
    ):
        self.prompt = prompt
        self.text_representer = text_representer
        self.image_representer = image_representer

    def encode(
        self, task: Task, examples: List[Tuple[Task, str]], **kwargs
    ) -> Tuple[MESSAGES, MESSAGE]:
        input_data = []

        if hasattr(task, "description"):
            description = task.description
            description = f"\n\n A possible description of the transformation: \n\n{description}\n"
            prompt = self.prompt.format(description=description)
        else:
            prompt = self.prompt.format(description="")

        if isinstance(
            self.text_representer.example_representer.grid_representer,
            ConnectedComponentRepresenter,
        ):
            connected_component = kwargs.get(
                "connected_component",
                self.text_representer.example_representer.grid_representer.connected_component,
            )
            connected_component = (
                "including diagonals" if connected_component == 8 else "excluding diagonals"
            )
            prompt += f"The input-output grids are provided with both as image and as indices of connected shapes ({connected_component}) of the same color."
        elif isinstance(
            self.text_representer.example_representer.grid_representer, PythonListGridRepresenter
        ):
            prompt += "The input-output grids are provided both as image and as python arrays:\n"
        elif isinstance(
            self.text_representer.example_representer.grid_representer, CompositeRepresenter
        ):
            connected_component = kwargs.get(
                "connected_component",
                self.text_representer.example_representer.grid_representer.connected_component,
            )
            connected_component = (
                "including diagonals" if connected_component == 8 else "excluding diagonals"
            )
            prompt += f"The input-output grids are provided as both python arrays and as indices of connected shapes ({connected_component}) of the same color."

        input_data.append({"role": "system", "content": prompt})
        # Iterate over the examples provided for few-shot learning
        for example_task, example_output in examples:
            content = []
            for j, example in enumerate(example_task.train_examples + [example_task.test_example]):

                query, output = self.text_representer.example_representer.encode(example)

                input_image = self.image_representer.example_representer.grid_representer.encode(
                    example.input
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{input_image}"},
                    }
                )
                if j != len(example_task.train_examples):
                    output_image = (
                        self.image_representer.example_representer.grid_representer.encode(
                            example.output
                        )
                    )
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{output_image}"},
                        }
                    )

            input_data.append({"role": "user", "content": content})
            # reasoning
            input_data.append({"role": "assistant", "content": example_output})

        content = []
        for j, example in enumerate(task.train_examples + [task.test_example]):
            query, output = self.text_representer.example_representer.encode(example)

            input_image = self.image_representer.example_representer.grid_representer.encode(
                example.input
            )
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{input_image}"}}
            )

            if j != len(task.train_examples):
                output_image = self.image_representer.example_representer.grid_representer.encode(
                    example.output
                )

                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{output_image}"},
                    }
                )

        input_data.append({"role": "user", "content": content})

        output_data = [{}]

        return input_data, output_data


class GPTTextImageCodeMessageRepresenterFewShot(MessageRepresenter):
    def __init__(
        self,
        text_representer: TextTaskRepresenter = TextTaskRepresenter(),
        image_representer: ImageTaskRepresenter = ImageTaskRepresenter(),
        prompt: Optional[
            str
        ] = "Figure out the underlying transformations in each task and complete the examples. You must follow the format.\n\n",
        disable_image: Optional[bool] = False,
    ):
        self.prompt = prompt
        self.disable_image = disable_image
        print("disable_image", self.disable_image)
        self.text_representer = text_representer
        self.image_representer = image_representer

    def encode(
        self, task: Task, task_reasoning: str, examples: List[Tuple[Task, str, str]]
    ) -> Tuple[MESSAGES, MESSAGE]:
        input_data = []

        # if hasattr(task, "description"):
        #     description = task.description
        #     description = f"\n\n A possible description of the transformation: \n\n{description}\n"
        #     prompt = self.prompt.format(description=description)
        # else:
        #     prompt = self.prompt.format(description="")

        # if isinstance(self.text_representer.example_representer.grid_representer, ConnectedComponentRepresenter):
        #     connected_component = kwargs.get("connected_component", self.text_representer.example_representer.grid_representer.connected_component)
        #     connected_component = "including diagonals" if connected_component == 8 else "excluding diagonals"
        #     prompt += f"The input-output grids are provided with both as image and as indices of connected shapes ({connected_component}) of the same color."
        # elif isinstance(self.text_representer.example_representer.grid_representer, PythonListGridRepresenter):
        #     prompt += "The input-output grids are provided both as image and as python arrays:\n"
        # elif isinstance(self.text_representer.example_representer.grid_representer, CompositeRepresenter):
        #     connected_component = kwargs.get("connected_component", self.text_representer.example_representer.grid_representer.connected_component)
        #     connected_component = "including diagonals" if connected_component == 8 else "excluding diagonals"
        #     prompt += f"The input-output grids are provided as both python arrays and as indices of connected shapes ({connected_component}) of the same color."

        prompt = self.prompt

        input_data.append({"role": "system", "content": prompt})
        # Iterate over the examples provided for few-shot learning
        for example_task, reasoning, example_output in examples:
            content = []
            for j, example in enumerate(example_task.train_examples + [example_task.test_example]):
                query, output = self.text_representer.example_representer.encode(example)

                content.append(
                    {
                        "type": "text",
                        "text": query.replace("\nOUTPUT:\n", ""),
                    }
                )
                if not self.disable_image:
                    input_image = (
                        self.image_representer.example_representer.grid_representer.encode(
                            example.input
                        )
                    )
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{input_image}"},
                        }
                    )
                if j != len(example_task.train_examples):

                    content.append({"type": "text", "text": "\nOUTPUT:\n" + output})

                    if not self.disable_image:
                        output_image = (
                            self.image_representer.example_representer.grid_representer.encode(
                                example.output
                            )
                        )

                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{output_image}"},
                            }
                        )

            input_data.append(
                {
                    "role": "user",
                    "content": content
                    + [{"type": "text", "text": "\n\n====REASONING FOR CODE=====\n\n" + reasoning}],
                }
            )
            # reasoning
            input_data.append({"role": "assistant", "content": example_output})

        content = []
        for j, example in enumerate(task.train_examples + [task.test_example]):
            query, output = self.text_representer.example_representer.encode(example)

            content.append(
                {
                    "type": "text",
                    "text": query.replace("\nOUTPUT:\n", ""),
                }
            )
            if not self.disable_image:
                input_image = self.image_representer.example_representer.grid_representer.encode(
                    example.input
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{input_image}"},
                    }
                )

            if j != len(task.train_examples):
                content.append({"type": "text", "text": "\nOUTPUT:\n" + output})

                if not self.disable_image:
                    output_image = (
                        self.image_representer.example_representer.grid_representer.encode(
                            example.output
                        )
                    )
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{output_image}"},
                        }
                    )

        input_data.append(
            {
                "role": "user",
                "content": content
                + [
                    {"type": "text", "text": "\n\n====REASONING FOR CODE=====\n\n" + task_reasoning}
                ],
            }
        )

        output_data = [{}]

        return input_data, output_data


class GPTCodeDebuggerMessager(MessageRepresenter):

    def __init__(
        self,
        text_representer: TextTaskRepresenter = TextTaskRepresenter(),
        prompt: str = "You are a debugging assistant. Please debug the code provided below.",
    ):
        self.prompt = prompt
        self.text_representer = text_representer

    def encode(self, task: Task, reasoning: str, code: str, error_message: str):
        # Prepare the input message for the model
        # system
        input_messages = [{"role": "system", "content": self.prompt}]

        demonstrations, query, output = self.text_representer.encode(task)
        query = demonstrations + "\n" + query.replace("\nOUTPUT:\n", "")

        content = []
        content.append(
            {
                "type": "text",
                "text": query
                + "\n\n====REASONING FOR CODE=====\n\n"
                + reasoning
                + "\n\n"
                + "```python\n"
                + code
                + "\n```",
            }
        )

        content.append(
            {
                "type": "text",
                "text": "\n\n Here is the error message:\n\n"
                + error_message
                + "\n\n Can you now give the debugged version of the code? Remember, the implementation must contain the ExampleRepresenter() class as that is used for testing. Do not make up your own Class names for the representer.",
            }
        )
        input_messages.append({"role": "user", "content": content})

        return input_messages, None


class GPTTextMessageRepresenterForBarc(MessageRepresenter):
    def __init__(
        self,
        prompt: Optional[
            str
        ] = "You are a world-class puzzle solver with exceptional pattern recognition skills. Your task is to analyze puzzles, spot patterns, and provide direct solutions.",
        task_representer: TaskRepresenter = TextTaskRepresenter(),
    ):
        self.prompt = prompt
        self.task_representer = task_representer
        # if example_representer is not None:
        #     self.task_representer.example_representer = example_representer(
        #                 io_sep=" -> ",
        #                 input_header="",
        #                 output_header="",
        #                 grid_representer=PythonListGridRepresenter
        #             )

    def encode(self, task: Task, **kwargs) -> Tuple[MESSAGES, MESSAGE]:
        input_data = []

        input_data.append({"role": "system", "content": self.prompt})

        content = "Given input-output grid pairs as reference examples, carefully observe the patterns to predict the output grid for new test input. Each pair follows the same transformation rule. Grids are 2D arrays represented as strings, with cells (colors) separated by spaces and rows by newlines.\nHere are the input and output grids for the reference examples:\n"

        for i, example in enumerate(task.train_examples):
            content += f"Example {i + 1}:\n"
            query, output = self.task_representer.example_representer.encode(example, **kwargs)
            if query is None or output is None:
                return None, None
            content += query + output + "\n"

        content += "\n\nHere is the input grid for the test example:\n"

        query, output = self.task_representer.example_representer.encode(
            task.test_example, **kwargs
        )

        query = query.replace("Output:", "")

        content += query + "Directly provide the output grids corresponding to the given test input grids, based on the patterns observed in the reference examples."

        input_data.append({"role": "user", "content": content})

        output = f"The output grid for the test input grid is:\n\n```\n{output}\n```"

        output_data = {"role": "assistant", "content": output}

        return input_data, output_data

    def decode(self, input_data: MESSAGES, output_data: MESSAGE, **kwargs) -> Task:
        raise NotImplementedError("Decoding for GPTTextMessageRepresenterV2 is not implemented.")

    def __repr__(self) -> str:
        return f"GPTTextMessageRepresenterForBarc(prompt={self.prompt!r}, task_representer={repr(self.task_representer)})"

    def __str__(self) -> str:
        return repr(self)




if __name__ == "__main__":
    print("Running tests")
    grid = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
    example = Example(input=grid, output=grid)
    task = Task(test_example=example, train_examples=[example])

    representer  = GPTTextMessageRepresenterForBarc(
        task_representer=TextTaskRepresenter(
            example_representer=TextExampleRepresenter(
        grid_representer=WordGridRepresenter(),
        input_header="Input:\n",
        output_header="\nOutput:\n",
        io_sep="\n"
        )
    )
    )

    input, output = representer.encode(task)
    breakpoint()

    representer = GPTTextMessagerepresenter()
    representer = GPTTextMessageRepresenterV2()
    breakpoint()
    input, output = representer.encode(task)
    print(input)
    html_output = representer.display(input)
    # Write to an HTML file
    with open("chat_view.html", "w", encoding="utf-8") as file:
        file.write(html_output)

    # representer = GPTTextMessageRepresenterV2(task_representer=TextTaskRepresenter(example_representer=TextExampleRepresenter(grid_representer=ConnectedComponentRepresenter())))

    representer = GPTTextImageMessagerepresenter()
    input, output = representer.encode(task)
    html_output = representer.display(input + [output])
    # Write to an HTML file
    with open("chat_view_w_image.html", "w", encoding="utf-8") as file:
        file.write(html_output)
