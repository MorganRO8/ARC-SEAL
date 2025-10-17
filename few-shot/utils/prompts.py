
# for self-edit.py
from textwrap import dedent


self_edit_prompt = """
You are configuring a model training pipeline by selecting from predefined tools.

You must make two decisions:

1. **Data Generation Tools** — For each of the following, choose true or false:
    - use_basic_augmentations
    - use_size_augmentations
    - use_chain_augmentations
    - use_repeat_augmentations

2. **Training Configuration** — Choose one of:
    - "train_using_all_tokens"
    - "train_using_output_tokens"

Also specify:
    - learning_rate (float)
    - num_train_epochs (integer)

### Output Format

Respond with a valid JSON object. Do not include any explanation, markdown, or extra text. Use lowercase `true`/`false` for booleans and ensure correct JSON syntax.

Example output:

{
  "data_generation": {
    "use_basic_augmentations": ...,
    "use_size_augmentations": ...,
    "use_chain_augmentations": ...,
    "use_repeat_augmentations": ...
  },
  "training": {
    "strategy": ...,
    "learning_rate": ...,
    "num_train_epochs": ...
  }
}
"""

system_message = "You are a helpful assistant that provide the correct output for the given task immediately."


python_solver_system_prompt = dedent(
    """
    You are an ARC task solving assistant that writes pure Python 3 code.
    Produce deterministic solutions that work for the provided train/test grids.
    Only import from the Python standard library. You may use: typing (List), collections, itertools,
    functools, math, statistics, heapq, and copy. Avoid any file system, network, or OS side effects.
    Your response must be valid Python source code without additional commentary.
    """
).strip()

python_solver_function_signature = "def solve(grid: List[List[int]]) -> List[List[int]]:"

python_solver_function_docstring = dedent(
    """
    Solve the ARC task for the provided test input grid.

    Args:
        grid: The test input grid represented as a list of lists of integers.

    Returns:
        A list of lists of integers representing the predicted output grid.
    """
).strip()

python_solver_user_template = dedent(
    """
    {description}Training examples (Python dictionaries with \"input\" and \"output\" grids):
    {train_examples}

    Test example data:
    {test_example}

    Implement the function below. Include the provided docstring exactly as the first statement of the
    function and limit your response to executable Python. Helper functions are allowed, but the entry
    point must match the signature.

    Signature: {function_signature}
    Docstring text:
    {function_docstring}
    """
).strip()


def build_python_solver_response_stub() -> str:
    docstring_lines = python_solver_function_docstring.splitlines()
    indented_docstring = "\n".join("    " + line if line else "    " for line in docstring_lines)
    return (
        f"{python_solver_function_signature}\n"
        f"{indented_docstring}\n"
        "    raise NotImplementedError(\"solver stub\")\n"
    )


python_solver_response_stub = build_python_solver_response_stub()
