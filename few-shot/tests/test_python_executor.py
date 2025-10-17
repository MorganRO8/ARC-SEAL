import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arclib.arc import Example
from utils.python_executor import extract_solver_code, run_solver


PYTHON_SOLVER_DOCSTRING = (
    "Solve the ARC task for the provided test input grid.\n\n"
    "    Args:\n"
    "        grid: The test input grid represented as a list of lists of integers.\n\n"
    "    Returns:\n"
    "        A list of lists of integers representing the predicted output grid.\n"
)


class PythonExecutorTests(unittest.TestCase):
    def test_extract_solver_code_prefers_fenced_block(self) -> None:
        response = (
            "Some preamble.\n```python\n# comment\n\n"
            "def solve(grid):\n    return grid\n```\nTrailing"
        )
        code = extract_solver_code(response)
        self.assertIn("def solve", code)
        self.assertNotIn("```", code)

    def test_run_solver_success(self) -> None:
        code = (
            "from typing import List\n\n"
            "def solve(grid: List[List[int]]) -> List[List[int]]:\n"
            f"    \"\"\"{PYTHON_SOLVER_DOCSTRING}\"\"\"\n"
            "    return [row[:] for row in grid]\n"
        )
        train_examples = [Example(np.array([[1]]), np.array([[1]]))]
        test_grid = np.array([[0, 1], [1, 0]])

        result = run_solver(code, train_examples, test_grid)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.output)
        self.assertTrue(np.array_equal(result.output, test_grid))

    def test_run_solver_timeout(self) -> None:
        code = (
            "from typing import List\n\n"
            "def solve(grid: List[List[int]]) -> List[List[int]]:\n"
            f"    \"\"\"{PYTHON_SOLVER_DOCSTRING}\"\"\"\n"
            "    while True:\n"
            "        pass\n"
        )
        result = run_solver(code, [], np.array([[0]]), timeout=0.5, cpu_time_limit_s=1)
        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "Timeout")

    def test_run_solver_missing_entry_point(self) -> None:
        result = run_solver("print('hello')", [], np.array([[0]]))
        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "MissingEntryPoint")


if __name__ == "__main__":
    unittest.main()
