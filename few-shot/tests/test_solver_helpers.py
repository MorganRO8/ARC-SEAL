import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arclib.messagers import PythonSolverMessageRepresenter
from utils.python_executor import run_solver
from utils.solver_helpers import (
    detect_symmetries,
    find_filled_rectangles,
    get_components_by_color,
    get_connected_components,
    get_helper_library,
    place_relative_object,
    reflect_grid,
    rotate_grid,
    to_relative_coordinates,
)


class DummyExample:
    def __init__(self, grid):
        self.input = np.array(grid)
        self.output = np.array(grid)


class DummyTask:
    def __init__(self):
        self.train_examples = [
            DummyExample([[1, 0], [0, 1]]),
            DummyExample([[2, 2], [0, 0]]),
        ]
        self.test_example = DummyExample([[0, 1], [1, 0]])
        self.description = "Dummy task"
        self.name = "dummy"


class SolverHelpersTests(unittest.TestCase):
    def test_component_and_rectangle_detection(self) -> None:
        grid = [
            [1, 1, 0, 2, 2],
            [1, 0, 0, 2, 2],
            [0, 0, 3, 3, 3],
        ]

        comps = get_connected_components(grid)
        colors = sorted({comp["color"] for comp in comps})
        self.assertEqual(colors, [0, 1, 2, 3])

        blue_components = get_components_by_color(grid, color=2)
        self.assertEqual(len(blue_components), 1)
        self.assertEqual(blue_components[0]["bbox"], (0, 3, 1, 4))

        rectangles = find_filled_rectangles(grid, color=2)
        self.assertEqual(len(rectangles), 1)
        self.assertEqual(rectangles[0]["width"], 2)
        self.assertEqual(rectangles[0]["height"], 2)

    def test_symmetry_and_transforms(self) -> None:
        grid = [
            [1, 0, 1],
            [2, 0, 2],
            [1, 0, 1],
        ]
        symmetries = detect_symmetries(grid)
        self.assertTrue(symmetries["horizontal"])
        self.assertTrue(symmetries["vertical"])

        rotated = rotate_grid(grid, k=1)
        self.assertEqual(rotated[0], [1, 2, 1])

        reflected = reflect_grid(grid, axis="vertical")
        self.assertEqual(reflected, grid)

        square = [(0, 0), (0, 1), (1, 0), (1, 1)]
        relative = to_relative_coordinates([(3, 4), (3, 5), (4, 4), (4, 5)])
        self.assertEqual(relative, square)

        base = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        placed = place_relative_object(base, square, color=5, top_left=(1, 1))
        self.assertEqual(placed[1][1:], [5, 5])
        self.assertEqual(placed[2][1:], [5, 5])

    def test_helper_library_injection_success(self) -> None:
        helper_library = get_helper_library()
        code = (
            "from typing import List\n\n"
            "def solve(grid: List[List[int]]) -> List[List[int]]:\n"
            "    return ARC_HELPERS.reflect_grid(grid, axis=\"vertical\")\n"
        )
        train_examples = [{"input": [[1, 2]], "output": [[2, 1]]}]
        test_grid = [[3, 4]]

        result = run_solver(
            code,
            train_examples,
            test_grid,
            helper_library=helper_library,
        )
        self.assertTrue(result.success)
        self.assertFalse(result.helper_error)
        self.assertEqual(result.output.tolist(), [[4, 3]])

    def test_helper_error_tagging(self) -> None:
        helper_library = get_helper_library()
        code = (
            "from typing import List\n\n"
            "def solve(grid: List[List[int]]) -> List[List[int]]:\n"
            "    return ARC_HELPERS.reflect_grid(grid, axis=\"diagonal\")\n"
        )
        result = run_solver(
            code,
            [],
            [[0]],
            helper_library=helper_library,
        )
        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "HelperExecutionError")
        self.assertTrue(result.helper_error)
        self.assertEqual(result.helper_exception_type, "ValueError")

    def test_prompt_helper_section(self) -> None:
        task = DummyTask()
        representer = PythonSolverMessageRepresenter()
        helper_library = get_helper_library()

        messages_with_helpers, _ = representer.encode(
            task,
            size_inference=((2, 2), {"method": "manual"}),
            execution_limits={},
            helper_summary=helper_library.prompt_summary,
            helper_groups=helper_library.prompt_groups,
            helper_namespace=helper_library.namespace,
        )
        helper_prompt = messages_with_helpers[1]["content"]
        self.assertIn(helper_library.namespace, helper_prompt)
        self.assertIn("utility functions", helper_prompt.lower())
        for group_text in helper_library.prompt_groups:
            self.assertIn(group_text.splitlines()[0], helper_prompt)

        messages_without_helpers, _ = representer.encode(
            task,
            size_inference=((2, 2), {"method": "manual"}),
            execution_limits={},
            helper_summary=None,
            helper_groups=None,
        )
        no_helper_prompt = messages_without_helpers[1]["content"]
        self.assertIn("No ARC utility functions", no_helper_prompt)


if __name__ == "__main__":
    unittest.main()
