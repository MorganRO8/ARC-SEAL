import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location("self_edit_module", ROOT / "self-edit.py")
assert _SPEC and _SPEC.loader
self_edit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(self_edit)


class FormatProgressDetailsTest(unittest.TestCase):
    def test_includes_changed_ratio_and_penalties(self) -> None:
        attempt = {
            "cells_requiring_change": 6,
            "correct_changed_cells": 4,
            "mismatched_unchanged_cells": 2,
            "incorrect_cells": 5,
        }

        summary = self_edit._format_progress_details(attempt)

        self.assertIn("4/6 changed", summary)
        self.assertIn("penalized 2", summary)
        # incorrect cells beyond the penalized unchanged ones should be listed
        self.assertIn("incorrect 3", summary)

    def test_falls_back_to_total_cells(self) -> None:
        attempt = {"correct_cells": 25, "total_cells": 36}

        summary = self_edit._format_progress_details(attempt)

        self.assertEqual(summary, " (25/36 cells)")

    def test_returns_empty_string_when_no_metrics(self) -> None:
        self.assertEqual(self_edit._format_progress_details({}), "")


class CodeModeAttemptRegistrationTest(unittest.TestCase):
    def test_run_code_mode_attempt_bound_outside_helper(self) -> None:
        source_path = ROOT / "self-edit.py"
        source = source_path.read_text()
        tree = ast.parse(source, filename=str(source_path))

        bound_outside_helper = False

        def visit(node: ast.AST, parents: List[ast.AST]) -> None:
            nonlocal bound_outside_helper

            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id == "run_code_mode_attempt"
                        and isinstance(node.value, ast.Name)
                        and node.value.id == "_run_code_mode_attempt"
                    ):
                        if not any(
                            isinstance(parent, ast.FunctionDef)
                            and parent.name == "_run_code_mode_attempt"
                            for parent in parents
                        ):
                            bound_outside_helper = True

            for child in ast.iter_child_nodes(node):
                visit(child, parents + [node])

        visit(tree, [])

        self.assertTrue(
            bound_outside_helper,
            "run_code_mode_attempt should be bound outside _run_code_mode_attempt",
        )


if __name__ == "__main__":
    unittest.main()
