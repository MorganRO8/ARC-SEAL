import importlib.util
import sys
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
