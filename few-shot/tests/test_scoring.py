"""Tests for reward scaling in ``score_grid_prediction``."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location("self_edit_module", ROOT / "self-edit.py")
assert _SPEC and _SPEC.loader  # for mypy/linters
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
score_grid_prediction = _MODULE.score_grid_prediction


class ScoreGridPredictionTests(unittest.TestCase):
    def test_prefers_input_reference_and_rewards_changes(self) -> None:
        input_grid = np.array([[0, 0], [0, 0]])
        expected = np.array([[0, 1], [0, 0]])
        predicted = expected.copy()

        reward, correct, total, reason, normalized, details = score_grid_prediction(
            predicted,
            expected,
            reference_input=input_grid,
        )

        self.assertEqual(reason, None)
        np.testing.assert_array_equal(normalized, expected)
        self.assertAlmostEqual(reward, 1.0)
        self.assertEqual(correct, 1)
        self.assertEqual(total, 1)
        self.assertEqual(details["reference_type"], "input")
        self.assertEqual(details["cells_requiring_change"], 1)
        self.assertTrue(details["exact_match"])

    def test_returns_zero_reward_when_changes_not_matched(self) -> None:
        input_grid = np.array([[0, 0], [0, 0]])
        expected = np.array([[0, 1], [0, 0]])
        predicted = input_grid.copy()

        reward, correct, total, reason, _, details = score_grid_prediction(
            predicted,
            expected,
            reference_input=input_grid,
        )

        self.assertEqual(reason, None)
        self.assertAlmostEqual(reward, 0.0)
        self.assertEqual(correct, 0)
        self.assertEqual(total, 1)
        self.assertEqual(details["reference_type"], "input")
        self.assertFalse(details["exact_match"])

    def test_identity_outputs_use_overall_accuracy(self) -> None:
        input_grid = np.array([[1, 2], [3, 4]])
        expected = input_grid.copy()
        predicted = expected.copy()
        predicted[0, 0] = 9

        reward, correct, total, reason, _, details = score_grid_prediction(
            predicted,
            expected,
            reference_input=input_grid,
        )

        self.assertIsNone(reason)
        self.assertEqual(details["cells_requiring_change"], 0)
        self.assertEqual(total, expected.size)
        self.assertEqual(correct, expected.size - 1)
        self.assertAlmostEqual(reward, (expected.size - 1) / expected.size)
        self.assertFalse(details["exact_match"])

    def test_penalises_errors_on_unchanged_cells(self) -> None:
        input_grid = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        expected = input_grid.copy()
        expected[1, 1] = 5

        predicted = expected.copy()
        predicted[0, 0] = 3  # Incorrect change outside the required delta.

        reward, correct, total, reason, _, details = score_grid_prediction(
            predicted,
            expected,
            reference_input=input_grid,
        )

        self.assertIsNone(reason)
        self.assertEqual(correct, 1)
        self.assertEqual(total, 1)
        self.assertEqual(details["mismatched_unchanged_cells"], 1)
        self.assertLess(reward, 1.0)
        self.assertAlmostEqual(
            reward,
            (expected.size - 1) / expected.size,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
