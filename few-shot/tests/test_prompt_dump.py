from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arclib.arc import Example, Task
from dump_code_prompt import build_code_mode_prompt_for_task
from utils.solver_helpers import HELPER_NAMESPACE


class _DummyTokenizer:
    """Minimal tokenizer stub for prompt rendering tests."""

    chat_template = ""
    all_special_tokens = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **_: object,
    ) -> str:
        parts = []
        for message in messages:
            parts.append(f"{message['role']}: {message['content']}")
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)


def _build_dummy_task() -> Task:
    train = [
        Example(
            input=np.array([[1, 0], [0, 0]]),
            output=np.array([[0, 1], [0, 0]]),
        ),
        Example(
            input=np.array([[2, 2], [0, 0]]),
            output=np.array([[0, 0], [2, 2]]),
        ),
    ]
    test = Example(
        input=np.array([[3, 0], [0, 0]]),
        output=np.array([[0, 0], [0, 3]]),
    )
    return Task(train_examples=train, test_example=test, name="dummy-task-0")


class PromptDumpTests(unittest.TestCase):
    def test_prompt_without_helpers_mentions_absence(self) -> None:
        task = _build_dummy_task()
        tokenizer = _DummyTokenizer()

        result = build_code_mode_prompt_for_task(task, tokenizer)

        self.assertIn("No ARC utility functions", result.prompt_text)
        self.assertIsNone(result.helper_library)

    def test_prompt_with_helpers_mentions_namespace(self) -> None:
        task = _build_dummy_task()
        tokenizer = _DummyTokenizer()

        result = build_code_mode_prompt_for_task(task, tokenizer, include_helpers=True)

        self.assertIn(HELPER_NAMESPACE, result.prompt_text)
        self.assertIn("utility functions", result.prompt_text.lower())
        self.assertIsNotNone(result.helper_library)


if __name__ == "__main__":
    unittest.main()
