import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
FEWSHOT_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(FEWSHOT_ROOT) not in sys.path:
    sys.path.append(str(FEWSHOT_ROOT))

MODULE_PATH = FEWSHOT_ROOT / "iterative_code_trainer.py"
SPEC = importlib.util.spec_from_file_location("iterative_code_trainer", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

EXECUTOR_PATH = FEWSHOT_ROOT / "utils" / "python_executor.py"
executor_spec = importlib.util.spec_from_file_location("python_executor", EXECUTOR_PATH)
executor_module = importlib.util.module_from_spec(executor_spec)
assert executor_spec is not None and executor_spec.loader is not None
sys.modules[executor_spec.name] = executor_module
executor_spec.loader.exec_module(executor_module)

SolverResult = executor_module.SolverResult

AttemptLog = module.AttemptLog
compute_reward = module.compute_reward
decode_model_outputs = module.decode_model_outputs
_tokenize_chat_transcript = module._tokenize_chat_transcript


class DummyTokenizer:
    def __init__(self, prompt_ids):
        self._prompt_ids = torch.tensor([prompt_ids], dtype=torch.long)

    def __call__(self, text, return_tensors="pt", truncation=False):
        return {
            "input_ids": self._prompt_ids.clone(),
            "attention_mask": torch.ones_like(self._prompt_ids),
        }

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(idx)) for idx in ids)


class ChatTokenizer:
    def __call__(self, text, return_tensors="pt", truncation=False):
        # Construct ids ending with the assistant header token pair (128007, 271)
        input_ids = torch.tensor([[11, 22, 33, 128007, 271, 44, 55]])
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


class IterativeCodeTrainerTests(unittest.TestCase):
    def test_compute_reward_success(self):
        predicted = np.array([[1, 2], [3, 4]])
        expected = np.array([[1, 2], [3, 4]])
        result = SolverResult(success=True, output=predicted)

        success, reward, normalized = compute_reward(result, expected)

        self.assertTrue(success)
        self.assertEqual(reward, 1.0)
        self.assertEqual(normalized, predicted.tolist())

    def test_compute_reward_failure(self):
        predicted = np.array([[1]])
        expected = np.array([[0]])
        result = SolverResult(success=True, output=predicted)

        success, reward, normalized = compute_reward(result, expected)

        self.assertFalse(success)
        self.assertEqual(reward, 0.0)
        self.assertEqual(normalized, predicted.tolist())

    def test_decode_model_outputs_strips_prompt(self):
        tokenizer = DummyTokenizer([101, 102, 103])
        sequences = torch.tensor([[101, 102, 103, 201, 202]])

        decoded = decode_model_outputs(tokenizer, "dummy", sequences)

        self.assertEqual(decoded, ["201 202"])

    def test_tokenize_chat_transcript_masks_prefix(self):
        tokenizer = ChatTokenizer()
        tokenized = _tokenize_chat_transcript("dummy", tokenizer)

        self.assertTrue((tokenized["labels"][:5] == -100).all())
        self.assertTrue((tokenized["labels"][5:] != -100).all())

    def test_attempt_log_serialization_roundtrip(self):
        attempt = AttemptLog(
            task_name="task-1",
            prompt_messages=[{"role": "system", "content": "hi"}],
            prompt_text="prompt",
            raw_response="code",
            code="def solve(): pass",
            success=True,
            reward=1.0,
            execution={"success": "True"},
            predicted_output=[[1]],
            target_output=[[1]],
        )

        payload = attempt.to_json()
        restored = json.loads(payload)

        self.assertEqual(restored["task_name"], "task-1")
        self.assertEqual(restored["execution"]["success"], "True")
        self.assertEqual(restored["prompt_messages"][0]["role"], "system")
