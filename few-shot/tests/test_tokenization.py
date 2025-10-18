import importlib.util
import sys
from pathlib import Path

import torch
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location("self_edit_module", ROOT / "self-edit.py")
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_tokenize_and_process = _MODULE._tokenize_and_process


class FakeTokenizer:
    def __call__(self, text, truncation=True, return_tensors=None):
        token_ids = torch.tensor([[c for c in text.encode("utf-8")]], dtype=torch.long)
        attention_mask = torch.ones_like(token_ids)
        return {"input_ids": token_ids, "attention_mask": attention_mask}

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        **kwargs,
    ):
        parts = []
        for index, message in enumerate(messages):
            role = message.get("role", "user")
            content = message.get("content", "")
            parts.append(f"<{role}>:\n{content}")
            if index != len(messages) - 1:
                parts.append("\n")
        text = "".join(parts)
        if add_generation_prompt:
            if text:
                text += "\n"
            text += "<assistant>:\n"
        if tokenize:
            return [c for c in text.encode("utf-8")]
        return text


class TokenizeAndProcessTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def test_masks_prompt_tokens_for_final_assistant(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        result = _tokenize_and_process(
            full_text,
            self.tokenizer,
            messages=messages,
        )
        prompt_text = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer(
            prompt_text,
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        prompt_len = prompt_ids.numel()
        self.assertTrue(prompt_len > 0)
        labels = result["labels"].tolist()
        input_ids = result["input_ids"].tolist()
        self.assertEqual(labels[:prompt_len], [-100] * prompt_len)
        self.assertEqual(labels[prompt_len:], input_ids[prompt_len:])

    def test_handles_conversation_without_prompt_messages(self):
        messages = [
            {"role": "assistant", "content": "standalone"},
        ]
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        result = _tokenize_and_process(
            full_text,
            self.tokenizer,
            messages=messages,
        )
        labels = result["labels"].tolist()
        self.assertNotIn(-100, labels)


if __name__ == "__main__":
    unittest.main()
