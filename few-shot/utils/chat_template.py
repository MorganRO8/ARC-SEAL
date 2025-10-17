"""Utilities for adapting chat templates to models with thinking support."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class ThinkingConfig:
    """Describe optional thinking/chain-of-thought support for a tokenizer."""

    enabled: bool
    apply_chat_kwargs: Mapping[str, Any]
    reason: Optional[str] = None


_THINK_TOKENS = (
    "<think>",
    "</think>",
    "<|begin_of_thought|>",
    "<|end_of_thought|>",
    "<|begin_think|>",
    "<|end_think|>",
    "<|start_think|>",
    "<|stop_think|>",
)


def detect_thinking_support(tokenizer) -> ThinkingConfig:
    """Best-effort detection of thinking support for a chat tokenizer.

    The heuristics check the model's chat template and registered special tokens
    to decide whether explicit "thinking" or reasoning modes are available. When
    a template exposes an ``enable_thinking`` switch we surface it so callers can
    toggle the mode via ``apply_chat_template``.
    """

    template = getattr(tokenizer, "chat_template", "") or ""
    template_lower = template.lower()

    apply_kwargs: Dict[str, Any] = {}
    reason_parts = []

    if "enable_thinking" in template:
        apply_kwargs["enable_thinking"] = True
        reason_parts.append("enable_thinking")

    if any(token.lower() in template_lower for token in _THINK_TOKENS):
        reason_parts.append("think_tokens")

    special_tokens = [tok.lower() for tok in getattr(tokenizer, "all_special_tokens", [])]
    if any(token.lower() in special_tokens for token in _THINK_TOKENS):
        if "think_tokens" not in reason_parts:
            reason_parts.append("think_tokens")

    enabled = bool(reason_parts)
    reason = ", ".join(dict.fromkeys(reason_parts)) if reason_parts else None

    return ThinkingConfig(enabled=enabled, apply_chat_kwargs=apply_kwargs, reason=reason)
