"""Utilities for sanitizing and formatting generated solver code."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FormattingResult:
    """Details about an attempted code-formatting pass."""

    formatted_code: Optional[str]
    changed: bool
    tool: str
    error: Optional[str] = None


def format_with_black(code: str) -> FormattingResult:
    """Try to reformat *code* using Black.

    Returns a :class:`FormattingResult` describing whether the formatter ran
    successfully and if it modified the text.  When Black is unavailable or the
    input cannot be parsed, the original code is returned and ``changed`` is
    ``False``.
    """

    try:
        import black  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive import guard
        return FormattingResult(
            formatted_code=code,
            changed=False,
            tool="black",
            error=f"Formatter unavailable: {exc}",
        )

    try:
        formatted = black.format_str(code, mode=black.Mode())
    except Exception as exc:  # pragma: no cover - rely on Black's own tests
        return FormattingResult(
            formatted_code=code,
            changed=False,
            tool="black",
            error=f"Formatter failed: {exc}",
        )

    if formatted != code:
        return FormattingResult(
            formatted_code=formatted,
            changed=True,
            tool="black",
            error=None,
        )

    return FormattingResult(formatted_code=code, changed=False, tool="black", error=None)


def try_fix_indentation(code: str) -> FormattingResult:
    """Attempt to automatically correct indentation issues in *code*.

    Currently this delegates to :func:`format_with_black`, but the helper is kept
    separate so additional strategies can be layered in the future without
    touching the call sites.
    """

    return format_with_black(code)
