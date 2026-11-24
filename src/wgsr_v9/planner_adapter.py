"""Convert a trained planner completion to the image-executor interface."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


_THINK_RE = re.compile(r"^\s*<think>\s*(?P<body>.*?)\s*</think>\s*$", re.DOTALL)
_GROWTH_RE = re.compile(r"^\s*<GROWTH_COT>\s*(?P<body>.*?)\s*</GROWTH_COT>\s*$", re.DOTALL)


def _completion_text(completion: str | Mapping[str, Any]) -> str:
    if isinstance(completion, str):
        return completion
    if not isinstance(completion, Mapping):
        raise TypeError("planner completion must be text or a mapping")
    for key in ("reasoning_content", "content", "text"):
        value = completion.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError("planner completion has no non-empty reasoning_content, content, or text")


def growth_cot_body(completion: str | Mapping[str, Any]) -> str:
    """Extract one planner ``<think>`` body for downstream adaptation.

    A completion that is already in the executor format is accepted for
    resumable export. Nested or empty wrappers are rejected so an accidental
    model preamble cannot be silently sent to the image executor.
    """

    text = _completion_text(completion).strip()
    growth_match = _GROWTH_RE.fullmatch(text)
    if growth_match is not None:
        body = growth_match.group("body").strip()
    else:
        think_match = _THINK_RE.fullmatch(text)
        if think_match is None:
            raise ValueError("planner completion must contain exactly one <think>...</think> block")
        body = think_match.group("body").strip()
    if not body:
        raise ValueError("planner completion body is empty")
    if "<think>" in body or "</think>" in body or "<GROWTH_COT>" in body or "</GROWTH_COT>" in body:
        raise ValueError("planner completion contains nested reasoning or executor wrappers")
    return body


def to_growth_cot(completion: str | Mapping[str, Any]) -> str:
    """Wrap a planner completion for the Qwen Image executor."""

    return f"<GROWTH_COT>\n{growth_cot_body(completion)}\n</GROWTH_COT>"
