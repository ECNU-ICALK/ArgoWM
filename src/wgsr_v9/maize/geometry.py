"""Deterministic maize input geometry normalization."""

from __future__ import annotations

from collections import Counter
from typing import Any

from PIL import Image

from .contract import CANONICAL_SOURCE_SIZE, CANONICAL_TARGET_SIZE


TOP_PADDING = CANONICAL_TARGET_SIZE[1] - CANONICAL_SOURCE_SIZE[1]
FILL_SAMPLE_ROWS = 16
FILL_SAMPLE_SIDE_MARGIN = 0.125


def _background_fill(image: Image.Image) -> tuple[int, int, int]:
    """Use the modal color in the central top background strip for padding."""

    width, height = image.size
    left = int(round(width * FILL_SAMPLE_SIDE_MARGIN))
    right = int(round(width * (1.0 - FILL_SAMPLE_SIDE_MARGIN)))
    sample = image.crop((left, 0, right, min(FILL_SAMPLE_ROWS, height))).convert("RGB")
    common = Counter(sample.getdata()).most_common(1)
    if not common:
        raise ValueError("Cannot estimate geometry padding fill color")
    return tuple(int(value) for value in common[0][0])


def normalize_geometry(image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
    """Pad the top of a maize image while preserving the bottom anchor.

    The input is not resized.  Ten pixels are added above the original canvas,
    which preserves plant scale and the bottom pot/base position while making
    the height divisible by 32 for later image-model stages.
    """

    source = image.convert("RGB")
    if source.size != CANONICAL_SOURCE_SIZE:
        raise ValueError(f"Expected maize source size {CANONICAL_SOURCE_SIZE}, got {source.size}")
    fill_rgb = _background_fill(source)
    normalized = Image.new("RGB", CANONICAL_TARGET_SIZE, fill_rgb)
    normalized.paste(source, (0, TOP_PADDING))
    return normalized, {
        "source_size": list(CANONICAL_SOURCE_SIZE),
        "target_size": list(CANONICAL_TARGET_SIZE),
        "resize_applied": False,
        "padding": {"left": 0, "top": TOP_PADDING, "right": 0, "bottom": 0},
        "padding_fill_rgb": list(fill_rgb),
        "padding_fill_strategy": "central_top_strip_mode",
        "fill_sample_rows": FILL_SAMPLE_ROWS,
        "fill_sample_side_margin": FILL_SAMPLE_SIDE_MARGIN,
    }
