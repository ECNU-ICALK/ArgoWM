"""Maize token spans for the existing stage/metric/text normalized loss."""
from __future__ import annotations

import re
from typing import Any

from .experiment import STABLE, validate_growth_cot

VERSION = "wgsr_maize_planner_grouped_loss_v1"
METRIC_GROUPS = {"Culm": 4, "Visible foreground area": 5, "Projected fill": 6,
                 "Visible yellow-brown composition": 8}


def token_groups(ids: tuple[int, ...], tokenizer: Any) -> list[int]:
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if text.count("<GROWTH_COT>") != 1 or text.count("</GROWTH_COT>") != 1:
        raise ValueError("Expected one complete supervised maize answer")
    begin = text.index("<GROWTH_COT>")
    close = text.index("</GROWTH_COT>") + len("</GROWTH_COT>")
    if text[:begin].strip() or text[close:].strip() not in {"", "<|im_end|>"}:
        raise ValueError("Unexpected supervised prefix/suffix; prompt and thinking must be masked")
    # ms-swift can retain a newline beside EOS. Keep its original offsets and
    # assign it to text, exactly like the existing rice grouped-loss parser.
    validate_growth_cot(text[begin:close])
    spans = []
    for label, group in (("Current stage", 2), ("Future stage", 3)):
        match = re.search(rf"(?m)^{label}: ([^\r\n]+)", text)
        if match is None:
            raise ValueError("Missing maize stage span")
        spans.append((match.start(1), match.end(1), group))
    start, end = text.index("Growth plan"), text.index("Target phenotype")
    pattern = r"(?m)^([^:\r\n]+):[^\r\n]*?(Source=\d[^\r\n]+|" + re.escape(STABLE) + r")$"
    for match in re.finditer(pattern, text):
        if not start <= match.start() < end or match[1] not in METRIC_GROUPS:
            raise ValueError("Unexpected maize numeric span")
        spans.append((match.start(2), match.end(2), METRIC_GROUPS[match[1]]))
    if len(spans) != 6:
        raise ValueError("Expected two stage and four metric spans")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != list(ids):
        raise ValueError("Supervised token round trip differs")
    groups = []
    for left, right in encoded["offset_mapping"]:
        overlaps = [group for start, end, group in spans if left < end and right > start]
        if len(overlaps) > 1 or right <= left:
            raise ValueError("Ambiguous maize token boundary")
        groups.append(overlaps[0] if overlaps else 1)
    if not {1, 2, 3, 4, 5, 6, 8}.issubset(groups):
        raise ValueError("Missing maize supervised token group")
    return groups
