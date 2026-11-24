"""Per-example, per-field normalized planner token cross entropy."""
from __future__ import annotations

from functools import lru_cache
import math
import re
from typing import Any

import torch

from .planner_direct import validate_direct_body

VERSION = "wgsr_planner_grouped_loss_v1"
# 0 = ignored; 1 = other answer tokens; 2/3 = current/future stage.
METRIC_GROUPS = {"Culm": 4, "Leaves/area": 5, "Plant density": 6,
                 "Panicle": 7, "Plant color development": 8, "Canopy": 9}


def validate_weights(weights: tuple[float, float, float]) -> None:
    if len(weights) != 3 or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("stage/metric/text weights must be finite and positive")
    if not math.isclose(sum(weights), 1., abs_tol=1e-8):
        raise ValueError("stage/metric/text weights must sum to one")


def token_groups(ids: tuple[int, ...], tokenizer: Any) -> list[int]:
    """Classify the existing supervised tokens without changing tokenization."""
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if text.count("<GROWTH_COT>") != 1 or text.count("</GROWTH_COT>") != 1:
        raise ValueError("expected one complete direct planner answer")
    start = text.index("<GROWTH_COT>") + len("<GROWTH_COT>")
    end = text.index("</GROWTH_COT>")
    if text[:start-len("<GROWTH_COT>")].strip() or text[end+len("</GROWTH_COT>"):].strip() not in {"", "<|im_end|>"}:
        raise ValueError("unexpected supervised prefix/suffix; prompt/thinking must be masked")
    body = text[start:end]
    view = "front" if re.search(r"(?m)^Culm:", body) else "top"
    validate_direct_body(body.strip(), view)
    spans = []
    for label, group in (("Current stage", 2), ("Future stage", 3)):
        matches = list(re.finditer(rf"(?m)^{label}:[ \t]*([^\r\n]+)", text))
        if len(matches) != 1:
            raise ValueError("missing or duplicate stage span")
        m = matches[0]
        spans.append((m.start(1), m.end(1), group))
    growth_start = text.index("Growth plan", start)
    growth_end = text.index("Target phenotype", growth_start)
    for m in re.finditer(r"(?m)^([^:\r\n]+):[^\r\n]*?(\bMetric = [^\r\n]+)", text):
        if not growth_start <= m.start() < growth_end:
            raise ValueError("metric outside Growth plan")
        spans.append((m.start(2), m.end(2), METRIC_GROUPS[m[1]]))
    if len(spans) != (7 if view == "front" else 6):
        raise ValueError("incomplete stage/metric spans")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != list(ids):
        raise ValueError("supervised token round trip differs; refuse approximate alignment")
    groups = []
    for a, b in encoded["offset_mapping"]:
        overlaps = [g for s, e, g in spans if a < e and b > s]
        if len(overlaps) > 1 or b <= a:
            raise ValueError("ambiguous token boundary")
        groups.append(overlaps[0] if overlaps else 1)
    if not {1, 2, 3, *(s[2] for s in spans)}.issubset(groups):
        raise ValueError("empty token group")
    return groups


def supervised_groups(labels: torch.Tensor, tokenizer: Any, classify=None) -> torch.Tensor:
    """Return unshifted IDs; all prompt/image/empty-think/padding labels stay zero."""
    if labels.ndim != 2 or torch.any(labels[:, 0] != -100):
        raise ValueError("first causal position must be ignored in each padded example")
    classify = classify or (lambda ids: token_groups(ids, tokenizer))
    output = []
    for row in labels.detach().cpu().tolist():
        positions = [i for i, v in enumerate(row) if v != -100]
        if not positions or positions != list(range(positions[0], positions[-1]+1)):
            raise ValueError("expected one contiguous supervised answer per sample")
        groups = classify(tuple(row[i] for i in positions))
        if len(groups) != len(positions):
            raise ValueError("token group alignment length mismatch")
        result = [0] * len(row)
        for i, group in zip(positions, groups, strict=True):
            result[i] = group
        output.append(result)
    return torch.tensor(output, dtype=torch.long, device=labels.device)


def grouped_loss(token_losses: torch.Tensor, shifted_groups: torch.Tensor,
                 weights: tuple[float, float, float]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Average tokens within fields, fields within groups, then examples."""
    validate_weights(weights)
    if token_losses.ndim != 2 or token_losses.shape != shifted_groups.shape:
        raise ValueError("loss/group shapes differ")
    parts: dict[str, list[torch.Tensor]] = {"stage": [], "metric": [], "text": [],
                                          "stage_current": [], "stage_future": []}
    for losses, groups in zip(token_losses, shifted_groups, strict=True):
        if torch.any((groups < 0) | (groups > 9)):
            raise ValueError("unknown token group")
        means = {g: losses[groups == g].mean() for g in range(1, 10) if torch.any(groups == g)}
        if not {1, 2, 3}.issubset(means) or not any(g >= 4 for g in means):
            raise ValueError("missing stage, metric or text loss group")
        parts["stage_current"].append(means[2])
        parts["stage_future"].append(means[3])
        parts["stage"].append((means[2]+means[3])/2)
        parts["metric"].append(torch.stack([v for g,v in means.items() if g >= 4]).mean())
        parts["text"].append(means[1])
    averaged = {k: torch.stack(v).mean() for k,v in parts.items()}
    loss = sum(w * averaged[k] for w,k in zip(weights, ("stage", "metric", "text"), strict=True))
    return loss, averaged
