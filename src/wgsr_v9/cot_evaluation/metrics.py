"""Metrics with explicit reference and prediction coverage denominators."""
from __future__ import annotations

from collections import Counter
import math
import statistics
from typing import Any, Sequence


def ratio(numerator: float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def classification(truth: Sequence[Any], prediction: Sequence[Any], labels: Sequence[Any],
                   *, weights: Sequence[float] | None = None) -> dict[str, Any]:
    if len(truth) != len(prediction):
        raise ValueError("classification arrays are not aligned")
    weight = list(weights) if weights is not None else [1.0] * len(truth)
    if len(weight) != len(truth):
        raise ValueError("classification weights are not aligned")
    counts: Counter[tuple[Any, Any]] = Counter()
    for y, p, w in zip(truth, prediction, weight):
        if y not in labels:
            raise ValueError(f"unknown reference label: {y}")
        counts[(y, p if p in labels else None)] += w
    total = sum(weight)
    valid = sum(w for p, w in zip(prediction, weight) if p in labels)
    correct = sum(counts[(label, label)] for label in labels)
    per_class = []
    for label in labels:
        tp = counts[(label, label)]
        support = sum(v for (y, _), v in counts.items() if y == label)
        predicted = sum(v for (_, p), v in counts.items() if p == label)
        per_class.append({"label": label, "support": support, "predicted": predicted,
                          "precision": ratio(tp, predicted), "recall": ratio(tp, support),
                          "f1": 2 * tp / (support + predicted) if support + predicted else 0.0})
    return {"n_reference": len(truth), "weight_sum": total, "n_parsed": sum(p in labels for p in prediction),
            "coverage": ratio(valid, total), "accuracy": ratio(correct, total),
            "macro_f1": statistics.fmean(r["f1"] for r in per_class) if total else None,
            "balanced_accuracy": statistics.fmean(r["recall"] for r in per_class if r["support"]) if total else None,
            "per_class": per_class,
            "confusion": [{"reference": y, "prediction": p if p is not None else "invalid", "count": count}
                          for (y, p), count in counts.items()]}


def direction(value: float, deadband: float) -> str:
    return "decrease" if value < -deadband else ("increase" if value > deadband else "stable")


def numeric(truth: Sequence[float | None], prediction: Sequence[float | None], *,
            tolerances: Sequence[float] = (1, 2, 5)) -> dict[str, Any]:
    if len(truth) != len(prediction):
        raise ValueError("numeric arrays are not aligned")
    eligible = [(float(y), float(p) if p is not None and math.isfinite(p) else None)
                for y, p in zip(truth, prediction) if y is not None and math.isfinite(y)]
    error = [p - y for y, p in eligible if p is not None]
    absolute = [abs(e) for e in error]
    return {"n_reference": len(eligible), "n_parsed": len(error), "coverage": ratio(len(error), len(eligible)),
            "mae": statistics.fmean(absolute) if error else None,
            "median_absolute_error": statistics.median(absolute) if error else None,
            "rmse": math.sqrt(statistics.fmean(e * e for e in error)) if error else None,
            "signed_bias": statistics.fmean(error) if error else None,
            "within_tolerance": {str(t): ratio(sum(e <= t + 1e-10 for e in absolute), len(eligible)) for t in tolerances}}


def direction_metrics(truth: Sequence[float | None], prediction: Sequence[float | None],
                      *, deadband: float) -> dict[str, Any]:
    pairs = [(y, p) for y, p in zip(truth, prediction) if y is not None]
    ys = [direction(y, deadband) for y, _ in pairs]
    ps = [direction(p, deadband) if p is not None else None for _, p in pairs]
    result = classification(ys, ps, ["decrease", "stable", "increase"])
    changed = [(y, p) for y, p in zip(ys, ps) if y != "stable"]
    result.update(deadband_pp=deadband, changed_reference_count=len(changed),
                  changed_only_accuracy=ratio(sum(y == p for y, p in changed), len(changed)))
    return result


def stage_metrics(truth: Sequence[str], prediction: Sequence[str | None], stages: list[str]) -> dict[str, Any]:
    result = classification(truth, prediction, stages)
    differences = [stages.index(p) - stages.index(y) for y, p in zip(truth, prediction) if p in stages]
    result.update(ordinal_mae=statistics.fmean(abs(d) for d in differences) if differences else None,
                  signed_stage_bias=statistics.fmean(differences) if differences else None,
                  within_one_stage_accuracy=ratio(sum(abs(d) <= 1 for d in differences), len(truth)))
    return result
