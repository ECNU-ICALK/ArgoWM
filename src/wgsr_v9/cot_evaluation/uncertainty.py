"""Plant-cluster bootstrap, including recomputed non-linear F1 statistics."""
from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import direction


def bootstrap(rows: list[dict[str, Any]], stages: list[str], *, replicates: int, seed: int) -> list[dict[str, Any]]:
    plants = sorted({r["plant_id"] for r in rows})
    plant_index = {plant: i for i, plant in enumerate(plants)}
    row_plant = np.array([plant_index[r["plant_id"]] for r in rows])
    draws = np.random.default_rng(seed).integers(0, len(plants), size=(replicates, len(plants)))
    weights = np.zeros((replicates, len(plants)), dtype=np.int16)
    for i in range(len(plants)):
        weights += (draws == i).sum(axis=1)[:, None] * np.eye(len(plants), dtype=np.int16)[i]
    weights = weights.astype(np.float64)
    reports = []

    def save(name: str, estimate: float | None, values: np.ndarray, n: int, details: dict[str, Any] | None = None) -> None:
        valid = values[np.isfinite(values)]
        reports.append({"metric": name, "estimate": estimate, "n": n, "plant_clusters": len(plants),
                        "ci95_lower": float(np.quantile(valid, .025)) if len(valid) else None,
                        "ci95_upper": float(np.quantile(valid, .975)) if len(valid) else None,
                        "valid_replicates": len(valid), "replicates": replicates, "seed": seed,
                        "unit": "plant_id", **(details or {})})

    def mean_ci(name: str, values: list[float | None]) -> None:
        array = np.array([np.nan if v is None else v for v in values])
        valid = np.isfinite(array)
        sums = np.bincount(row_plant[valid], weights=array[valid], minlength=len(plants))
        counts = np.bincount(row_plant[valid], minlength=len(plants))
        numerator, denominator = weights @ sums, weights @ counts
        estimates = np.divide(numerator, denominator, out=np.full(replicates, np.nan), where=denominator != 0)
        save(name, float(array[valid].mean()) if valid.any() else None, estimates, int(valid.sum()))

    def class_ci(name: str, truth: list[Any], prediction: list[Any], labels: list[Any], *, positive_only: bool = False) -> None:
        size = len(labels)
        counts = np.zeros((len(plants), size, size + 1), dtype=np.float64)
        eligible = 0
        for i, (y, p) in enumerate(zip(truth, prediction)):
            if y is None:
                continue
            counts[row_plant[i], labels.index(y), labels.index(p) if p in labels else size] += 1
            eligible += 1
        matrix = (weights @ counts.reshape(len(plants), -1)).reshape(replicates, size, size+1)
        all_matrix = counts.sum(axis=0)[None]

        def calculate(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            tp = np.diagonal(mat[:, :, :size], axis1=1, axis2=2)
            support, predicted = mat.sum(axis=2), mat[:, :, :size].sum(axis=1)
            f1 = np.divide(2*tp, support+predicted, out=np.zeros_like(tp), where=(support+predicted)>0)
            denominator = mat.sum(axis=(1, 2))
            accuracy = np.divide(tp.sum(axis=1), denominator, out=np.full(len(mat), np.nan), where=denominator>0)
            score = f1[:, -1] if positive_only else f1.mean(axis=1)
            score[denominator == 0] = np.nan
            return accuracy, score
        acc, f1 = calculate(matrix)
        original_acc, original_f1 = calculate(all_matrix)
        save(f"{name}.accuracy", float(original_acc[0]) if eligible else None, acc, eligible)
        save(f"{name}.{'positive_f1' if positive_only else 'macro_f1'}", float(original_f1[0]) if eligible else None, f1, eligible)

    mean_ci("structure.strict_valid_rate", [float(r["strict_structure_valid"]) for r in rows])
    for role, reference in (("current", "source_stage"), ("future", "target_stage")):
        class_ci(f"stage.{role}", [r[reference] for r in rows], [r["parsed"]["stages"][role] for r in rows], stages)
    for key in sorted({key for r in rows for key in r["numeric"] if key != "panicle"}):
        for role in ("source", "target", "declared_change", "derived_change"):
            errors, within2 = [], []
            ys, ps = [], []
            for row in rows:
                f = row["numeric"].get(key)
                y = f["reference"][role] if f else None
                p = f["prediction"][role] if f else None
                e = abs(p-y) if p is not None and y is not None else None
                errors.append(e)
                within2.append(float(e is not None and e <= 2+1e-10) if y is not None else None)
                ys.append(direction(y, 1) if y is not None else None)
                ps.append(direction(p, 1) if p is not None else None)
            mean_ci(f"{key}.{role}.mae", errors)
            mean_ci(f"{key}.{role}.accuracy_within_2pp", within2)
            if role in {"declared_change", "derived_change"}:
                class_ci(f"{key}.{role}.direction", ys, ps, ["decrease", "stable", "increase"])
    for role in ("source", "target", "emergence"):
        ys, ps = [], []
        for row in rows:
            field = row["numeric"]["panicle"]
            endpoint = "derived_change" if role == "emergence" else role
            y, p = field["reference"][endpoint], field["prediction"][endpoint]
            ys.append(None if y is None else int(y == 1))
            ps.append(None if p is None else int(p == 1))
        class_ci(f"panicle.{role}", ys, ps, [0, 1], positive_only=True)
    return reports
