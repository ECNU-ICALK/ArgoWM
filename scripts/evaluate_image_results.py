#!/usr/bin/env python3
"""Evaluate existing WGSR V9 image predictions against a dataset manifest."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wgsr_v9.v9_image_metrics import (  # noqa: E402
    MIN_COLOR_COVERAGE,
    classify_colors,
    foreground_mask,
    normalize_view,
)

from wgsr_v9.evaluation_io import (  # noqa: E402
    ALGORITHM_VERSION,
    PROTOCOL_VERSION,
    add_common_arguments,
    ensure_new_output_dir,
    load_dataset_manifest,
    load_predictions,
    pair_key_for_dataset_row,
    parse_result_specs,
    write_json_atomic,
    write_jsonl_atomic,
)
from wgsr_v9.executor_training import prediction_interval_weeks  # noqa: E402


METRICS = (
    "mae_255",
    "mae_01",
    "mse_255",
    "psnr_db",
    "ssim",
    "foreground_iou",
    "foreground_precision",
    "foreground_recall",
    "foreground_f1",
    "target_foreground_area_pct",
    "prediction_foreground_area_pct",
    "foreground_area_abs_error_pp",
    "target_visible_extent_y_pct",
    "prediction_visible_extent_y_pct",
    "visible_extent_y_abs_error_pp",
    "target_yellow_brown_fraction",
    "prediction_yellow_brown_fraction",
    "yellow_brown_fraction_abs_error_pp",
)


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if "A" in image.getbands():
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba)
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _ssim(reference: np.ndarray, generated: np.ndarray) -> tuple[float | None, str | None]:
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        return None, "scipy is not installed; SSIM unavailable"
    x = reference.astype(np.float32)
    y = generated.astype(np.float32)
    sigma = (1.5, 1.5, 0.0)
    mu_x = gaussian_filter(x, sigma=sigma, mode="reflect", truncate=3.5)
    mu_y = gaussian_filter(y, sigma=sigma, mode="reflect", truncate=3.5)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x_sq = gaussian_filter(x * x, sigma=sigma, mode="reflect", truncate=3.5) - mu_x_sq
    sigma_y_sq = gaussian_filter(y * y, sigma=sigma, mode="reflect", truncate=3.5) - mu_y_sq
    sigma_xy = gaussian_filter(x * y, sigma=sigma, mode="reflect", truncate=3.5) - mu_xy
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    score = numerator / np.maximum(denominator, np.finfo(np.float32).eps)
    return float(np.mean(score, dtype=np.float64)), None


def _extent_y_pct(mask: np.ndarray) -> float | None:
    positions = np.where(mask.any(axis=1))[0]
    if positions.size == 0:
        return None
    return 100.0 * float(positions[-1] - positions[0] + 1) / float(mask.shape[0])


def _area_pct(mask: np.ndarray) -> float:
    return 100.0 * float(mask.sum()) / float(mask.size)


def _color_evidence(rgb: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    return classify_colors(rgb, mask, min_color_coverage=MIN_COLOR_COVERAGE)


def _evaluate_one(
    dataset_row: dict[str, Any],
    prediction_path: Path,
    *,
    allow_resize: bool,
) -> dict[str, Any]:
    source_path = Path(str(dataset_row["source_image_path"]))
    target_path = Path(str(dataset_row["target_image_path"]))
    if not source_path.is_file():
        raise FileNotFoundError(f"canonical source image is missing: {source_path}")
    if not target_path.is_file():
        raise FileNotFoundError(f"canonical target image is missing: {target_path}")
    source = _load_rgb(source_path)
    target = _load_rgb(target_path)
    generated = _load_rgb(prediction_path)
    prediction_resized = generated.shape != target.shape
    if generated.shape != target.shape:
        if not allow_resize:
            raise ValueError(
                f"prediction dimensions {generated.shape[1]}x{generated.shape[0]} do not match "
                f"target {target.shape[1]}x{target.shape[0]} (pass --resize-predictions to resize)"
            )
        with Image.fromarray(generated, mode="RGB") as image:
            generated = np.asarray(image.resize((target.shape[1], target.shape[0]), Image.Resampling.LANCZOS))

    difference = generated.astype(np.float32) - target.astype(np.float32)
    mae = float(np.mean(np.abs(difference), dtype=np.float64))
    mse = float(np.mean(difference * difference, dtype=np.float64))
    psnr = None if mse == 0.0 else float(20.0 * math.log10(255.0 / math.sqrt(mse)))
    ssim, ssim_reason = _ssim(target, generated)

    target_mask = foreground_mask(target)
    prediction_mask = foreground_mask(generated)
    intersection = int(np.logical_and(target_mask, prediction_mask).sum())
    target_pixels = int(target_mask.sum())
    prediction_pixels = int(prediction_mask.sum())
    union = int(np.logical_or(target_mask, prediction_mask).sum())
    iou = None if union == 0 else intersection / float(union)
    precision = None if prediction_pixels == 0 else intersection / float(prediction_pixels)
    recall = None if target_pixels == 0 else intersection / float(target_pixels)
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)

    target_color = _color_evidence(target, target_mask)
    prediction_color = _color_evidence(generated, prediction_mask)
    target_yb = target_color.get("yellow_brown_fraction")
    prediction_yb = prediction_color.get("yellow_brown_fraction")
    yb_error = None
    if target_yb is not None and prediction_yb is not None:
        yb_error = 100.0 * abs(float(target_yb) - float(prediction_yb))

    target_extent = _extent_y_pct(target_mask)
    prediction_extent = _extent_y_pct(prediction_mask)
    extent_error = None
    if target_extent is not None and prediction_extent is not None:
        extent_error = abs(target_extent - prediction_extent)
    target_area = _area_pct(target_mask)
    prediction_area = _area_pct(prediction_mask)

    return {
        "status": "completed",
        "error": None,
        "prediction_resized": prediction_resized,
        "image_width": int(target.shape[1]),
        "image_height": int(target.shape[0]),
        "mae_255": mae,
        "mae_01": mae / 255.0,
        "mse_255": mse,
        "psnr_db": psnr,
        "ssim": ssim,
        "ssim_unavailable_reason": ssim_reason,
        "foreground_iou": iou,
        "foreground_precision": precision,
        "foreground_recall": recall,
        "foreground_f1": f1,
        "target_foreground_area_pct": target_area,
        "prediction_foreground_area_pct": prediction_area,
        "foreground_area_abs_error_pp": abs(target_area - prediction_area),
        "target_visible_extent_y_pct": target_extent,
        "prediction_visible_extent_y_pct": prediction_extent,
        "visible_extent_y_abs_error_pp": extent_error,
        "target_yellow_brown_fraction": target_yb,
        "prediction_yellow_brown_fraction": prediction_yb,
        "yellow_brown_fraction_abs_error_pp": yb_error,
        "target_color_coverage": target_color.get("color_coverage"),
        "prediction_color_coverage": prediction_color.get("color_coverage"),
        "target_color_status": target_color.get("status"),
        "prediction_color_status": prediction_color.get("status"),
        "source_shape": [int(source.shape[1]), int(source.shape[0])],
    }


def _mean_median(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(statistics.fmean(values)), float(statistics.median(values))


def _evaluate_task(task: tuple[dict[str, Any], Path, bool]) -> dict[str, Any]:
    dataset_row, prediction_path, allow_resize = task
    try:
        return {**_evaluate_one(dataset_row, prediction_path, allow_resize=allow_resize),
                "metric_status": "available"}
    except (OSError, ValueError, RuntimeError) as exc:
        return {"status": "invalid", "metric_status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}


def horizon_fields(row: dict[str, Any], rule: str) -> dict[str, Any]:
    if rule == "manifest":
        return {"horizon_bucket": row.get("horizon_bucket")}
    # Ignore the historical manifest's ceil-week field; preserve it as provenance.
    weeks = prediction_interval_weeks({"actual_interval_days": row.get("actual_interval_days")})
    if weeks > 20:
        raise ValueError("rounded horizon falls outside reviewed 1–20 week bins")
    return {"horizon_bucket": "1-4" if weeks <= 4 else "5-8" if weeks <= 8 else "9-20",
            "prediction_interval_weeks": weeks, "horizon_rule": "rounded_weeks",
            "original_horizon_bucket": row.get("horizon_bucket")}


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, Any]], group_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(field) or "") for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        valid = [row for row in group if row.get("status") == "completed"]
        record: dict[str, Any] = {field: value for field, value in zip(group_fields, key)}
        record["dataset_rows"] = len(group)
        record["completed"] = len(valid)
        record["failed_or_missing"] = len(group) - len(valid)
        for metric in METRICS:
            values = [float(row[metric]) for row in valid if row.get(metric) is not None]
            mean, median = _mean_median(values)
            record[f"{metric}_mean"] = mean
            record[f"{metric}_median"] = median
            record[f"{metric}_n"] = len(values)
        output.append(record)
    return output


def _write_aggregate_csv(path: Path, rows: list[dict[str, Any]], group_fields: tuple[str, ...]) -> None:
    columns = [*group_fields, "dataset_rows", "completed", "failed_or_missing"]
    for metric in METRICS:
        columns.extend((f"{metric}_mean", f"{metric}_median", f"{metric}_n"))
    _write_csv(path, rows, columns)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1,
                        help="CPU image-metric processes; ordered results and metric formulas are unchanged.")
    parser.add_argument("--horizon-rule", choices=("manifest", "rounded_weeks"), default="manifest")
    parser.add_argument("--require-complete", action="store_true",
                        help="Exit nonzero for missing/invalid predictions or metadata mismatches.")
    parser.add_argument(
        "--resize-predictions",
        action="store_true",
        help="Resize prediction images to target dimensions with Lanczos before pixel metrics.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    output_dir = ensure_new_output_dir(args.output_dir)

    specs = parse_result_specs(args.result)
    dataset_path, dataset_rows, dataset_sha256 = load_dataset_manifest(args.dataset_manifest, args.path_root)
    duplicate_ids: set[str] = set()
    seen_ids: set[str] = set()
    for row in dataset_rows:
        key = pair_key_for_dataset_row(row)
        if key is None:
            continue
        stable_id = "|".join(key)
        if stable_id in seen_ids:
            duplicate_ids.add(stable_id)
        seen_ids.add(stable_id)
    if duplicate_ids:
        raise ValueError(f"dataset manifest has duplicate alignment keys: {sorted(duplicate_ids)[:10]}")

    predictions_by_model: dict[str, dict[tuple[str, str], Any]] = {}
    alignment_models: dict[str, Any] = {}
    for model_name, result_path in specs.items():
        predictions_by_model[model_name], alignment_models[model_name] = load_predictions(
            result_path,
            extra_roots=args.path_root,
            dataset_rows=dataset_rows,
        )

    per_pair: list[dict[str, Any]] = []
    tasks: list[tuple[dict[str, Any], Path, bool]] = []
    task_records: list[int] = []
    for dataset_row in dataset_rows:
        key = pair_key_for_dataset_row(dataset_row)
        for model_name in specs:
            prediction = predictions_by_model[model_name].get(key) if key is not None else None
            record = {
                "algorithm_version": ALGORITHM_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "model_name": model_name,
                "pair_id": dataset_row.get("pair_id"),
                "pair_group_id": dataset_row.get("pair_group_id"),
                "row_index": dataset_row.get("row_index"),
                "split": dataset_row.get("split"),
                "view": dataset_row.get("view"),
                "plant_id": dataset_row.get("plant_id"),
                "source_capture_date": dataset_row.get("source_capture_date"),
                "target_capture_date": dataset_row.get("target_capture_date"),
                "source_stage": dataset_row.get("source_stage"),
                "target_stage": dataset_row.get("target_stage"),
                **horizon_fields(dataset_row, args.horizon_rule),
                "actual_interval_days": dataset_row.get("actual_interval_days"),
                "dataset_manifest": str(dataset_path),
                "dataset_manifest_sha256": dataset_sha256,
                "source_path": dataset_row["source_image_path"],
                "target_path": dataset_row["target_image_path"],
                "prediction_path": str(prediction.output_path) if prediction else None,
                "result_ledger_path": str(prediction.result_file) if prediction else None,
                "result_protocol_version": prediction.row.get("protocol_version") if prediction else None,
                "status": "missing_prediction" if prediction is None else "pending",
                "error": None if prediction else "no successful result record with an existing output file matched this dataset row",
                "retry_count": int(prediction.row.get("retry_count") or 0) if prediction else 0,
                "metric_status": "pending" if prediction else "unavailable",
            }
            if prediction is not None:
                tasks.append((dataset_row, prediction.output_path, args.resize_predictions))
                task_records.append(len(per_pair))
            per_pair.append(record)

    def collect(results: Any) -> None:
        for number, (index, metrics) in enumerate(zip(task_records, results), 1):
            per_pair[index].update(metrics)
            if number % 100 == 0 or number == len(tasks):
                print(f"Evaluated {number}/{len(tasks)} predictions", flush=True)

    if args.workers == 1:
        collect(map(_evaluate_task, tasks))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            collect(pool.map(_evaluate_task, tasks, chunksize=4))

    write_jsonl_atomic(output_dir / "per_pair_results.jsonl", per_pair)
    alignment_report = {
        "algorithm_version": ALGORITHM_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "dataset_manifest": str(dataset_path),
        "dataset_manifest_sha256": dataset_sha256,
        "dataset_row_count": len(dataset_rows),
        "alignment_key_policy": "pair_id primary; row_index fallback only when pair_id is absent",
        "canonical_source_target_policy": "source_image_path and target_image_path are taken only from the dataset manifest; result ledger image paths are ignored",
        "models": alignment_models,
    }
    write_json_atomic(output_dir / "alignment_report.json", alignment_report)

    model_summary: list[dict[str, Any]] = []
    for model_name in specs:
        model_rows = [row for row in per_pair if row["model_name"] == model_name]
        model_summary.extend(_aggregate(model_rows, ("model_name",)))
    _write_aggregate_csv(output_dir / "per_model_summary.csv", model_summary, ("model_name",))
    for suffix, fields in (
        ("view", ("model_name", "view")),
        ("split", ("model_name", "split")),
        ("horizon", ("model_name", "horizon_bucket")),
        ("stage", ("model_name", "source_stage", "target_stage")),
    ):
        aggregates: list[dict[str, Any]] = []
        for model_name in specs:
            selected = [row for row in per_pair if row["model_name"] == model_name]
            aggregates.extend(_aggregate(selected, fields))
        _write_aggregate_csv(output_dir / f"per_model_by_{suffix}.csv", aggregates, fields)

    overall: dict[str, Any] = {}
    for model_name in specs:
        model_rows = [row for row in per_pair if row["model_name"] == model_name]
        completed = [row for row in model_rows if row.get("status") == "completed"]
        overall[model_name] = {
            "dataset_rows": len(model_rows),
            "completed": len(completed),
            "missing_prediction": sum(row.get("status") == "missing_prediction" for row in model_rows),
            "invalid": sum(row.get("status") == "invalid" for row in model_rows),
            "metrics": {},
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in completed if row.get(metric) is not None]
            mean, median = _mean_median(values)
            overall[model_name]["metrics"][metric] = {"mean": mean, "median": median, "n": len(values)}
    summary = {
        "algorithm_version": ALGORITHM_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "dataset_manifest": str(dataset_path),
        "dataset_manifest_sha256": dataset_sha256,
        "dataset_rows": len(dataset_rows),
        "model_count": len(specs),
        "record_count": len(per_pair),
        "resize_predictions": bool(args.resize_predictions),
        "horizon_rule": args.horizon_rule,
        "workers": args.workers,
        "foreground_definition": "V9 pure-white foreground_mask normalization",
        "color_algorithm_version": "wgsr_v9_color_hsv_baseline_v1",
        "color_coverage_threshold": MIN_COLOR_COVERAGE,
        "ssim_definition": "mean channel-wise Gaussian-window SSIM (sigma=1.5 px, truncate=3.5; scipy backend)",
        "ssim_unavailable_reason": next(
            (row.get("ssim_unavailable_reason") for row in per_pair if row.get("ssim_unavailable_reason")), None
        ),
        "models": overall,
        "outputs": {
            "per_pair_results": str(output_dir / "per_pair_results.jsonl"),
            "alignment_report": str(output_dir / "alignment_report.json"),
            "per_model_summary": str(output_dir / "per_model_summary.csv"),
        },
    }
    write_json_atomic(output_dir / "summary.json", summary)
    print(f"Evaluation complete: {output_dir}")
    for name, data in overall.items():
        print(
            f"{name}: completed={data['completed']}/{data['dataset_rows']} "
            f"missing={data['missing_prediction']} invalid={data['invalid']}"
        )
    incomplete = any(data["completed"] != data["dataset_rows"] for data in overall.values())
    mismatched = any(data["metadata_mismatch_row_count"] for data in alignment_models.values())
    return int(args.require_complete and (incomplete or mismatched))


if __name__ == "__main__":
    raise SystemExit(main())
