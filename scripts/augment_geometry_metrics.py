#!/usr/bin/env python3
"""Add CoT-aligned Height/Density to existing evaluations without recomputing DINO."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wgsr_v9.evaluation_io import (  # noqa: E402
    load_dataset_manifest, read_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic,
)
from wgsr_v9.image_geometry_evaluation import (  # noqa: E402
    ALGORITHM_VERSION, PROTOCOL_VERSION, METRICS, compare_geometry, measure_geometry_image,
)
from wgsr_v9.v9_image_metrics import normalize_view  # noqa: E402


def _measure_task(task: dict[str, Any]) -> dict[str, Any]:
    result = dict(task)
    path = Path(task["input_path"])
    try:
        result.update(measure_geometry_image(path, task["view"]))
        result["image_sha256"] = sha256_file(path)
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (task["input_size"], task["input_mtime_ns"]):
            raise ValueError(f"image changed during measurement: {path}")
        result.update(status="completed", error=None)
    except (OSError, ValueError, RuntimeError) as exc:
        result.update(status="invalid", error=f"{type(exc).__name__}: {exc}")
    return result


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        summary[metric] = {
            "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "n": len(values),
        }
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    output = args.output_dir.resolve()
    inputs = [p.resolve() for p in args.evaluation_dir]
    if any(output == p or output in p.parents for p in inputs):
        raise ValueError("output must not contain or replace input evaluation directories")
    manifest, dataset, manifest_hash = load_dataset_manifest(args.dataset_manifest, [])
    canonical = {row["pair_id"]: row for row in dataset}
    if len(canonical) != len(dataset):
        raise ValueError("canonical pair IDs must be unique")

    original: list[dict[str, Any]] = []
    model_summaries: dict[str, Any] = {}
    model_inputs: dict[str, Path] = {}
    provenance = []
    for directory in inputs:
        ledger = directory / "per_pair_results.jsonl"
        summary_path = directory / "summary.json"
        summary = json.loads(summary_path.read_text())
        # Historical merged summaries omit this field; every row is still
        # required below to match the canonical manifest hash and image paths.
        if summary.get("dataset_manifest_sha256") not in (None, manifest_hash):
            raise ValueError(f"dataset differs: {directory}")
        for name, model_summary in summary["models"].items():
            if name in model_summaries:
                raise ValueError(f"duplicate model: {name}")
            model_summaries[name] = model_summary
            model_inputs[name] = ledger
        original.extend(read_jsonl(ledger))
        provenance.append({
            "input_path": str(ledger), "sha256": sha256_file(ledger),
            "summary_path": str(summary_path), "summary_sha256": sha256_file(summary_path),
            "summary_dataset_manifest_sha256": summary.get("dataset_manifest_sha256"),
        })

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    image_keys: set[tuple[str, str]] = set()
    for row in original:
        name, pair_id = row["model_name"], row["pair_id"]
        if name not in model_summaries or pair_id not in canonical:
            raise ValueError(f"unknown model/pair: {name}, {pair_id}")
        expected = canonical[pair_id]
        view = normalize_view(expected["view"])
        if row["dataset_manifest_sha256"] != manifest_hash or normalize_view(row["view"]) != view:
            raise ValueError(f"dataset/view mismatch: {name}, {pair_id}")
        if row["status"] != "completed" or row.get("prediction_resized"):
            raise ValueError("augmentation requires completed evaluations without resizing")
        for field, source in (("source_path", "source_image_path"), ("target_path", "target_image_path")):
            if Path(row[field]).resolve() != Path(expected[source]).resolve():
                raise ValueError(f"canonical image mismatch: {name}, {pair_id}, {field}")
        if any(field in row for field in METRICS):
            raise ValueError("input already contains Height/Density; select the original evaluation")
        for field in ("target_path", "prediction_path"):
            path = Path(row[field]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            image_keys.add((str(path), view))
        grouped[name].append(row)
    for name in model_summaries:
        rows = grouped[name]
        if len(rows) != len(dataset) or {r["pair_id"] for r in rows} != set(canonical):
            raise ValueError(f"incomplete or duplicate query coverage: {name}")

    code_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in (
        Path(__file__).resolve(), ROOT / "src/wgsr_v9/image_geometry_evaluation.py",
        ROOT / "src/wgsr_v9/v9_image_metrics.py",
    )}
    plan = {
        "algorithm_version": ALGORITHM_VERSION, "protocol_version": PROTOCOL_VERSION,
        "input_path": str(manifest), "dataset_manifest_sha256": manifest_hash,
        "output_path": str(output), "input_evaluations": provenance, "code_hashes": code_hashes,
    }
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "evaluation_plan.json"
    if plan_path.exists():
        previous = json.loads(plan_path.read_text())
        if any(previous.get(key) != value for key, value in plan.items()):
            raise ValueError("output belongs to different inputs/code; use a new directory")
    elif any(output.iterdir()):
        raise ValueError("nonempty output directory has no matching evaluation plan")
    write_json_atomic(plan_path, {**plan, "status": "running", "error": None, "retry_count": 0,
                                  "workers": args.workers, "image_count": len(image_keys)})

    cache_path = output / "image_geometry.jsonl"
    cache = {(row["input_path"], row["view"]): row for row in read_jsonl(cache_path)} if cache_path.exists() else {}
    pending = []
    for path_string, view in sorted(image_keys):
        stat = Path(path_string).stat()
        key = (path_string, view)
        old = cache.get(key)
        if old and old["status"] == "completed" and (old["input_size"], old["input_mtime_ns"]) == (stat.st_size, stat.st_mtime_ns):
            continue
        pending.append({
            "algorithm_version": ALGORITHM_VERSION, "protocol_version": PROTOCOL_VERSION,
            "input_path": path_string, "output_path": str(cache_path), "view": view,
            "input_size": stat.st_size, "input_mtime_ns": stat.st_mtime_ns,
            "status": "pending", "error": None, "retry_count": int(old["retry_count"]) + 1 if old else 0,
        })
    print(f"Aligned {len(original)} rows, {len(grouped)} models; {len(image_keys)} images, {len(pending)} pending", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(_measure_task, pending, chunksize=4), 1):
            cache[(result["input_path"], result["view"])] = result
            if index % 500 == 0 or index == len(pending):
                write_jsonl_atomic(cache_path, [cache[k] for k in sorted(cache)])
                print(f"Measured {index}/{len(pending)} new images", flush=True)
    failed = [cache[key] for key in image_keys if cache[key]["status"] != "completed"]
    if failed:
        raise RuntimeError(f"{len(failed)} images failed; inspect image_geometry.jsonl and resume")

    augmented = []
    for row in original:
        view = normalize_view(row["view"])
        target, prediction = (cache[(str(Path(row[field]).resolve()), view)] for field in ("target_path", "prediction_path"))
        for side, measured in (("target", target), ("prediction", prediction)):
            if (measured["image_width"], measured["image_height"]) != (row["image_width"], row["image_height"]):
                raise ValueError(f"dimensions differ from original evaluation: {row['pair_id']}")
            for old_suffix, new_field in (("foreground_area_pct", "foreground_area_pct"), ("visible_extent_y_pct", "visible_extent_y_pct")):
                old_value, new_value = row[f"{side}_{old_suffix}"], measured[new_field]
                if not (old_value == new_value or (old_value is not None and new_value is not None and math.isclose(old_value, new_value, abs_tol=1e-10, rel_tol=0))):
                    raise ValueError(f"foreground differs from historical evaluation: {row['model_name']}, {row['pair_id']}")
        record = {
            **row, **compare_geometry(target, prediction),
            "geometry_algorithm_version": ALGORITHM_VERSION, "geometry_protocol_version": PROTOCOL_VERSION,
            "geometry_input_path": str(model_inputs[row["model_name"]]),
            "geometry_output_path": str(output / "per_pair_results.jsonl"),
            "geometry_status": "completed", "geometry_error": None, "geometry_retry_count": 0,
            "target_geometry_measurement_status": target["measurement_status"],
            "prediction_geometry_measurement_status": prediction["measurement_status"],
        }
        if any(record[key] != value for key, value in row.items()):
            raise AssertionError("original evaluation field was changed")
        augmented.append(record)
    write_jsonl_atomic(output / "per_pair_results.jsonl", augmented)

    by_view = []
    comparison = []
    for name, summary in model_summaries.items():
        rows = [r for r in augmented if r["model_name"] == name]
        summary["metrics"].update(_summarize(rows))
        summary["height_metric_status"] = dict(Counter(r["height_metric_status"] for r in rows))
        summary["density_metric_status"] = dict(Counter(r["density_metric_status"] for r in rows))
        summary["prediction_geometry_measurement_status"] = dict(Counter(r["prediction_geometry_measurement_status"] for r in rows))
        comparison_row: dict[str, Any] = {"model_name": name}
        for field in ("prediction_dino_similarity", "foreground_iou", "foreground_area_abs_error_pp",
                      "front_height_abs_error_pp", "projection_density_abs_error_pp",
                      "yellow_brown_fraction_abs_error_pp", "dino_stage_recall_at_1",
                      "dino_event_recall_at_1", "psnr_db", "ssim"):
            metric = summary["metrics"].get(field, {"mean": None, "n": 0})
            comparison_row[field] = metric["mean"]
            comparison_row[field + "_n"] = metric["n"]
        comparison.append(comparison_row)
        for view in ("front", "top"):
            selected = [r for r in rows if normalize_view(r["view"]) == view]
            item: dict[str, Any] = {"model_name": name, "view": view, "query_count": len(selected)}
            for field, values in _summarize(selected).items():
                for statistic, value in values.items():
                    item[f"{field}_{statistic}"] = value
            by_view.append(item)
    _write_csv(output / "comparison.csv", comparison)
    _write_csv(output / "per_model_by_view.csv", by_view)
    definitions = {
        "height": "Front only: 100 * (H - top_foreground_row) / H; absolute error in pp, matching CoT culm.",
        "density": "Both views: foreground pixels / rasterized convex-hull pixel centers; absolute error * 100 in pp, matching CoT plant_density/canopy.",
        "empty_or_degenerate": "Empty foreground: height/density null. Degenerate convex hull: density null. No zero imputation.",
        "boundary_policy": "Retain valid visible measurements at boundaries and bottom gaps; record diagnostics; height is an image proxy, not physical height.",
        "preprocessing": "Pixel-evaluator white alpha compositing and existing V9 near-white foreground normalization; no resizing.",
        "aggregation": "Query-weighted mean over valid measurements; Height front only, Density both views, with per-view breakdown.",
        "other_metrics": "Preserved from original evaluation rows and summaries, including legacy extent and DINOv2 Base timing.",
    }
    write_json_atomic(output / "summary.json", {
        **plan, "status": "completed", "error": None, "retry_count": 0,
        "dataset_rows": len(dataset), "record_count": len(augmented), "model_count": len(model_summaries),
        "definitions": definitions, "models": model_summaries,
    })
    write_json_atomic(output / "audit.json", {
        **plan, "status": "passed", "error": None, "retry_count": 0,
        "original_rows_preserved": len(augmented), "foreground_area_and_extent_checks": 4 * len(augmented),
        "canonical_pairs_per_model": len(dataset), "models": list(model_summaries),
        "image_count": len(image_keys), "failed_image_count": len(failed),
    })
    write_json_atomic(plan_path, {**plan, "status": "completed", "error": None, "retry_count": 0,
                                  "workers": args.workers, "image_count": len(image_keys)})
    print(f"Completed: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
