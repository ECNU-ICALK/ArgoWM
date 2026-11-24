#!/usr/bin/env python3
"""Evaluate maize predictions with the same four proxies used in the teacher data."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wgsr_v9.evaluation_io import add_common_arguments, load_dataset_manifest, load_predictions, parse_result_specs
from wgsr_v9.io import write_json, write_jsonl
from wgsr_v9.maize.experiment import SIZE
from wgsr_v9.maize.training import provenance
from wgsr_v9.v9_image_metrics import measure_rgb

METRICS = {"culm": 1.0, "leaves_area": 1.0, "plant_density": 100.0, "yellow_brown_fraction": 100.0}


def measure(path: str) -> tuple[str, dict[str, Any]]:
    with Image.open(path) as image:
        if image.size != SIZE:
            raise ValueError(f"Expected native maize geometry {SIZE}: {path}")
        measured = measure_rgb(np.asarray(image.convert("RGB")), view="side0")
    return path, {"metrics": measured["metrics"], "metric_status": measured["metric_status"],
                  "metric_reason_codes": measured["metric_reason_codes"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    manifest, rows, manifest_hash = load_dataset_manifest(args.dataset_manifest, args.path_root)
    ids = {row["pair_id"] for row in rows}
    if len(ids) != len(rows) or any(row.get("species") != "maize" for row in rows):
        raise ValueError("Expected unique maize validation IDs")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use a new evaluation directory")
    predictions = {}
    paths = {row["target_image_path"] for row in rows}
    for name, path in parse_result_specs(args.result).items():
        values, _ = load_predictions(path, extra_roots=args.path_root, dataset_rows=rows)
        if any(("pair_id", key) not in values for key in ids):
            raise ValueError(f"{name}: incomplete predictions; all canonical IDs are required")
        predictions[name] = values
        paths.update(str(values[("pair_id", key)].output_path) for key in ids)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        measured = dict(pool.map(measure, sorted(paths), chunksize=8))
    per_pair = []
    summary = {}
    for name, predictions_for_method in predictions.items():
        method_rows = []
        for row in rows:
            pred = predictions_for_method[("pair_id", row["pair_id"])].output_path
            target, generated = measured[row["target_image_path"]], measured[str(pred)]
            record = {**provenance(pred, args.output_dir / "per_pair.jsonl", status="completed"),
                      "model_name": name, "pair_id": row["pair_id"], "horizon_bucket": row["horizon_bucket"],
                      "target_path": row["target_image_path"], "metric_status": {}, "errors_pp": {}}
            for key, scale in METRICS.items():
                states = [m["metric_status"].get(key) for m in (target, generated)]
                available = states == ["available", "available"]
                record["metric_status"][key] = {"target": states[0], "prediction": states[1]}
                record["errors_pp"][key] = (scale * abs(target["metrics"][key] - generated["metrics"][key])
                                              if available else None)
            method_rows.append(record)
        summary[name] = {}
        for group in ("all", "short", "mid", "long"):
            selected = [row for row in method_rows if group == "all" or row["horizon_bucket"] == group]
            summary[name][group] = {"rows": len(selected), "metrics": {}}
            for key in METRICS:
                values = [row["errors_pp"][key] for row in selected if row["errors_pp"][key] is not None]
                summary[name][group]["metrics"][key] = {
                    "mae_pp": statistics.mean(values) if values else None,
                    "available": len(values), "unavailable": len(selected) - len(values)}
        per_pair.extend(method_rows)
    write_jsonl(args.output_dir / "per_pair.jsonl", per_pair)
    output = {**provenance(manifest, args.output_dir, status="completed"), "dataset_sha256": manifest_hash,
              "method_summary": summary,
              "metric_policy": "Available evidence only; report each denominator. Culm uses image-bottom anchor. No prediction cleanup or crop."}
    write_json(args.output_dir / "summary.json", output)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
