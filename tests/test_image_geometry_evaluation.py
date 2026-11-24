from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wgsr_v9.evaluation_io import read_jsonl, write_jsonl_atomic
from wgsr_v9.image_geometry_evaluation import (
    compare_geometry, measure_geometry_image, measure_geometry_mask,
)


class GeometryDefinitionTests(unittest.TestCase):
    def test_height_uses_image_bottom_and_is_unavailable_for_top_view(self) -> None:
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:4, 1:4] = True
        front = measure_geometry_mask(mask, "side0")
        top = measure_geometry_mask(mask, "top")
        self.assertEqual(front["front_height_pct"], 80.0)
        self.assertEqual(front["visible_extent_y_pct"], 60.0)
        self.assertEqual(front["bottom_gap_pct"], 20.0)
        self.assertEqual(front["measurement_status"], "bottom_gap")
        self.assertIsNone(top["front_height_pct"])
        self.assertEqual(compare_geometry(top, top)["height_metric_status"], "not_applicable")

    def test_density_detects_internal_fill_at_identical_extent(self) -> None:
        solid = np.zeros((5, 5), dtype=bool)
        solid[1:4, 1:4] = True
        sparse = np.zeros_like(solid)
        sparse[np.ix_([1, 3], [1, 3])] = True
        for view in ("front", "top"):
            target = measure_geometry_mask(solid, view)
            prediction = measure_geometry_mask(sparse, view)
            self.assertEqual(target["visible_extent_y_pct"], prediction["visible_extent_y_pct"])
            self.assertEqual(target["convex_hull_area_pixels"], 9)
            self.assertAlmostEqual(prediction["projection_density_fraction"], 4 / 9)
            self.assertAlmostEqual(compare_geometry(target, prediction)["projection_density_abs_error_pp"], 500 / 9)

    def test_empty_and_degenerate_masks_have_explicit_missing_values(self) -> None:
        mask = np.zeros((5, 5), dtype=bool)
        empty = measure_geometry_mask(mask, "front")
        self.assertIsNone(empty["front_height_pct"])
        self.assertIsNone(empty["projection_density_fraction"])
        mask[1:4, 2] = True
        line = measure_geometry_mask(mask, "front")
        self.assertEqual(line["front_height_pct"], 80.0)
        self.assertIsNone(line["projection_density_fraction"])
        comparison = compare_geometry(line, empty)
        self.assertIsNone(comparison["front_height_abs_error_pp"])
        self.assertIsNone(comparison["projection_density_abs_error_pp"])
        self.assertIn("prediction:empty_foreground", comparison["density_reason_codes"])
        self.assertIn("target:degenerate_hull", comparison["density_reason_codes"])

    def test_transparent_background_and_near_white_match_pixel_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgba.png"
            pixels = np.zeros((5, 5, 4), dtype=np.uint8)
            pixels[1:4, 1:4] = (40, 120, 50, 255)
            pixels[0, 0] = (230, 230, 230, 255)
            Image.fromarray(pixels).save(path)
            result = measure_geometry_image(path, "side0")
        self.assertEqual(result["visible_foreground_pixels"], 9)
        self.assertEqual(result["front_height_pct"], 80.0)
        self.assertEqual(result["projection_density_fraction"], 1.0)


class GeometryAugmentationTests(unittest.TestCase):
    def test_augmentation_preserves_fields_counts_missing_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgb = np.full((8, 8, 3), 255, dtype=np.uint8)
            rgb[2:6, 2:6] = (40, 120, 50)
            image, blank = root / "plant.png", root / "blank.png"
            Image.fromarray(rgb).save(image)
            Image.new("RGB", (8, 8), "white").save(blank)
            dataset = [
                {"pair_id": f"p{i}", "row_index": i, "view": view,
                 "source_image_path": str(image), "target_image_path": str(image)}
                for i, view in enumerate(("side0", "side0", "top"))
            ]
            manifest, ledger = root / "dataset.jsonl", root / "results.jsonl"
            write_jsonl_atomic(manifest, dataset)
            write_jsonl_atomic(ledger, [
                {"pair_id": row["pair_id"], "status": "completed", "output_path": str(blank if i == 1 else image)}
                for i, row in enumerate(dataset)
            ])
            env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
            subprocess.run([
                sys.executable, str(ROOT / "scripts/evaluate_image_results.py"),
                "--dataset-manifest", str(manifest), "--result", f"test={ledger}",
                "--output-dir", str(root / "original"),
            ], check=True, capture_output=True, text=True, env=env)
            old = read_jsonl(root / "original/per_pair_results.jsonl")
            # Legacy merged summaries have no manifest hash; per-row hashes
            # and canonical paths must still provide exact input alignment.
            summary_path = root / "original/summary.json"
            legacy = json.loads(summary_path.read_text())
            legacy["dataset_manifest_sha256"] = None
            summary_path.write_text(json.dumps(legacy))
            command = [
                sys.executable, str(ROOT / "scripts/augment_geometry_metrics.py"),
                "--dataset-manifest", str(manifest), "--evaluation-dir", str(root / "original"),
                "--output-dir", str(root / "augmented"), "--workers", "2",
            ]
            subprocess.run(command, check=True, capture_output=True, text=True, env=env)
            result = read_jsonl(root / "augmented/per_pair_results.jsonl")
            for before, after in zip(old, result, strict=True):
                self.assertEqual(before, {key: after[key] for key in before})
            summary = json.loads((root / "augmented/summary.json").read_text())["models"]["test"]
            self.assertEqual(summary["metrics"]["front_height_abs_error_pp"]["n"], 1)
            self.assertEqual(summary["metrics"]["projection_density_abs_error_pp"]["n"], 2)
            self.assertEqual(summary["height_metric_status"], {"available": 1, "unavailable": 1, "not_applicable": 1})
            resumed = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
            self.assertIn("0 pending", resumed.stdout)
            self.assertEqual(result, read_jsonl(root / "augmented/per_pair_results.jsonl"))


if __name__ == "__main__":
    unittest.main()
