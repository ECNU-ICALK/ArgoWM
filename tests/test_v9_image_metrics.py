import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from wgsr_v9.v9_image_metrics import (
    ALGORITHM_VERSION,
    MIN_COLOR_COVERAGE,
    build_pair_image_map,
    classify_colors,
    compute_image_metrics,
    measure_rgb,
)


class V9ImageMetricTests(unittest.TestCase):
    def test_front_area_height_and_bottom_diagnostics_use_pure_white_foreground(self):
        rgb = np.full((4, 5, 3), 255, dtype=np.uint8)
        rgb[1:3, 1:2] = (40, 150, 40)
        rgb[1:3, 2:3] = (120, 120, 120)

        result = measure_rgb(rgb, view="side0")

        self.assertEqual(result["visible_foreground_pixels"], 4)
        self.assertAlmostEqual(result["metrics"]["leaves_area"], 20.0)
        self.assertAlmostEqual(result["metrics"]["culm"], 75.0)
        self.assertAlmostEqual(result["metrics"]["plant_density"], 1.0)
        self.assertEqual(result["top_row"], 1)
        self.assertEqual(result["bottom_row"], 2)
        self.assertAlmostEqual(result["bottom_gap_pct"], 25.0)
        self.assertFalse(result["touches_top"])
        self.assertFalse(result["touches_bottom"])
        self.assertEqual(result["measurement_status"], "bottom_gap")
        self.assertEqual(result["metric_status"]["culm"], "available")
        self.assertEqual(result["metric_status"]["yellow_brown_fraction"], "pending")

    def test_top_exposes_canopy_and_boundary_diagnostics_without_height_slot(self):
        rgb = np.full((4, 5, 3), 255, dtype=np.uint8)
        rgb[0:2, 1:3] = (40, 150, 40)

        result = measure_rgb(rgb, view="top")

        self.assertAlmostEqual(result["metrics"]["leaves_area"], 20.0)
        self.assertAlmostEqual(result["metrics"]["canopy"], 1.0)
        self.assertNotIn("culm", result["metrics"])
        self.assertTrue(result["touches_top"])
        self.assertFalse(result["touches_bottom"])
        self.assertTrue(result["touches_image_boundary"])
        self.assertEqual(result["measurement_status"], "boundary_contact")

    def test_front_bottom_anchor_is_not_reported_as_a_boundary_warning(self):
        rgb = np.full((5, 5, 3), 255, dtype=np.uint8)
        rgb[2:5, 2:3] = (40, 150, 40)

        result = measure_rgb(rgb, view="side0")

        self.assertTrue(result["touches_bottom"])
        self.assertTrue(result["touches_image_boundary"])
        self.assertFalse(result["boundary_contact"])

    def test_color_classes_are_disjoint_and_available_at_threshold(self):
        rgb = np.array(
            [[(40, 150, 40), (210, 190, 40), (120, 70, 30), (200, 200, 200)]],
            dtype=np.uint8,
        )
        mask = np.ones((1, 4), dtype=bool)

        result = classify_colors(rgb, mask)

        self.assertAlmostEqual(result["green_fraction"], 0.25)
        self.assertAlmostEqual(result["yellow_fraction"], 0.25)
        self.assertAlmostEqual(result["brown_fraction"], 0.25)
        self.assertAlmostEqual(result["yellow_brown_fraction"], 0.50)
        self.assertAlmostEqual(result["color_coverage"], 0.75)
        self.assertEqual(result["status"], "available")
        self.assertGreaterEqual(result["color_coverage"], MIN_COLOR_COVERAGE)

    def test_low_color_coverage_keeps_raw_fraction_but_marks_metric_pending(self):
        rgb = np.array(
            [[(40, 150, 40), (210, 190, 40), (200, 200, 200), (200, 200, 200)]],
            dtype=np.uint8,
        )
        mask = np.ones((1, 4), dtype=bool)

        result = classify_colors(rgb, mask)

        self.assertAlmostEqual(result["yellow_brown_fraction"], 0.25)
        self.assertLess(result["color_coverage"], MIN_COLOR_COVERAGE)
        self.assertEqual(result["status"], "pending")

    def test_pair_manifest_maps_jpg_names_to_unique_pure_white_pngs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            pair_file = root / "pairs.jsonl"
            rows = [
                {
                    "view": "side0",
                    "source_image_id": "I0",
                    "target_image_id": "I1",
                    "source_filename": "plant_2020-06-27_side0.jpg",
                    "target_filename": "plant_2020-07-04_side0.jpg",
                    "source_group_id": "G0",
                    "target_group_id": "G1",
                },
                {
                    "view": "top",
                    "source_image_id": "I0T",
                    "target_image_id": "I1T",
                    "source_filename": "plant_2020-06-27_top.jpg",
                    "target_filename": "plant_2020-07-04_top.jpg",
                    "source_group_id": "G0",
                    "target_group_id": "G1",
                },
            ]
            pair_file.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            mapping = build_pair_image_map([pair_file], root / "images_pure_white")

        self.assertEqual(len(mapping), 4)
        self.assertEqual(
            mapping["I0"]["pure_white_path"],
            str(root / "images_pure_white" / "plant_2020-06-27_side0.png"),
        )
        self.assertEqual(mapping["I0"]["view"], "side0")
        self.assertEqual(mapping["I1T"]["view"], "top")

    def test_measurement_row_carries_traceability_and_algorithm_version(self):
        rgb = np.full((2, 2, 3), 255, dtype=np.uint8)
        rgb[:, :] = (40, 150, 40)

        result = measure_rgb(rgb, view="side0")

        self.assertEqual(result["algorithm_version"], ALGORITHM_VERSION)
        self.assertEqual(result["status"], "ready")
        self.assertIsNone(result["error"])
        self.assertEqual(result["retry_count"], 0)

    def test_compute_image_metrics_supports_deterministic_offset_and_limit_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            image_dir = root / "images_pure_white"
            image_dir.mkdir()
            rows = []
            for index in range(3):
                filename = f"plant_2020-06-2{index + 7}_side0.jpg"
                png_path = image_dir / filename.replace(".jpg", ".png")
                Image.new("RGB", (4, 4), (40, 150, 40)).save(png_path)
                rows.append(
                    {
                        "pair_id": f"P{index}",
                        "view": "side0",
                        "source_image_id": f"I{index}",
                        "target_image_id": f"IT{index}",
                        "source_filename": filename,
                        "target_filename": filename.replace("side0", "side0_target"),
                        "source_group_id": f"G{index}",
                        "target_group_id": f"GT{index}",
                    }
                )
                target_png = image_dir / filename.replace(".jpg", "_target.png")
                Image.new("RGB", (4, 4), (40, 150, 40)).save(target_png)
            pair_file = root / "pairs.jsonl"
            pair_file.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            output = root / "metrics.jsonl"

            summary = compute_image_metrics(
                [pair_file], pure_white_dir=image_dir, output=output, offset=2, limit=2
            )
            measured = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual(summary["measured_image_count"], 2)
        self.assertEqual([row["image_id"] for row in measured], ["I2", "IT0"])


if __name__ == "__main__":
    unittest.main()
