"""Prevent obsolete inference contracts and missing default teacher schemas."""
import unittest

from wgsr_v9.cot_synthesis import render_cot
from wgsr_v9.planner_inference import protocol_version_for_cot


class MainlineProtocolTests(unittest.TestCase):
    def test_legacy_planner_contracts_are_rejected_before_inference(self):
        for version, numeric, unit in (
            ("v2", None, None),
            ("v3", None, None),
            ("v3", "metric_named_v1", "weeks"),
        ):
            with self.subTest(version=version, numeric=numeric):
                with self.assertRaisesRegex(ValueError, "mainline"):
                    protocol_version_for_cot(version, numeric, unit)

    def test_default_rice_renderer_uses_named_metrics_for_both_views(self):
        for view, keys in (
            ("front", ("culm", "leaves_area", "plant_density", "panicle", "plant_color_development")),
            ("top", ("leaves_area", "canopy", "panicle", "plant_color_development")),
        ):
            with self.subTest(view=view):
                annotation = {
                    "stage_reasoning": {"current_stage": "S1_tillering", "future_stage": "S2_jointing"},
                    "growth_plan": {key: {"text": "Plant develops"} for key in keys},
                    "target_phenotype": {key: {"text": "Green plant"} for key in keys},
                }
                evidence = {"leaves_area": {
                    "qc": {"valid": True},
                    "source": {"status": "available", "value": 10},
                    "target": {"status": "available", "value": 12},
                }}
                try:
                    text, omissions = render_cot(annotation, view=view, metric_evidence=evidence)
                except FileNotFoundError as exc:
                    self.fail(f"Default teacher schema is not shipped: {exc}")
                self.assertIn("Metric = Foreground area; Source = 10.00%; Target = 12.00%; Change = +2.00 pp.", text)
                self.assertIn("<GROWTH_COT>", text)
                self.assertIn("Current stage: S1_tillering.", text)
                self.assertNotIn("leaves_area", omissions)
