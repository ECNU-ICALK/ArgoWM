from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wgsr_v9.diffsynth_batching import LayoutBatchSampler, collate_cached  # noqa: E402
from wgsr_v9.executor_training import (  # noqa: E402
    ALGORITHM_VERSION,
    PROTOCOL_VERSION,
    WEEKS_PROTOCOL_VERSION,
    executor_prompt,
    growth_cot_body,
    make_executor_row,
    prediction_interval_weeks,
)


PROMPT_TEMPLATE = "Source {accession_id}; {view_label}; {interval_weeks} weeks ({interval_days} days)."
COT_SUFFIX = "\n<GROWTH_COT>\n{growth_cot}\n</GROWTH_COT>"


def _request(root: Path) -> dict[str, object]:
    source = root / "source.png"
    target = root / "target.png"
    source.touch()
    target.touch()
    return {
        "request_id": "req-1",
        "pair_group_id": "pair-1",
        "split": "train",
        "plant_id": "plant-1",
        "view": "front",
        "accession_id": "R0001",
        "actual_interval_days": 8,
        "source": {"path": str(source)},
        "target": {"path": str(target)},
    }


class ExecutorPromptTests(unittest.TestCase):
    def test_weeks_protocol_rounding_and_legacy_compatibility(self) -> None:
        template = "Source {accession_id}; {view_label}; Prediction interval: {interval_weeks} weeks."
        with tempfile.TemporaryDirectory() as temporary:
            request = _request(Path(temporary))
            for days, expected in ((4, 1), (8, 1), (10, 1), (11, 2), (29, 4), (57, 8), (71, 10), (140, 20)):
                request["actual_interval_days"] = days
                self.assertEqual(prediction_interval_weeks(request), expected)
                prompt = executor_prompt(request, prompt_template=template, cot_suffix=COT_SUFFIX,
                                         variant="teacher_cot", teacher_cot="Original CoT.",
                                         protocol_version=WEEKS_PROTOCOL_VERSION)
                self.assertIn(f"Prediction interval: {expected} weeks.", prompt)
                self.assertNotIn("days", prompt)
                self.assertTrue(prompt.endswith("Original CoT.\n</GROWTH_COT>"))
                row = make_executor_row(request, prompt=prompt, variant="teacher_cot",
                                        original_row_index=0, protocol_version=WEEKS_PROTOCOL_VERSION)
                self.assertEqual(row["actual_interval_days"], days)
                self.assertEqual(row["prediction_interval_weeks"], expected)
                self.assertEqual(row["time_condition_unit"], "weeks")
            request.update(actual_interval_days=71, prediction_interval_weeks=11)
            with self.assertRaises(ValueError):
                prediction_interval_weeks(request)
            request["prediction_interval_weeks"] = 10
            with self.assertRaises(ValueError):
                executor_prompt(request, prompt_template=PROMPT_TEMPLATE, cot_suffix=COT_SUFFIX,
                                variant="simple", protocol_version=WEEKS_PROTOCOL_VERSION)

    def test_simple_and_teacher_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            request = _request(Path(temporary))
            simple = executor_prompt(
                request,
                prompt_template=PROMPT_TEMPLATE,
                cot_suffix=COT_SUFFIX,
                variant="simple",
            )
            cot = executor_prompt(
                request,
                prompt_template=PROMPT_TEMPLATE,
                cot_suffix=COT_SUFFIX,
                variant="teacher_cot",
                teacher_cot="Line 1.\nLine 2.",
            )
            self.assertEqual(simple, "Source R0001; front view; 2 weeks (8 days).")
            self.assertNotIn("GROWTH_COT", simple)
            self.assertTrue(cot.endswith("<GROWTH_COT>\nLine 1.\nLine 2.\n</GROWTH_COT>"))

    def test_rejects_bad_wrapper_and_simple_cot(self) -> None:
        with self.assertRaises(ValueError):
            growth_cot_body("not wrapped", "req-1")
        with self.assertRaises(ValueError):
            growth_cot_body("<GROWTH_COT><GROWTH_COT>x</GROWTH_COT></GROWTH_COT>", "req-1")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                executor_prompt(
                    _request(Path(temporary)),
                    prompt_template=PROMPT_TEMPLATE,
                    cot_suffix=COT_SUFFIX,
                    variant="simple",
                    teacher_cot="should not be present",
                )

    def test_diff_synth_row_roles_and_audit_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = _request(root)
            row = make_executor_row(
                request,
                prompt="simple prompt",
                variant="simple",
                original_row_index=3,
            )
            self.assertEqual(row["image"], str((root / "target.png").resolve()))
            self.assertEqual(row["edit_image"], str((root / "source.png").resolve()))
            self.assertEqual(row["input_path"], row["edit_image"])
            self.assertEqual(row["output_path"], row["image"])
            self.assertEqual(row["algorithm_version"], ALGORITHM_VERSION)
            self.assertEqual(row["protocol_version"], PROTOCOL_VERSION)
            self.assertEqual((row["status"], row["error"], row["retry_count"]), ("ready", None, 0))


def _cache_row(mask: torch.Tensor, latent_shape: tuple[int, ...] = (1, 64, 56, 42)):
    return (
        {
            "input_latents": torch.ones(latent_shape),
            "edit_latents": [torch.ones(latent_shape) * 2],
        },
        {"prompt_embeds": torch.ones((1, mask.shape[1], 4)), "edit_image_pad_mask": mask},
        {"negative_prompt": ""},
    )


class CachedBatchTests(unittest.TestCase):
    def test_collate_batches_values_without_padding(self) -> None:
        mask = torch.ones((1, 4), dtype=torch.bool)
        rows = [_cache_row(mask), _cache_row(mask)]
        shared, positive, _ = collate_cached(rows)
        self.assertEqual(tuple(shared["input_latents"].shape), (2, 64, 56, 42))
        self.assertEqual(tuple(shared["edit_latents"][0].shape), (2, 64, 56, 42))
        self.assertEqual(tuple(positive["prompt_embeds"].shape), (2, 4, 4))
        self.assertEqual(tuple(positive["edit_image_pad_mask"].shape), (1, 4))

    def test_collate_rejects_mixed_layout(self) -> None:
        with self.assertRaises(ValueError):
            collate_cached(
                [
                    _cache_row(torch.ones((1, 4), dtype=torch.bool)),
                    _cache_row(torch.tensor([[True, False, True, False]])),
                ]
            )

    def test_collate_pads_prompt_sequences_and_masks_tail(self) -> None:
        short = _cache_row(torch.tensor([[True, True, True, True, False]], dtype=torch.bool))
        long = _cache_row(torch.tensor([[True, True, True, True, False, False, False]], dtype=torch.bool))
        shared, positive, _ = collate_cached([short, long])
        self.assertEqual(tuple(positive["prompt_embeds"].shape), (2, 7, 4))
        self.assertEqual(tuple(positive["edit_image_pad_mask"].shape), (1, 7))
        self.assertEqual(tuple(positive["prompt_embeds_mask"].shape), (2, 7))
        self.assertTrue(bool(positive["prompt_embeds_mask"][0, :5].all()))
        self.assertFalse(bool(positive["prompt_embeds_mask"][0, 5:].any()))
        self.assertEqual(tuple(shared["input_latents"].shape[:2]), (2, 64))

    def test_sampler_covers_rows_and_closes_accumulation_windows(self) -> None:
        keys = ["front-short"] * 9 + ["top-long"] * 5
        sampler = LayoutBatchSampler(
            keys,
            batch_size=2,
            world_size=4,
            seed=3,
            gradient_accumulation_steps=2,
        )
        batches = list(sampler)
        self.assertEqual(len(batches) % (4 * 2), 0)
        self.assertTrue(all(len(batch) == 2 for batch in batches))
        self.assertTrue(set(range(len(keys))).issubset({index for batch in batches for index in batch}))
        self.assertTrue(all(len({keys[index] for index in batch}) == 1 for batch in batches))


if __name__ == "__main__":
    unittest.main()
