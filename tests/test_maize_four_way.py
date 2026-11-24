from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from wgsr_v9.io import sha256_path, write_jsonl
from wgsr_v9.maize.cot import REFUSAL_TEXT
from wgsr_v9.maize.experiment import (
    EXECUTOR_PROTOCOL, INFERENCE_PROTOCOL, LABELS, PLANNER_FORMAT, ROOT,
    executor_prompt, planner_messages, simple_prompt, validate_executor_row, validate_growth_cot,
)
from wgsr_v9.maize.planner_export import export_maize_results
from wgsr_v9.maize.training import unique_index
from wgsr_v9.executor_inference.runner import expected_source_size
from wgsr_v9.executor_inference.manifest import validate_prompt_rows
from wgsr_v9.planner_inference import protocol_version_for_cot, source_request, validate_completion


def cot() -> str:
    lines = ["<GROWTH_COT>", "Stage reasoning", "Current stage: S2.", "Future stage: S9.", "Growth plan"]
    for label in LABELS:
        line = label + ": The visible plant develops."
        if label not in ("Tassel", "Ear"):
            line += " Source=10.0%; Target=20.0%; Change=10.0 pp."
        lines.append(line)
    lines.extend(["Target phenotype", *(label + ": The plant is visible." for label in LABELS), "</GROWTH_COT>"])
    return "\n".join(lines)


def metadata() -> dict:
    return {"plant_id": "001a", "actual_interval_days": 18, "request_id": "p1", "pair_id": "p1",
            "species": "maize", "native_view": "sv1", "view": "side0", "time_condition_unit": "days",
            "source_size": [768, 1280], "protocol_version": EXECUTOR_PROTOCOL}


class MaizeFourWayTests(unittest.TestCase):
    def test_maize_grouped_loss_spans_keep_organs_in_text(self) -> None:
        from wgsr_v9.maize.grouped_loss import token_groups
        from wgsr_v9.maize.experiment import STABLE

        class CharacterTokenizer:
            def decode(self, ids, **kwargs):
                return "".join(map(chr, ids))

            def __call__(self, text, **kwargs):
                return {"input_ids": list(map(ord, text)),
                        "offset_mapping": [(i, i + 1) for i in range(len(text))]}

        for text in (cot(), "\n" + cot() + "<|im_end|>\n",
                     cot().replace("Source=10.0%; Target=20.0%; Change=10.0 pp.", STABLE, 1)):
            groups = token_groups(tuple(map(ord, text)), CharacterTokenizer())
            self.assertEqual(set(groups), {1, 2, 3, 4, 5, 6, 8})
            for organ in ("Tassel:", "Ear:"):
                start = text.index(organ)
                self.assertEqual(set(groups[start:text.index("\n", start)]), {1})
            future = text.index("S9.")
            self.assertEqual(groups[future:future + 3], [3, 3, 3])

    def test_source_inputs_ignore_all_future_and_teacher_fields(self) -> None:
        clean = metadata()
        dirty = {**clean, "target": "SECRET_TARGET", "source_stage_id": "SECRET_STAGE",
                 "metric_evidence": {"target": "SECRET_METRIC"}, "teacher_cot": "SECRET_TEACHER"}
        teacher_root = ROOT / "protocols/maize_sv1_v1/cot_v2_en_v2"
        self.assertEqual(simple_prompt(clean), simple_prompt(dirty))
        self.assertEqual(planner_messages(clean, teacher_root), planner_messages(dirty, teacher_root))
        self.assertIn("18 days", simple_prompt(dirty))
        self.assertNotIn("SECRET", json.dumps(planner_messages(dirty, teacher_root)))
        with self.assertRaises(ValueError):
            simple_prompt({**clean, "actual_interval_days": -1})

    def test_maize_stages_and_numeric_contract_are_separate_from_rice(self) -> None:
        valid, error = validate_completion(cot(), "stop", numeric_evidence_format=PLANNER_FORMAT,
                                           view="side0", output_wrapper="GROWTH_COT")
        self.assertEqual(valid, cot())
        self.assertIsNone(error)
        self.assertEqual(protocol_version_for_cot("v3", PLANNER_FORMAT, "days", enable_thinking=False,
                                                 output_wrapper="GROWTH_COT"), INFERENCE_PROTOCOL)
        for invalid in (cot().replace("S9.", "S10."), cot().replace("Change=10.0", "Change=-10.0"),
                        cot() + "\nextra", cot().replace("Tassel:", "Panicle:")):
            with self.assertRaises(ValueError):
                validate_growth_cot(invalid)
        with self.assertRaises(ValueError):
            protocol_version_for_cot("v3", PLANNER_FORMAT, "weeks", enable_thinking=False, output_wrapper="GROWTH_COT")

    def test_refusals_cannot_become_training_phenotypes(self) -> None:
        for text in ("I cannot provide these annotations.", "I can’t provide a forecast.",
                     "I therefore cannot complete this task."):
            self.assertRegex(text, REFUSAL_TEXT)
            with self.assertRaises(ValueError):
                validate_growth_cot(cot().replace("The visible plant develops.", text))
        self.assertNotRegex("The ear cannot be observed from this view.", REFUSAL_TEXT)

    def test_executor_refuses_teacher_provenance_for_predicted_arm(self) -> None:
        row = {**metadata(), "executor_variant": "predicted_cot", "prompt": executor_prompt(metadata(), cot()),
               "cot_source": "teacher_annotation", "adapter_sha256": "adapter"}
        with self.assertRaises(ValueError):
            validate_prompt_rows([row], "predicted_cot", expected_protocol=EXECUTOR_PROTOCOL)
        row["cot_source"] = "planner_prediction"
        validate_executor_row(row, "predicted_cot")
        row["prompt"] = row["prompt"].replace("18 days", "3 weeks")
        with self.assertRaises(ValueError):
            validate_executor_row(row, "predicted_cot")

    def test_native_size_opt_in_does_not_change_rice(self) -> None:
        with Image.new("RGB", (768, 1280)) as maize:
            self.assertEqual(expected_source_size(metadata(), maize), (768, 1280))
            with self.assertRaises(ValueError):
                expected_source_size({"request_id": "rice"}, maize)
        with Image.new("RGB", (672, 896)) as rice:
            self.assertEqual(expected_source_size({"request_id": "rice"}, rice), (672, 896))
            with self.assertRaises(ValueError):
                expected_source_size(metadata(), rice)

    def test_duplicate_ids_rejected(self) -> None:
        with self.assertRaises(ValueError):
            unique_index([{"id": "x"}, {"id": "x"}], "id")

    def test_export_verifies_source_request_and_never_reads_future_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            source.write_bytes(b"source image bytes")
            package = root / "package"
            row = {"id": "p1", "input_path": str(source), "images": [str(source)],
                   "messages": [{"role": "system", "content": "source only"},
                                {"role": "user", "content": "<image> forecast 18 days"},
                                {"role": "assistant", "content": "SECRET_TEACHER"}],
                   "planner_metadata": metadata()}
            write_jsonl(package / "val.jsonl", [row])
            request_hash = hashlib.sha256(json.dumps(source_request(row, package), sort_keys=True).encode()).hexdigest()
            manifests = [{"dataset_sha256": sha256_path(package / "val.jsonl"), "protocol_version": INFERENCE_PROTOCOL,
                          "source_only": True, "benchmark": False, "status": "completed", "adapter_sha256": "trained_adapter"}]
            results = {"p1": {"input_path": str(source), "output_path": str(root / "raw.txt"),
                              "prompt_sha256": request_hash, "format_valid": True, "finish_reason": "stop", "growth_cot": cot()}}
            canonical = {"p1": {"source_image_path": str(source), "source_image_sha256": sha256_path(source),
                                 "target_image_path": str(root / "SECRET_MISSING_FUTURE.png")}}
            config = {"data_root": str(package), "expected_rows": 1}
            exported = root / "exported"
            summary = export_maize_results(config, exported, [row], canonical, manifests, results)
            self.assertEqual(summary["valid_rows"], 1)
            row_out = json.loads((exported / "executor_prompts.jsonl").read_text())
            self.assertNotIn("SECRET", row_out["prompt"])
            results["p1"]["prompt_sha256"] = "target-inclusive request hash"
            with self.assertRaises(ValueError):
                export_maize_results(config, root / "bad", [row], canonical, manifests, results)
            results["p1"].update(prompt_sha256=request_hash, format_valid=False)
            with self.assertRaises(ValueError):
                export_maize_results(config, root / "invalid", [row], canonical, manifests, results)
            self.assertFalse((root / "invalid/executor_prompts.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
