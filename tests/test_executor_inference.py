from __future__ import annotations

import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wgsr_v9.executor_inference import (  # noqa: E402
    ALGORITHM_VERSION,
    PROTOCOL_VERSION,
    canonical_sha256,
    default_checkpoint,
    default_input_manifest,
    default_output_root,
    load_input_rows,
    load_seed_map,
    make_transformer_view,
    parse_indices,
    row_seed,
    validate_config,
)
from wgsr_v9.executor_inference.checkpoint import pretrained_transformer_identity
from wgsr_v9.executor_inference.cli import parse_args
from wgsr_v9.executor_inference.runner import validate_resume
import yaml


class ExecutorInferenceContractTests(unittest.TestCase):
    def test_predicted_cot_uses_cot_checkpoint_and_own_input_and_output(self) -> None:
        config = {
            "executor_training_root": "runs/test_training",
            "executor_data_root": "data/test_training",
            "inference_run_root": "runs/test_predictions",
            "predicted_cot_manifest": "runs/planner/generated.jsonl",
        }
        self.assertEqual(default_checkpoint(config, "predicted_cot"),
                         default_checkpoint(config, "teacher_cot"))
        self.assertEqual(default_input_manifest(config, "predicted_cot"),
                         (ROOT / "runs/planner/generated.jsonl").resolve())
        self.assertNotEqual(default_input_manifest(config, "predicted_cot"),
                            default_input_manifest(config, "teacher_cot"))
        self.assertEqual(default_output_root(config, "predicted_cot").name, "predicted_cot")
        del config["predicted_cot_manifest"]
        with self.assertRaisesRegex(ValueError, "explicit generated CoT"):
            default_input_manifest(config, "predicted_cot")

    def test_parse_indices_and_stable_hash(self) -> None:
        self.assertEqual(parse_indices("3, 14,3"), {3, 14})
        self.assertEqual(canonical_sha256({"b": 1, "a": 2}), canonical_sha256({"a": 2, "b": 1}))
        with self.assertRaises(ValueError):
            parse_indices("-1")

    def test_load_rows_preserves_original_indices_and_seed_map(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "val.jsonl"
            rows = [
                {"request_id": "r0", "image": "target0.png", "edit_image": "source0.png", "prompt": "p0", "view": "top"},
                {"request_id": "r1", "image": "target1.png", "edit_image": "source1.png", "prompt": "p1", "view": "front"},
            ]
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            selected, digest = load_input_rows(manifest, {1})
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["inference_row_index"], 1)
            self.assertEqual(len(digest), 64)
            seed_manifest = root / "seeds.jsonl"
            seed_manifest.write_text(json.dumps({"request_id": "r1", "seed": 123}) + "\n", encoding="utf-8")
            seeds, _ = load_seed_map(seed_manifest)
            self.assertEqual(row_seed(selected[0], seeds, 99), 123)

    def test_transformer_view_is_checkpoint_specific_and_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            (model / "transformer").mkdir(parents=True)
            (model / "transformer/config.json").write_text("{}\n", encoding="utf-8")
            checkpoint = root / "epoch-1.safetensors"
            checkpoint.write_bytes(b"checkpoint")
            view = make_transformer_view(model, checkpoint, root / "run")
            self.assertEqual((view / "config.json").read_text(encoding="utf-8"), "{}\n")
            self.assertTrue((view / "diffusion_pytorch_model.safetensors").is_symlink())
            self.assertEqual((view / "diffusion_pytorch_model.safetensors").resolve(), checkpoint.resolve())
            self.assertEqual(make_transformer_view(model, checkpoint, root / "run"), view)

    def test_versions_are_explicit(self) -> None:
        self.assertTrue(ALGORITHM_VERSION.startswith("wgsr_v9_"))
        self.assertTrue(PROTOCOL_VERSION.startswith("wgsr_v9_"))

    def test_pretrained_uses_original_shards_and_resume_detects_weight_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transformer = root / "model/transformer"
            transformer.mkdir(parents=True)
            (transformer / "config.json").write_text("{}")
            index = transformer / "diffusion_pytorch_model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"first": "shard1.safetensors", "second": "shard2.safetensors"}}))
            (transformer / "shard1.safetensors").write_bytes(b"original1")
            shard = transformer / "shard2.safetensors"
            shard.write_bytes(b"original2")
            self.assertEqual(make_transformer_view(root / "model", None, root / "run"), transformer)
            self.assertFalse((root / "run").exists())
            identity = pretrained_transformer_identity(root / "model")
            self.assertEqual(len(identity["files_sha256"]), 4)
            shard.write_bytes(b"replaced2")
            changed = pretrained_transformer_identity(root / "model")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                validate_resume({"run_identity": identity}, changed)
            shard.unlink()
            with self.assertRaises(FileNotFoundError):
                pretrained_transformer_identity(root / "model")

    def test_pretrained_cli_never_falls_back_to_trained_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "infer.yaml"
            config = yaml.safe_load((ROOT / "configs/recipes/rice/executor_base.json").read_text())
            config.update(pretrained=True, physical_gpu_ids="0", world_size=1)
            path.write_text(yaml.safe_dump(config))
            command = ["infer", "--config", str(path), "--variant", "predicted_cot"]
            with patch.object(sys, "argv", command):
                args = parse_args()
                self.assertIsNone(args.checkpoint)
                self.assertTrue(args.pretrained)
            for extra in (["--checkpoint", "/tmp/trained.safetensors"],):
                with patch.object(sys, "argv", command + extra):
                    with self.assertRaisesRegex(ValueError, "cannot be combined"):
                        parse_args()
            config["checkpoints"] = {"teacher_cot": "/tmp/trained.safetensors"}
            path.write_text(yaml.safe_dump(config))
            with patch.object(sys, "argv", command):
                with self.assertRaisesRegex(ValueError, "must not contain"):
                    parse_args()

    def test_config_enforces_source_only_contract(self) -> None:
        config = {
            "version": ALGORITHM_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "source_only": True,
            "target_usage": "eval_reference_only",
            "num_inference_steps": 20,
            "world_size": 2,
            "physical_gpu_ids": "0,1",
        }
        validate_config(config)
        config["source_only"] = False
        with self.assertRaises(ValueError):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
