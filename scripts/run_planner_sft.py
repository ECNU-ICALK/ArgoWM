#!/usr/bin/env python3
"""Preview or explicitly launch the single proposed-method planner SFT run.

The default action is a command preview. Use ``--launch`` only after the
prepared source-only package, the real Qwen3.8-27B checkpoint, and the
reviewed ms-swift environment have been independently checked. Simple is a
direct image-executor baseline and has no planner launcher or no-CoT run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
DEFAULT_CONFIG = REPO_ROOT / "configs/local/rice/planner_train.json"


def _flag(args: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        args.extend([name, "true" if value else "false"])
    else:
        args.extend([name, str(value)])


def _complete_model_directory(path: Path) -> bool:
    config = path / "config.json"
    index = path / "model.safetensors.index.json"
    tokenizer = path / "tokenizer.json"
    tokenizer_config = path / "tokenizer_config.json"
    shards = sorted(path.glob("model-*-of-*.safetensors"))
    if not all(item.is_file() for item in (config, index, tokenizer, tokenizer_config)) or not shards:
        return False
    marker = shards[0].name.split("-of-")[-1].split(".")[0]
    try:
        return len(shards) == int(marker)
    except ValueError:
        return False


def build_command(config: dict[str, Any]) -> list[str]:
    section = config.get("planner")
    if not isinstance(section, dict):
        raise ValueError("training config must contain one planner section")
    data_dir = (REPO_ROOT / str(section["dataset_dir"])).resolve()
    output_dir = (REPO_ROOT / str(section["output_dir"])).resolve()
    command = ["swift", "sft"]
    shared = (
        ("--model", config["model_path"]),
        # ms-swift 4.6 names this CLI field ``tuner_type``.  Keep the
        # repository config's historical ``train_type`` key for readability,
        # but translate it at the process boundary.
        ("--tuner_type", config["train_type"]),
        ("--template", config["template"]),
        ("--torch_dtype", config["torch_dtype"]),
        ("--num_train_epochs", config["num_train_epochs"]),
        ("--per_device_train_batch_size", config["per_device_train_batch_size"]),
        ("--per_device_eval_batch_size", config["per_device_eval_batch_size"]),
        ("--gradient_accumulation_steps", config["gradient_accumulation_steps"]),
        ("--learning_rate", config["learning_rate"]),
        ("--lr_scheduler_type", config["lr_scheduler_type"]),
        ("--warmup_ratio", config["warmup_ratio"]),
        ("--weight_decay", config["weight_decay"]),
        ("--max_length", config["max_length"]),
        ("--max_pixels", config["max_pixels"]),
        ("--eval_strategy", config["eval_strategy"]),
        ("--save_strategy", config["save_strategy"]),
        ("--save_steps", config["save_steps"]),
        ("--save_total_limit", config["save_total_limit"]),
        ("--logging_steps", config["logging_steps"]),
        ("--seed", config["seed"]),
        ("--data_seed", config["data_seed"]),
        ("--lora_rank", config["lora_rank"]),
        ("--lora_alpha", config["lora_alpha"]),
        ("--lora_dropout", config["lora_dropout"]),
        ("--target_modules", config["target_modules"]),
        ("--freeze_vit", config["freeze_vision_tower"]),
        ("--freeze_aligner", config["freeze_aligner"]),
        ("--check_model", config["check_model"]),
    )
    for name, value in shared:
        _flag(command, name, value)
    _flag(command, "--dataset", str(data_dir / "train.jsonl"))
    _flag(command, "--val_dataset", str(data_dir / "val.jsonl"))
    _flag(command, "--lr_scheduler_kwargs", json.dumps({"min_lr": config["lr_scheduler_min_lr"]}))
    _flag(command, "--output_dir", str(output_dir))
    _flag(command, "--enable_thinking", section["enable_thinking"])
    # FSDP2's preset enables native activation checkpointing.  ms-swift warns
    # and may disable the legacy Trainer flag when both are supplied, so keep
    # the command unambiguous and only use the legacy flag without FSDP2.
    fsdp = config.get("fsdp")
    _flag(command, "--fsdp", fsdp)
    _flag(command, "--deepspeed", config.get("deepspeed"))
    _flag(command, "--gradient_checkpointing", not bool(fsdp))
    if config.get("max_steps") is not None:
        _flag(command, "--max_steps", config["max_steps"])
    grouped = config.get("grouped_loss")
    if grouped is not None:
        if config["gradient_accumulation_steps"] != 1:
            raise ValueError("grouped planner loss currently requires gradient accumulation 1")
        from wgsr_v9.planner_grouped_loss import validate_weights
        validate_weights(tuple(float(grouped["weights"][k]) for k in ("stage", "metric", "text")))
        maize = config.get("species") == "maize"
        plugin = "maize_planner_grouped_loss_plugin.py" if maize else "planner_grouped_loss_plugin.py"
        _flag(command, "--external_plugins", str(REPO_ROOT / "scripts" / plugin))
        _flag(command, "--loss_type", "wgsr_maize_planner_grouped_v1" if maize else "wgsr_planner_grouped_v1")
        _flag(command, "--average_tokens_across_devices", False)
        _flag(command, "--loss_scale", "default+ignore_empty_think")
        _flag(command, "--packing", False)
        _flag(command, "--padding_free", False)
    _flag(command, "--dataset_num_proc", 4)
    _flag(command, "--dataloader_num_workers", 4)
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--launch", action="store_true", help="Run the one planner command.")
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Override max_steps for a bounded smoke test without editing the reviewed config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override planner output_dir, primarily for isolated smoke tests.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="Override planner dataset_dir, primarily for isolated smoke tests.",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("training config must be a YAML object")
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be positive")
        config["max_steps"] = args.max_steps
    if args.output_dir is not None:
        section = config.setdefault("planner", {})
        if not isinstance(section, dict):
            raise ValueError("training config planner section must be an object")
        section["output_dir"] = str(args.output_dir)
    if args.dataset_dir is not None:
        section = config.setdefault("planner", {})
        if not isinstance(section, dict):
            raise ValueError("training config planner section must be an object")
        section["dataset_dir"] = str(args.dataset_dir)
    if config.get("experiment_role") != "proposed_method_planner":
        raise ValueError("training config is not marked as the proposed-method planner run")
    if config.get("active_planner_variant") != "proposed_cot":
        raise ValueError("training config must select the single proposed_cot planner")
    if str(config.get("cot_version", "")) not in {"v2", "v3"}:
        raise ValueError("training config must select cot_version v2 or v3")
    if config.get("simple_baseline_uses_planner") is not False:
        raise ValueError("Simple baseline must bypass the planner")
    section = config.get("planner")
    if not isinstance(section, dict):
        raise ValueError("training config must contain exactly one active planner section")
    data_dir = (REPO_ROOT / str(section["dataset_dir"])).resolve()
    required = (data_dir / "train.jsonl", data_dir / "val.jsonl", data_dir / "package_manifest.json")
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"missing prepared planner artifact {path}; run scripts/prepare_data.py first")
    package_manifest = json.loads((data_dir / "package_manifest.json").read_text(encoding="utf-8"))
    if package_manifest.get("planner_variant") != "proposed_cot":
        raise ValueError("dataset is not the active proposed_cot planner package")
    if package_manifest.get("active_no_cot_planner") is not False:
        raise ValueError("active package incorrectly advertises a no-CoT planner")
    if package_manifest.get("cot_version") != config.get("cot_version"):
        raise ValueError(
            f"dataset CoT version {package_manifest.get('cot_version')!r} does not match "
            f"training config {config.get('cot_version')!r}"
        )
    for field in ("protocol_version", "numeric_evidence_format", "time_condition_unit"):
        expected = config.get("planner_protocol_version" if field == "protocol_version" else field)
        if expected is not None and package_manifest.get(field) != expected:
            raise ValueError(f"Dataset {field} does not match the configured planner protocol")
    if package_manifest.get("enable_thinking", True) != section["enable_thinking"]:
        raise ValueError("Dataset thinking mode does not match training configuration")
    if package_manifest.get("output_wrapper", "think") != config.get("output_wrapper", "think"):
        raise ValueError("Dataset output wrapper does not match training configuration")
    command = build_command(config)
    model_path = Path(str(config["model_path"]))
    swift_executable = Path(str(config.get("swift_executable", "swift")))
    swift_command = str(swift_executable) if swift_executable.is_file() else "swift"
    if command and command[0] == "swift":
        command[0] = swift_command
    payload = {
        "algorithm_version": "wgsr_v9_planner_sft_launcher_v3",
        "config": str(args.config.resolve()),
        "planner_variant": "proposed_cot",
        "cot_version": config["cot_version"],
        "enable_thinking": section["enable_thinking"],
        "output_wrapper": config.get("output_wrapper", "think"),
        "simple_baseline_uses_planner": False,
        "distributed_strategy": config.get("deepspeed") or config.get("fsdp") or "ddp",
        "command": command,
        "launcher_env": {
            "NPROC_PER_NODE": str(config["world_size"]),
            "CUDA_VISIBLE_DEVICES": str(config.get("cuda_visible_devices", "")),
            "NCCL_P2P_DISABLE": "1",
            "NCCL_IB_DISABLE": "1",
            "ROOT_IMAGE_DIR": str(data_dir),
        },
        "status": "launch_requested" if args.launch else "preview",
        "error": None,
        "retry_count": 0,
    }
    if config.get("grouped_loss") is not None:
        payload["grouped_loss"] = config["grouped_loss"]
        payload["launcher_env"]["WGSR_PLANNER_LOSS_WEIGHTS"] = json.dumps(config["grouped_loss"]["weights"], sort_keys=True)
    if args.print_json or not args.launch:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not args.launch:
        return 0
    swift_executable = Path(str(config.get("swift_executable", ""))).expanduser()
    swift_available = swift_executable.is_file() if swift_executable.name else shutil.which("swift") is not None
    if not swift_available:
        raise RuntimeError(
            f"ms-swift executable is unavailable: {swift_executable or 'swift on PATH'}; "
            "see docs/INSTALL.md"
        )
    if config.get("deepspeed"):
        try:
            import importlib.util

            if importlib.util.find_spec("deepspeed") is None:
                raise RuntimeError("configured DeepSpeed strategy requires deepspeed in the planner environment")
        except ModuleNotFoundError as exc:
            raise RuntimeError("configured DeepSpeed strategy requires deepspeed in the planner environment") from exc
    if not _complete_model_directory(model_path):
        raise RuntimeError(
            f"configured Qwen3.8-27B directory is incomplete: {model_path}; "
            "wait for all safetensors shards, index, and tokenizer files"
        )
    launch_env = os.environ.copy()
    launch_env["NPROC_PER_NODE"] = str(config["world_size"])
    if config.get("cuda_visible_devices"):
        launch_env["CUDA_VISIBLE_DEVICES"] = str(config["cuda_visible_devices"])
    # Accelerate refuses NCCL P2P/IB probing on RTX 4000-series devices.  The
    # same settings are required when ms-swift dispatches torchrun directly.
    # Keep them in the launcher so previewed configs are reproducible on the
    # current 4090 host and on other Ada workstations.
    launch_env.setdefault("NCCL_P2P_DISABLE", "1")
    launch_env.setdefault("NCCL_IB_DISABLE", "1")
    # ShareGPT rows intentionally store package-relative image paths.  ms-
    # swift resolves these through ROOT_IMAGE_DIR; point it at the selected
    # package so both the smoke override and the regular data package work.
    launch_env.setdefault("ROOT_IMAGE_DIR", str(data_dir))
    if config.get("grouped_loss") is not None:
        launch_env["WGSR_PLANNER_LOSS_WEIGHTS"] = json.dumps(config["grouped_loss"]["weights"], sort_keys=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True, env=launch_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
