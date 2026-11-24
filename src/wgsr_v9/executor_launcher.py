#!/usr/bin/env python3
"""Preview or explicitly run one V9 Qwen Image 2.1 DiffSynth executor SFT.

Default behavior prints the exact commands. ``--launch`` is required to build
latent caches or train. Data packages and DiffSynth caches stay under ``data/``;
logs and checkpoints stay under ``runs/``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import yaml

from wgsr_v9.executor_lr import LearningRateSchedule

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/local/rice/executor_teacher_cot.json"
INDEX_SCRIPT = REPO_ROOT / "scripts/build_executor_cache_index.py"
CACHE_CHUNK_SCRIPT = REPO_ROOT / "scripts/build_executor_cache_chunked.py"
TRAIN_SCRIPT = REPO_ROOT / "scripts/train_executor_cached.py"


def _path(config: dict[str, Any], key: str) -> Path:
    value = Path(str(config[key])).expanduser()
    return value if value.is_absolute() else (REPO_ROOT / value).resolve()


def _model_paths(model_root: Path) -> tuple[list[Any], list[Any]]:
    text_encoder = sorted((model_root / "text_encoder").glob("model-*.safetensors"))
    vae = model_root / "vae/diffusion_pytorch_model.safetensors"
    dit = sorted((model_root / "transformer").glob("diffusion_pytorch_model-*.safetensors"))
    if not text_encoder or not vae.is_file() or not dit:
        raise FileNotFoundError(f"incomplete Qwen Image 2.1 model layout under {model_root}")
    return [[str(path.resolve()) for path in text_encoder], str(vae.resolve())], [
        [str(path.resolve()) for path in dit]
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_plan(config: dict[str, Any], variant: str, phase: str) -> dict[str, Any]:
    if config.get("numeric_condition", {}).get("enabled", False):
        raise ValueError("Numeric injection is not included in this public release; use a text Executor recipe")
    lr_schedule = LearningRateSchedule.from_config(config.get("lr_scheduler"))
    if variant not in config["variants"]:
        raise ValueError(f"variant is not enabled in config: {variant}")
    if float(config.get("image_dropout", -1)) != 0.0:
        raise ValueError("this reviewed recipe requires image_dropout=0.0")
    dropout = config.get("condition_dropout")
    ablation = config.get("condition_ablation")
    if ablation is not None:
        from wgsr_v9.executor_ablation import validate_policy
        validate_policy(ablation)
        if phase != "train" or variant != "teacher_cot" or dropout is not None:
            raise ValueError("fixed ablations require cached text-only training without random dropout")
    if dropout is not None:
        from wgsr_v9.executor_dropout import validate_policy
        validate_policy(dropout)
        if phase != "train" or variant != "teacher_cot":
            raise ValueError("use prepare_executor_dropout_cache.py for dropout caches; launch phase=train")
    if (
        int(config["world_size"])
        * int(config["per_device_train_batch_size"])
        * int(config["gradient_accumulation_steps"])
        != int(config["effective_global_batch_size"])
    ):
        raise ValueError("effective global batch size does not match world_size × batch × accumulation")

    diffsynth_repo = _path(config, "framework_repo")
    python = _path(config, "python")
    accelerate_bin = _path(config, "accelerate")
    model_root = _path(config, "model_path")
    data_root = _path(config, "data_root")
    run_root = _path(config, "run_root")
    package_root = data_root / variant
    train_metadata = package_root / "train.jsonl"
    cache_root = package_root / "cache_train"
    layout_index = package_root / "layout_index.json"
    compaction_audit = package_root / "cache_compaction_audit.json"
    variant_run_root = run_root / variant
    checkpoint_root = variant_run_root / "checkpoints"
    upstream_cache, upstream_dit = _model_paths(model_root)
    upstream_train = diffsynth_repo / "examples/qwen_image_21/model_training/train.py"
    if not upstream_train.is_file():
        raise FileNotFoundError(upstream_train)
    if not train_metadata.is_file():
        raise FileNotFoundError(f"executor package is missing: {train_metadata}; run scripts/prepare_data.py first")
    package_manifest_path = data_root / "package_manifest.json"
    if not package_manifest_path.is_file():
        raise FileNotFoundError(f"executor package manifest is missing: {package_manifest_path}")
    package_manifest = json.loads(package_manifest_path.read_text(encoding="utf-8"))
    variant_manifest = package_manifest.get("variants", {}).get(variant, {})
    if package_manifest.get("status") != "complete" or variant_manifest.get("status") != "complete":
        raise ValueError("executor package manifest is not complete")
    if package_manifest.get("algorithm_version") != config["version"]:
        raise ValueError("executor package algorithm version does not match training config")
    if package_manifest.get("protocol_version") != config["protocol_version"]:
        raise ValueError("executor package protocol version does not match training config")
    expected_train_hash = variant_manifest.get("generated_file_sha256", {}).get("train.jsonl")
    if not expected_train_hash or _sha256(train_metadata) != expected_train_hash:
        raise ValueError("executor train metadata hash does not match the package manifest")
    expected_train_rows = 20_000
    if config.get("species") == "maize":
        from wgsr_v9.maize.experiment import EXECUTOR_PROTOCOL, PROTOCOL_ROOT
        if package_manifest.get("species") != "maize" or config["protocol_version"] != EXECUTOR_PROTOCOL:
            raise ValueError("maize training requires the reviewed maize data contract")
        expected_train_rows = json.loads((PROTOCOL_ROOT / "CONTRACT.json").read_text())["train_rows"]
    if variant_manifest.get("rows_by_split", {}).get("train") != expected_train_rows:
        raise ValueError(f"executor package must contain the reviewed {expected_train_rows:,}-row train split")

    py = str(python)
    accel = str(accelerate_bin)
    env = {
        "CUDA_VISIBLE_DEVICES": str(config["physical_gpu_ids"]),
        "OMP_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DIFFSYNTH_SKIP_DOWNLOAD": "True",
        "PYTORCH_CUDA_ALLOC_CONF": str(config.get("cuda_allocator_config", "expandable_segments:True")),
        "PYTHONPATH": str(diffsynth_repo),
    }
    cache_command = [
        py,
        str(CACHE_CHUNK_SCRIPT),
        "--python",
        py,
        "--accelerate",
        accel,
        "--accelerate-config",
        str(REPO_ROOT / "configs/accelerate_executor_cache_8gpu.yaml"),
        "--upstream-train",
        str(upstream_train),
        "--model-paths",
        json.dumps(upstream_cache),
        "--processor-path",
        str((model_root / "processor").resolve()),
        "--metadata",
        str(train_metadata),
        "--cache-root",
        str(cache_root),
        "--tmpfs-root",
        str(Path(str(config["cache_tmpfs_root"])).expanduser() / variant),
        "--tmpfs-ranks",
        *[str(rank) for rank in config.get("cache_tmpfs_ranks", [])],
        "--output-audit",
        str(compaction_audit),
        "--world-size",
        str(config["world_size"]),
        "--chunk-size",
        str(config.get("cache_chunk_size", 1024)),
        "--compact-workers",
        str(config.get("cache_compact_workers", 4)),
        "--max-pixels",
        str(config["max_pixels"]),
        "--main-process-port",
        str(29680 if variant == "simple" else 29681),
    ]
    index_command = [
        py,
        str(INDEX_SCRIPT),
        "--cache-root",
        str(cache_root),
        "--metadata",
        str(train_metadata),
        "--output",
        str(layout_index),
        "--world-size",
        str(config["world_size"]),
        "--variant",
        variant,
    ]
    train_command = [
        accel,
        "launch",
        "--config_file",
        str(REPO_ROOT / "configs/accelerate_executor_train_8gpu_bs2_ga1.yaml"),
        "--main_process_port",
        str(29682 if variant == "simple" else 29683),
        str(TRAIN_SCRIPT),
        "--diffsynth-repo",
        str(diffsynth_repo),
        "--variant",
        variant,
        "--model_paths",
        json.dumps(upstream_dit),
        "--processor_path",
        str((model_root / "processor").resolve()),
        "--dataset_base_path",
        str(cache_root),
        "--layout-index",
        str(layout_index),
        "--batch-size",
        str(config["per_device_train_batch_size"]),
        "--expected-world-size",
        str(config["world_size"]),
        "--expected-global-batch-size",
        str(config["effective_global_batch_size"]),
        "--gradient_accumulation_steps",
        str(config["gradient_accumulation_steps"]),
        "--learning_rate",
        str(config["learning_rate"]),
        "--weight_decay",
        str(config["weight_decay"]),
        "--num_epochs",
        str(config["num_train_epochs"]),
        "--seed",
        str(config["seed"]),
        "--task",
        "sft:train",
        "--trainable_models",
        "dit",
        "--use_gradient_checkpointing",
        "--find_unused_parameters",
        "--dataset_num_workers",
        str(config["dataset_num_workers"]),
        "--dataset_repeat",
        str(config["dataset_repeat"]),
        "--initialize_model_on_cpu",
        "--output_path",
        str(checkpoint_root),
    ]
    if lr_schedule.manual_step:
        train_command.extend(("--lr-scheduler", lr_schedule.name,
                              "--lr-warmup-ratio", str(lr_schedule.warmup_ratio),
                              "--lr-cosine-ratio", str(lr_schedule.cosine_ratio),
                              "--lr-warmup-start-factor", str(lr_schedule.warmup_start_factor),
                              "--lr-min-factor", str(lr_schedule.min_lr_factor)))
    if dropout is not None:
        train_command.extend(("--condition-dropout-cache", str(_path(dropout, "cache_root")),
                              "--condition-dropout-seed", str(dropout["seed"])))
    if ablation is not None:
        train_command.extend(("--condition-ablation-cache", str(_path(ablation, "cache_root")),
                              "--condition-ablation", ablation["mode"]))
    if config.get("max_optimizer_steps") is not None:
        train_command.extend(("--max-optimizer-steps", str(config["max_optimizer_steps"])))
    phases = {
        "cache": [cache_command, index_command],
        "train": [train_command],
        "all": [cache_command, index_command, train_command],
    }
    if phase not in phases:
        raise ValueError(f"unsupported phase: {phase}")
    return {
        "variant": variant,
        "phase": phase,
        "config_path": str(DEFAULT_CONFIG),
        "algorithm_version": str(config["version"]),
        "protocol_version": str(config["protocol_version"]),
        "world_size": int(config["world_size"]),
        "physical_gpu_ids": str(config["physical_gpu_ids"]),
        "per_device_train_batch_size": int(config["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "effective_global_batch_size": int(config["effective_global_batch_size"]),
        "image_dropout": 0.1 if dropout is not None else 0.0,
        "condition_dropout": dropout,
        "condition_ablation": ablation,
        "data_root": str(data_root),
        "numeric_condition": {"enabled": False},  # Retain text-run manifest compatibility.
        "lr_scheduler": lr_schedule.to_dict(),
        "cache_root": str(cache_root),
        "checkpoint_root": str(checkpoint_root),
        "log_root": str(variant_run_root / "logs"),
        "commands": [[str(part) for part in command] for command in phases[phase]],
        "environment": env,
        "cwd": str(diffsynth_repo),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _check_runtime(config: dict[str, Any], plan: dict[str, Any]) -> None:
    for key in ("python", "accelerate", "framework_repo", "model_path"):
        path = _path(config, key)
        if not path.exists():
            raise FileNotFoundError(f"configured {key} does not exist: {path}")
    if not _path(config, "python").is_file() or not _path(config, "accelerate").is_file():
        raise FileNotFoundError("configured Python or Accelerate executable is not a file")
    repo = _path(config, "framework_repo")
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != config["framework_commit"]:
        raise RuntimeError(f"DiffSynth commit mismatch: expected {config['framework_commit']}, found {commit}")
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
    if dirty:
        raise RuntimeError("DiffSynth source checkout has local modifications; refusing to train")
    model_root = _path(config, "model_path")
    _model_paths(model_root)
    selected_gpus = [token.strip() for token in str(config["physical_gpu_ids"]).split(",") if token.strip()]
    if len(selected_gpus) != int(config["world_size"]):
        raise ValueError("physical_gpu_ids must list exactly world_size GPU indices")
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    memory_by_index = {}
    for line in output.splitlines():
        index, used = line.split(",", maxsplit=1)
        memory_by_index[index.strip()] = int(used.strip())
    for gpu_id in selected_gpus:
        if gpu_id not in memory_by_index:
            raise RuntimeError(f"selected physical GPU is unavailable: {gpu_id}")
        if memory_by_index[gpu_id] >= 1000:
            raise RuntimeError(f"selected physical GPU {gpu_id} is already in use ({memory_by_index[gpu_id]} MiB)")
    if plan["phase"] in {"cache", "all"} and not _cache_is_complete(plan, config):
        data_root = _path(config, "data_root")
        data_root.mkdir(parents=True, exist_ok=True)
        free_gib = shutil.disk_usage(data_root).free / 1024**3
        if free_gib < 150:
            raise RuntimeError(f"need at least 150 GiB free for one variant cache; found {free_gib:.1f} GiB")


def _cache_is_complete(plan: dict[str, Any], config: dict[str, Any]) -> bool:
    package_root = Path(plan["data_root"]) / plan["variant"]
    metadata = package_root / "train.jsonl"
    index_path = package_root / "layout_index.json"
    compaction_path = package_root / "cache_compaction_audit.json"
    if not index_path.is_file() or not compaction_path.is_file():
        return False
    index = json.loads(index_path.read_text(encoding="utf-8"))
    compaction = json.loads(compaction_path.read_text(encoding="utf-8"))
    from sys import path as python_path

    python_path.insert(0, str(REPO_ROOT / "src"))
    from wgsr_v9.io import sha256_path

    return (
        index.get("status") == "complete"
        and compaction.get("status") == "complete"
        and index.get("metadata_sha256") == sha256_path(metadata)
        and index.get("executor_variant") == plan["variant"]
        and index.get("count", 0) > 0
        and len(index.get("items", [])) == index["count"]
        and all(Path(item["output_path"]).is_file()
                and Path(item["output_path"]).stat().st_size == item["bytes"]
                for item in index["items"])
    )


def _run_command(command: list[str], plan: dict[str, Any], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("COMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
        log.flush()
        subprocess.run(command, cwd=plan["cwd"], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def launch(plan: dict[str, Any], config: dict[str, Any], *, phase: str) -> dict[str, Any]:
    variant = plan["variant"]
    run_root = Path(plan["log_root"]).parent
    log_root = Path(plan["log_root"])
    data_variant = Path(plan["data_root"]) / variant
    run_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(plan["environment"])
    env["TMPDIR"] = str(run_root / "tmp")
    env["TORCH_EXTENSIONS_DIR"] = str(run_root / "torch_extensions")
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["TORCH_EXTENSIONS_DIR"]).mkdir(parents=True, exist_ok=True)

    run_manifest_path = run_root / "run_manifest.json"
    run_manifest: dict[str, Any] = {
        "algorithm_version": plan["algorithm_version"],
        "protocol_version": plan["protocol_version"],
        "executor_variant": variant,
        "input_path": str(data_variant / "train.jsonl"),
        "output_path": str(Path(plan["checkpoint_root"])),
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "framework": config["framework"],
        "framework_commit": config["framework_commit"],
        "model_path": config["model_path"],
        "world_size": config["world_size"],
        "physical_gpu_ids": config["physical_gpu_ids"],
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "effective_global_batch_size": config["effective_global_batch_size"],
        "num_train_epochs": config["num_train_epochs"],
        "learning_rate": config["learning_rate"],
        "lr_scheduler": plan["lr_scheduler"],
        "weight_decay": config["weight_decay"],
        "torch_dtype": config["torch_dtype"],
        "cuda_allocator_config": env["PYTORCH_CUDA_ALLOC_CONF"],
        "deepspeed_zero_stage": config["deepspeed_zero_stage"],
        "gradient_checkpointing": config["gradient_checkpointing"],
        "image_dropout": plan["image_dropout"],
        "condition_dropout": plan.get("condition_dropout"),
        "condition_ablation": plan.get("condition_ablation"),
        "validation_during_training": False,
        "fresh_base_model_initialization": True,
        "numeric_condition": plan["numeric_condition"],
        "config_path": plan["config_path"],
        "status": "running",
        "error": None,
        "retry_count": 0,
    }
    _write_json(run_manifest_path, run_manifest)

    try:
        commands = plan["commands"]
        if phase in {"cache", "all"}:
            if not data_variant.is_dir():
                raise FileNotFoundError(f"package directory does not exist: {data_variant}")
            cache_root = data_variant / "cache_train"
            if not _cache_is_complete(plan, config):
                if cache_root.exists() and not cache_root.is_dir():
                    raise FileExistsError(f"cache path is not a directory: {cache_root}")
                if cache_root.exists() and any(cache_root.iterdir()):
                    if not (cache_root / "_chunks").is_dir():
                        raise FileExistsError(
                            f"cache directory is nonempty but has no complete audit: {cache_root}; "
                            "use a new data package/run root"
                        )
                cache_root.mkdir(parents=True, exist_ok=True)
                _run_command(commands[0], plan, log_root / "cache.log", env)
                index_command = commands[1] if phase == "cache" else commands[1]
                _run_command(index_command, plan, log_root / "cache_index.log", env)
            else:
                run_manifest["cache_reused"] = True
        if phase in {"train", "all"}:
            if not _cache_is_complete(plan, config):
                raise RuntimeError("validated train cache is missing; run --phase cache first")
            checkpoint_root = Path(plan["checkpoint_root"])
            if (checkpoint_root / "completed.json").is_file():
                complete = json.loads((checkpoint_root / "completed.json").read_text(encoding="utf-8"))
                if complete.get("status") == "complete":
                    run_manifest["status"] = "already_complete"
                    run_manifest["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
                    _write_json(run_manifest_path, run_manifest)
                    return run_manifest
            if checkpoint_root.exists() and any(checkpoint_root.iterdir()):
                raise FileExistsError(
                    f"checkpoint directory is nonempty and has no completion marker: {checkpoint_root}; "
                    "use a new run_root to start a fresh run"
                )
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            train_command = commands[0] if phase == "train" else commands[2]
            _run_command(train_command, plan, log_root / "train.log", env)
        run_manifest["status"] = "complete" if phase == "all" else f"{phase}_complete"
        run_manifest["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _write_json(run_manifest_path, run_manifest)
        return run_manifest
    except Exception as exc:
        run_manifest["status"] = "failed"
        run_manifest["error"] = f"{type(exc).__name__}: {exc}"
        run_manifest["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _write_json(run_manifest_path, run_manifest)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variant", choices=("simple", "teacher_cot"), required=True)
    parser.add_argument("--phase", choices=("cache", "train", "all"), default="all")
    parser.add_argument("--launch", action="store_true", help="Run the printed DiffSynth commands.")
    parser.add_argument("--print-json", action="store_true", help="Emit the plan as machine-readable JSON.")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    plan = build_plan(config, args.variant, args.phase)
    plan["config_path"] = str(args.config.resolve())
    if args.print_json:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    else:
        print(f"{args.phase} plan for {args.variant}; image_dropout={plan['image_dropout']}")
        for index, command in enumerate(plan["commands"], start=1):
            print(f"{index}. " + json.dumps(command, ensure_ascii=False))
        print(f"cache/data: {plan['cache_root']}")
        print(f"checkpoints: {plan['checkpoint_root']}")
        print(f"logs: {plan['log_root']}")
        if not args.launch:
            print("Preview only. Add --launch to execute these commands.")
    if not args.launch:
        return 0
    _check_runtime(config, plan)
    result = launch(plan, config, phase=args.phase)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
