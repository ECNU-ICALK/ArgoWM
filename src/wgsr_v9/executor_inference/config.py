"""Configuration and path resolution for executor inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .constants import ALGORITHM_VERSION, DEFAULT_CONFIG, PROTOCOL_VERSION, REPO_ROOT


def resolve_path(value: str | Path, *, base: Path = REPO_ROOT) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"inference config must contain a mapping: {path}")
    validate_config(config, path=path)
    return config


def validate_config(config: dict[str, Any], *, path: Path | None = None) -> None:
    if (any(config.get(key) is not None for key in (
        "numeric_bundle", "numeric_mode", "numeric_protocol_root",
        "numeric_normalizer_sha256", "numeric_training_metadata_sha256",
    )) or config.get("numeric_condition", {}).get("enabled", False)
            or config.get("condition_dropout") == "drop_numeric"):
        raise ValueError("Numeric injection is not included in this public release; use a text Executor recipe")
    location = f" in {path}" if path else ""
    if config.get("version") != ALGORITHM_VERSION:
        raise ValueError(
            f"executor inference config version{location} must be {ALGORITHM_VERSION!r}"
        )
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(
            f"executor inference protocol_version{location} must be {PROTOCOL_VERSION!r}"
        )
    if config.get("source_only") is not True:
        raise ValueError("executor inference must be source_only=true")
    if config.get("target_usage") != "eval_reference_only":
        raise ValueError("executor inference target_usage must be eval_reference_only")
    if int(config.get("num_inference_steps", -1)) != 20:
        raise ValueError("the reviewed executor inference contract requires 20 steps")
    gpu_ids = [
        token.strip()
        for token in str(config.get("physical_gpu_ids", "")).split(",")
        if token.strip()
    ]
    if int(config.get("world_size", -1)) != len(gpu_ids) or not gpu_ids:
        raise ValueError("world_size must equal the number of configured physical_gpu_ids")


def default_checkpoint(config: dict[str, Any], variant: str) -> Path:
    checkpoint_variant = "teacher_cot" if variant == "predicted_cot" else variant
    explicit = config.get("checkpoints", {}).get(checkpoint_variant)
    if explicit:
        return resolve_path(explicit)
    return (
        resolve_path(config["executor_training_root"])
        / checkpoint_variant
        / "checkpoints/epoch-1.safetensors"
    )


def default_input_manifest(config: dict[str, Any], variant: str) -> Path:
    if variant == "predicted_cot":
        manifest = config.get("predicted_cot_manifest")
        if not manifest:
            raise ValueError("predicted_cot requires an explicit generated CoT input manifest")
        return resolve_path(manifest)
    return resolve_path(config["executor_data_root"]) / variant / "val.jsonl"


def default_output_root(config: dict[str, Any], variant: str) -> Path:
    return resolve_path(config["inference_run_root"]) / variant
