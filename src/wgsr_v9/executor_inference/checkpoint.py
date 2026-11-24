"""Diffusers-compatible view over a DiffSynth transformer checkpoint."""

from __future__ import annotations

import os
import json
from pathlib import Path
import shutil
from typing import Any

from ..io import sha256_path


def pretrained_transformer_identity(model_path: Path) -> dict[str, Any]:
    """Fingerprint the actual original shards, not just their index file."""
    root = model_path / "transformer"
    index = root / "diffusion_pytorch_model.safetensors.index.json"
    files = {root / "config.json"}
    if index.is_file():
        files.add(index)
        weights = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        if not weights:
            raise ValueError("pretrained transformer index has no weights")
        for name in set(weights.values()):
            if Path(name).name != name:
                raise ValueError("pretrained transformer shard must be a local filename")
            files.add(root / name)
    else:
        files.add(root / "diffusion_pytorch_model.safetensors")
    return {
        "algorithm_version": "wgsr_v9_pretrained_transformer_identity_v1",
        "path": str(root.resolve()),
        "files_sha256": {path.name: sha256_path(path) for path in sorted(files)},
    }


def make_transformer_view(model_path: Path, checkpoint: Path | None, output_root: Path) -> Path:
    if checkpoint is None:
        root = model_path / "transformer"
        if not (root / "config.json").is_file():
            raise FileNotFoundError(root / "config.json")
        if not any((root / name).is_file() for name in (
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.safetensors.index.json",
        )):
            raise FileNotFoundError(f"no pretrained transformer weights in {root}")
        return root
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    base_config = model_path / "transformer/config.json"
    if not base_config.is_file():
        raise FileNotFoundError(base_config)
    view = output_root / "_trained_transformer_view"
    view.mkdir(parents=True, exist_ok=True)
    config = view / "config.json"
    if config.exists():
        if sha256_path(config) != sha256_path(base_config):
            raise ValueError(f"existing transformer view config differs: {config}")
    else:
        temporary = config.with_name(f".{config.name}.{os.getpid()}.tmp")
        shutil.copy2(base_config, temporary)
        try:
            temporary.replace(config)
        except FileExistsError:
            temporary.unlink(missing_ok=True)
    link = view / "diffusion_pytorch_model.safetensors"
    expected = checkpoint.resolve()
    if link.is_symlink():
        if link.resolve() != expected:
            raise ValueError(f"transformer view points to a different checkpoint: {link.resolve()}")
    elif link.exists():
        raise ValueError(f"refusing to replace non-symlink transformer view file: {link}")
    else:
        try:
            link.symlink_to(expected)
        except FileExistsError:
            if not link.is_symlink() or link.resolve() != expected:
                raise ValueError(f"transformer view points to a different checkpoint: {link}")
    return view


def load_pipeline(model_path: Path, checkpoint: Path | None, view: Path, dtype_name: str):
    import torch
    from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    print(f"loading transformer checkpoint={checkpoint} pretrained={checkpoint is None} view={view}", flush=True)
    transformer, loading = QwenImage21Transformer2DModel.from_pretrained(
        str(view), torch_dtype=dtype, output_loading_info=True, local_files_only=True
    )
    unexpected = {key: value for key, value in loading.items() if value}
    if unexpected:
        raise RuntimeError(f"checkpoint loading reported non-empty fields: {unexpected}")
    pipe = QwenImage21Pipeline.from_pretrained(
        str(model_path), transformer=transformer, torch_dtype=dtype, local_files_only=True
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe, loading, dtype
