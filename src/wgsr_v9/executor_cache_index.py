#!/usr/bin/env python3
"""Validate DiffSynth train caches and write the homogeneous-layout index."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

from wgsr_v9.diffsynth_batching import batch_layout_key, layout_key  # noqa: E402
from wgsr_v9.io import read_jsonl, sha256_path, write_json  # noqa: E402
from wgsr_v9.executor_training import ALGORITHM_VERSION  # noqa: E402


def expected_latent_shape(row: dict[str, Any]) -> tuple[int, int, int, int]:
    """Validate species geometry before accepting a cached training example."""
    if row.get("species") == "maize":
        from wgsr_v9.maize.experiment import EXECUTOR_PROTOCOL

        if (row.get("protocol_version") != EXECUTOR_PROTOCOL
                or row.get("source_size") != [768, 1280]
                or row.get("view") != "side0"):
            raise ValueError("maize cache requires the reviewed native 768x1280 side0 contract")
        return (1, 64, 80, 48)
    return (1, 64, 56, 42) if row["view"] in {"front", "side0"} else (1, 64, 42, 56)


def _cache_files(cache_root: Path, world_size: int) -> list[tuple[int, int, Path]]:
    records: list[tuple[int, int, Path]] = []
    rank_dirs = [path for path in cache_root.iterdir() if path.is_dir() and path.name.isdigit()]
    ranks = sorted(int(path.name) for path in rank_dirs)
    if ranks != list(range(world_size)):
        raise ValueError(f"cache ranks must be 0..{world_size - 1}, found {ranks}")
    for rank_dir in sorted(rank_dirs, key=lambda path: int(path.name)):
        rank = int(rank_dir.name)
        files = sorted(rank_dir.glob("*.pth"), key=lambda path: int(path.stem))
        if any(not path.stem.isdigit() for path in files):
            raise ValueError(f"cache contains nonnumeric pth filename under {rank_dir}")
        for path in files:
            records.append((rank, int(path.stem), path))
    return records


def build_index(
    cache_root: Path,
    metadata_path: Path,
    output_path: Path,
    *,
    world_size: int,
    variant: str,
) -> dict[str, Any]:
    cache_root = cache_root.resolve()
    metadata_path = metadata_path.resolve()
    output_path = output_path.resolve()
    rows = read_jsonl(metadata_path)
    if not rows:
        raise ValueError("training metadata is empty")
    records = _cache_files(cache_root, world_size)
    if len(records) != len(rows):
        raise ValueError(f"cache count {len(records)} does not match metadata count {len(rows)}")

    items: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for rank, local_index, path in records:
        original_index = local_index * world_size + rank
        if original_index >= len(rows) or original_index in seen_indices:
            raise ValueError(f"invalid distributed cache order at rank={rank}, local_index={local_index}")
        row = rows[original_index]
        inputs = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        shared, positive, _negative = inputs
        if shared.get("use_gradient_checkpointing") is not True:
            raise ValueError(f"gradient checkpointing is not enabled in {path}")
        if len(shared.get("edit_latents", [])) != 1:
            raise ValueError(f"expected exactly one source latent in {path}")
        if positive.get("prompt") != row["prompt"]:
            raise ValueError(f"cached prompt does not match metadata row {original_index}: {path}")
        expected_latent = expected_latent_shape(row)
        target_latent = shared["input_latents"]
        source_latent = shared["edit_latents"][0]
        if tuple(target_latent.shape) != expected_latent or tuple(source_latent.shape) != expected_latent:
            raise ValueError(
                f"unexpected target/source latent layout for {row['request_id']}: "
                f"{tuple(target_latent.shape)} / {tuple(source_latent.shape)}"
            )
        for tensor in (target_latent, source_latent, positive["prompt_embeds"]):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"nonfinite cached training tensor: {path}")
        key = layout_key(inputs)
        batch_key = batch_layout_key(inputs)
        items.append(
            {
                "algorithm_version": ALGORITHM_VERSION,
                "request_id": row["request_id"],
                "original_row_index": original_index,
                "rank": rank,
                "rank_local_index": local_index,
                "layout_key": repr(key),
                "batch_layout_key": repr(batch_key),
                "input_path": row["edit_image"],
                "output_path": str(path.resolve()),
                "status": "ready",
                "error": None,
                "retry_count": 0,
                "bytes": path.stat().st_size,
            }
        )
        seen_indices.add(original_index)
        del inputs, shared, positive
    if seen_indices != set(range(len(rows))):
        raise ValueError("distributed cache does not cover every metadata row exactly once")
    items.sort(key=lambda item: item["original_row_index"])
    report = {
        "algorithm_version": ALGORITHM_VERSION,
        "executor_variant": variant,
        "input_path": str(metadata_path),
        "output_path": str(output_path),
        "cache_root": str(cache_root),
        "metadata_sha256": sha256_path(metadata_path),
        "world_size": world_size,
        "count": len(items),
        "all_condition_and_target_tensors_finite": True,
        "all_rows_order_and_prompt_checked": True,
        "layout_counts": dict(Counter(item["layout_key"] for item in items)),
        "batch_layout_counts": dict(Counter(item["batch_layout_key"] for item in items)),
        "total_bytes": sum(item["bytes"] for item in items),
        "items": items,
        "status": "complete",
        "error": None,
        "retry_count": 0,
    }
    write_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--variant", choices=("simple", "teacher_cot"), required=True)
    args = parser.parse_args()
    report = build_index(
        args.cache_root,
        args.metadata,
        args.output,
        world_size=args.world_size,
        variant=args.variant,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "items"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
