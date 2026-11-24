#!/usr/bin/env python3
"""Compact generated DiffSynth caches to exact BF16 training inputs."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

from wgsr_v9.executor_training import ALGORITHM_VERSION  # noqa: E402
from wgsr_v9.io import write_json  # noqa: E402

SHARED_KEYS = {
    "input_latents",
    "edit_latents",
    "height",
    "width",
    "cfg_scale",
    "use_gradient_checkpointing",
    "use_gradient_checkpointing_offload",
    "use_flex_attention",
    "min_timestep_boundary",
    "max_timestep_boundary",
    "noise_scale",
}
POSITIVE_KEYS = {"prompt", "prompt_embeds", "prompt_embeds_mask", "edit_image_pad_mask", "kv_cache"}


def _compact_inputs(inputs: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    shared, positive, _negative = inputs
    compact_shared = {key: value for key, value in shared.items() if key in SHARED_KEYS}
    compact_shared["input_latents"] = shared["input_latents"].to(torch.bfloat16).contiguous()
    compact_shared["edit_latents"] = [value.to(torch.bfloat16).contiguous() for value in shared["edit_latents"]]
    compact_positive = {key: positive[key] for key in POSITIVE_KEYS if key in positive}
    compact_positive["prompt_embeds"] = positive["prompt_embeds"].to(torch.bfloat16).contiguous()
    return compact_shared, compact_positive, {"negative_prompt": ""}


def _validate_same_effective_values(
    original: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    compact: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    path: Path,
) -> None:
    tensor_pairs = [
        (original[0]["input_latents"], compact[0]["input_latents"]),
        (original[1]["prompt_embeds"], compact[1]["prompt_embeds"]),
    ]
    tensor_pairs.extend(zip(original[0]["edit_latents"], compact[0]["edit_latents"]))
    for before, after in tensor_pairs:
        if not torch.equal(before.to(torch.bfloat16), after):
            raise ValueError(f"effective BF16 training values changed: {path}")
        if not bool(torch.isfinite(after).all()):
            raise ValueError(f"nonfinite compact training tensor: {path}")
    if not torch.equal(original[1]["edit_image_pad_mask"], compact[1]["edit_image_pad_mask"]):
        raise ValueError(f"source image-token mask changed: {path}")
    if original[1]["prompt"] != compact[1]["prompt"]:
        raise ValueError(f"prompt changed: {path}")


def _is_compact(inputs: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]) -> bool:
    shared, positive, negative = inputs
    tensors = [shared["input_latents"], positive["prompt_embeds"], *shared["edit_latents"]]
    return (
        set(shared).issubset(SHARED_KEYS)
        and set(positive).issubset(POSITIVE_KEYS)
        and set(negative).issubset({"negative_prompt"})
        and all(tensor.dtype == torch.bfloat16 for tensor in tensors)
    )


def compact_one(path_string: str) -> dict[str, Any]:
    torch.set_num_threads(1)
    path = Path(path_string).resolve()
    before_bytes = path.stat().st_size
    original = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    compact = _compact_inputs(original)
    _validate_same_effective_values(original, compact, path)
    if _is_compact(original):
        return {
            "algorithm_version": ALGORITHM_VERSION,
            "input_path": str(path),
            "output_path": str(path),
            "before_bytes": before_bytes,
            "after_bytes": before_bytes,
            "effective_training_tensors_exact": True,
            "status": "ready",
            "error": None,
            "retry_count": 0,
        }
    temporary = path.with_suffix(".compact.tmp")
    torch.save(compact, temporary)
    reloaded = torch.load(temporary, map_location="cpu", weights_only=False, mmap=True)
    _validate_same_effective_values(original, reloaded, path)
    after_bytes = temporary.stat().st_size
    if after_bytes >= before_bytes:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"compacted cache is not smaller ({before_bytes} -> {after_bytes}): {path}")
    temporary.replace(path)
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "input_path": str(path),
        "output_path": str(path),
        "before_bytes": before_bytes,
        "after_bytes": after_bytes,
        "effective_training_tensors_exact": True,
        "status": "ready",
        "error": None,
        "retry_count": 0,
    }


def compact_cache(cache_root: Path, output_path: Path, workers: int) -> dict[str, Any]:
    cache_root = cache_root.resolve()
    output_path = output_path.resolve()
    if workers <= 0:
        raise ValueError("workers must be positive")
    paths = sorted(cache_root.rglob("*.pth"))
    if not paths:
        raise ValueError(f"no .pth cache files found under {cache_root}")
    if any(not path.resolve().is_relative_to(cache_root) for path in paths):
        raise ValueError("cache file escaped cache root")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(compact_one, str(path)): path for path in paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:  # write a row-level audit before stopping
                failures.append(
                    {
                        "algorithm_version": ALGORITHM_VERSION,
                        "input_path": str(path.resolve()),
                        "output_path": str(path.resolve()),
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "retry_count": 0,
                    }
                )
    rows.sort(key=lambda row: row["input_path"])
    failures.sort(key=lambda row: row["input_path"])
    report = {
        "algorithm_version": ALGORITHM_VERSION,
        "input_path": str(cache_root),
        "output_path": str(output_path),
        "count": len(paths),
        "completed_count": len(rows),
        "failed_count": len(failures),
        "before_bytes": sum(row["before_bytes"] for row in rows),
        "after_bytes": sum(row["after_bytes"] for row in rows),
        "all_effective_training_tensors_exact": all(row["effective_training_tensors_exact"] for row in rows),
        "files": rows + failures,
        "status": "complete" if not failures else "failed",
        "error": None if not failures else f"{len(failures)} cache compaction failures",
        "retry_count": 0,
    }
    write_json(output_path, report)
    if failures:
        raise RuntimeError(report["error"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    report = compact_cache(args.cache_root, args.output, args.workers)
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
