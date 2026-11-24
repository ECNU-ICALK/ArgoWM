#!/usr/bin/env python3
"""Build a validated DiffSynth cache in bounded, compacted chunks.

The upstream Qwen Image 2.1 data processor writes large per-sample pickle
files.  Running it over the full 20,000-row package can exceed local disk
capacity before compaction starts.  This entry point processes contiguous
chunks, compacts each chunk in place, and moves the compact files into the
final distributed cache layout.
"""

from __future__ import annotations

import argparse
import errno
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

from wgsr_v9.executor_training import ALGORITHM_VERSION  # noqa: E402
from wgsr_v9.io import read_jsonl, sha256_path, write_json, write_jsonl  # noqa: E402

COMPACT_SCRIPT = REPO_ROOT / "scripts/compact_executor_caches.py"


def _move_cache_file(source: Path, target: Path) -> None:
    """Move a compact cache file, including across the workspace/tmpfs boundary.

    ``Path.replace`` is atomic only when both paths are on the same filesystem.
    The reviewed layout intentionally puts ranks 0--3 on ``/dev/shm`` and the
    remaining ranks on the workspace filesystem, so those files need a
    copy-then-remove fallback when a chunk crosses devices.
    """

    try:
        source.replace(target)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copy2(source, target)
        if target.stat().st_size != source.stat().st_size:
            target.unlink(missing_ok=True)
            raise OSError(f"cross-device cache copy was truncated: {source} -> {target}")
        source.unlink()


def _chunk_ranges(count: int, chunk_size: int, world_size: int) -> list[tuple[int, int]]:
    if count <= 0:
        raise ValueError("metadata is empty")
    if chunk_size < world_size or chunk_size % world_size:
        raise ValueError("chunk size must be a multiple of world size and at least world size")
    ranges: list[tuple[int, int]] = []
    for start in range(0, count, chunk_size):
        end = min(count, start + chunk_size)
        if start % world_size or end % world_size:
            raise ValueError("all chunk boundaries must align to distributed world size")
        ranges.append((start, end))
    return ranges


def _rank_files(root: Path, world_size: int) -> list[tuple[int, int, Path]]:
    rank_dirs = sorted((path for path in root.iterdir() if path.is_dir() and path.name.isdigit()), key=lambda p: int(p.name))
    ranks = [int(path.name) for path in rank_dirs]
    if ranks != list(range(world_size)):
        raise ValueError(f"chunk cache ranks must be 0..{world_size - 1}, found {ranks}")
    records: list[tuple[int, int, Path]] = []
    for rank_dir in rank_dirs:
        for path in sorted(rank_dir.glob("*.pth"), key=lambda p: int(p.stem)):
            if not path.stem.isdigit():
                raise ValueError(f"chunk cache has nonnumeric file: {path}")
            records.append((int(rank_dir.name), int(path.stem), path))
    return records


def _run_cache_chunk(
    *,
    accelerate: Path,
    accelerate_config: Path,
    upstream_train: Path,
    model_paths: str,
    processor_path: Path,
    metadata_path: Path,
    output_path: Path,
    max_pixels: int,
    main_process_port: int,
    environment: dict[str, str] | None,
) -> None:
    command = [
        str(accelerate),
        "launch",
        "--config_file",
        str(accelerate_config),
        "--main_process_port",
        str(main_process_port),
        str(upstream_train),
        "--task",
        "sft:data_process",
        "--trainable_models",
        "dit",
        "--model_paths",
        model_paths,
        "--processor_path",
        str(processor_path),
        "--dataset_base_path",
        "/",
        "--dataset_metadata_path",
        str(metadata_path),
        "--data_file_keys",
        "image,edit_image",
        "--extra_inputs",
        "edit_image",
        "--max_pixels",
        str(max_pixels),
        "--dataset_repeat",
        "1",
        "--dataset_num_workers",
        "0",
        "--use_gradient_checkpointing",
        "--output_path",
        str(output_path),
    ]
    subprocess.run(command, cwd=upstream_train.parents[3], env=environment, check=True)


def _compact(python: Path, staging_root: Path, audit_path: Path, workers: int) -> dict[str, Any]:
    command = [
        str(python),
        str(COMPACT_SCRIPT),
        "--cache-root",
        str(staging_root),
        "--output",
        str(audit_path),
        "--workers",
        str(workers),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return json.loads(audit_path.read_text(encoding="utf-8"))


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    metadata_path = args.metadata.resolve()
    cache_root = args.cache_root.resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    rows = read_jsonl(metadata_path)
    ranges = _chunk_ranges(len(rows), args.chunk_size, args.world_size)
    cache_root.mkdir(parents=True, exist_ok=True)
    chunks_root = cache_root / "_chunks"
    chunks_root.mkdir(parents=True, exist_ok=True)
    tmpfs_root = args.tmpfs_root.resolve() if args.tmpfs_root is not None else None
    tmpfs_ranks = set(args.tmpfs_ranks)
    if tmpfs_ranks and tmpfs_root is None:
        raise ValueError("--tmpfs-root is required when --tmpfs-ranks is used")
    if any(rank < 0 or rank >= args.world_size for rank in tmpfs_ranks):
        raise ValueError("tmpfs rank is outside world size")
    if tmpfs_root is not None:
        tmpfs_root.mkdir(parents=True, exist_ok=True)
    final_rank_dirs: list[Path] = []
    for rank in range(args.world_size):
        if rank in tmpfs_ranks:
            final_dir = tmpfs_root / str(rank)
            link = cache_root / str(rank)
            final_dir.mkdir(parents=True, exist_ok=True)
            if link.is_symlink():
                if link.resolve() != final_dir.resolve():
                    raise FileExistsError(f"cache rank symlink points elsewhere: {link}")
            elif link.exists():
                if any(link.iterdir()):
                    raise FileExistsError(f"cache rank path is nonempty and not a symlink: {link}")
                link.rmdir()
                link.symlink_to(final_dir, target_is_directory=True)
            else:
                link.symlink_to(final_dir, target_is_directory=True)
        else:
            final_dir = cache_root / str(rank)
            final_dir.mkdir(parents=True, exist_ok=True)
        final_rank_dirs.append(final_dir)

    all_files: list[dict[str, Any]] = []
    chunk_reports: list[dict[str, Any]] = []
    for chunk_number, (start, end) in enumerate(ranges):
        chunk_tag = f"{start:06d}_{end:06d}"
        chunk_manifest_path = chunks_root / f"chunk_{chunk_tag}.json"
        staging_root = chunks_root / f"staging_{chunk_tag}"
        chunk_metadata = chunks_root / f"metadata_{chunk_tag}.jsonl"
        compaction_audit = chunks_root / f"compaction_{chunk_tag}.json"
        if chunk_manifest_path.is_file():
            report = json.loads(chunk_manifest_path.read_text(encoding="utf-8"))
            if report.get("status") != "complete" or report.get("metadata_sha256") != sha256_path(metadata_path):
                raise ValueError(f"chunk manifest is incomplete or stale: {chunk_manifest_path}")
            chunk_reports.append(report)
            all_files.extend(report.get("files", []))
            continue
        if staging_root.exists() and any(staging_root.iterdir()):
            raise FileExistsError(f"incomplete staging directory requires a new cache root: {staging_root}")
        if chunk_metadata.exists() or compaction_audit.exists():
            raise FileExistsError(f"stale chunk intermediates require a new cache root: {chunks_root}")
        staging_root.mkdir(parents=True, exist_ok=True)
        write_jsonl(chunk_metadata, rows[start:end])
        _run_cache_chunk(
            accelerate=args.accelerate,
            accelerate_config=args.accelerate_config,
            upstream_train=args.upstream_train,
            model_paths=args.model_paths,
            processor_path=args.processor_path,
            metadata_path=chunk_metadata,
            output_path=staging_root,
            max_pixels=args.max_pixels,
            main_process_port=args.main_process_port,
            environment=None,
        )
        records = _rank_files(staging_root, args.world_size)
        expected_count = end - start
        if len(records) != expected_count:
            raise ValueError(f"chunk {chunk_tag} produced {len(records)} files; expected {expected_count}")
        compact_report = _compact(args.python, staging_root, compaction_audit, args.compact_workers)
        if compact_report.get("status") != "complete":
            raise RuntimeError(f"chunk compaction failed: {compaction_audit}")
        offset = start // args.world_size
        moved_files: list[dict[str, Any]] = []
        for rank, local_index, source_path in records:
            target_path = final_rank_dirs[rank] / f"{offset + local_index}.pth"
            if target_path.exists():
                raise FileExistsError(f"final cache file already exists: {target_path}")
            _move_cache_file(source_path, target_path)
            moved_files.append(
                {
                    "algorithm_version": ALGORITHM_VERSION,
                    "input_path": str((metadata_path).resolve()),
                    "output_path": str(target_path.resolve()),
                    "rank": rank,
                    "rank_local_index": offset + local_index,
                    "chunk_start": start,
                    "chunk_end": end,
                    "status": "ready",
                    "error": None,
                    "retry_count": 0,
                    "bytes": target_path.stat().st_size,
                }
            )
        for rank_dir in sorted((path for path in staging_root.iterdir() if path.is_dir()), key=lambda p: p.name):
            rank_dir.rmdir()
        staging_root.rmdir()
        report = {
            "algorithm_version": ALGORITHM_VERSION,
            "input_path": str(metadata_path),
            "output_path": str(cache_root),
            "metadata_sha256": sha256_path(metadata_path),
            "chunk_start": start,
            "chunk_end": end,
            "count": expected_count,
            "files": moved_files,
            "status": "complete",
            "error": None,
            "retry_count": 0,
        }
        write_json(chunk_manifest_path, report)
        chunk_reports.append(report)
        all_files.extend(moved_files)

    all_files.sort(key=lambda item: (int(item["rank"]), int(item["rank_local_index"])))
    if len(all_files) != len(rows):
        raise ValueError(f"final cache file count {len(all_files)} does not match metadata count {len(rows)}")
    final_audit = {
        "algorithm_version": ALGORITHM_VERSION,
        "input_path": str(metadata_path),
        "output_path": str(args.output_audit.resolve()),
        "cache_root": str(cache_root),
        "metadata_sha256": sha256_path(metadata_path),
        "world_size": args.world_size,
        "count": len(all_files),
        "completed_count": len(all_files),
        "failed_count": 0,
        "before_bytes": None,
        "after_bytes": sum(int(item["bytes"]) for item in all_files),
        "all_effective_training_tensors_exact": True,
        "chunk_count": len(chunk_reports),
        "files": all_files,
        "status": "complete",
        "error": None,
        "retry_count": 0,
    }
    write_json(args.output_audit, final_audit)
    return {key: value for key, value in final_audit.items() if key != "files"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--accelerate", type=Path, required=True)
    parser.add_argument("--accelerate-config", type=Path, required=True)
    parser.add_argument("--upstream-train", type=Path, required=True)
    parser.add_argument("--model-paths", required=True)
    parser.add_argument("--processor-path", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--tmpfs-root", type=Path, default=None)
    parser.add_argument("--tmpfs-ranks", type=int, nargs="*", default=[])
    parser.add_argument("--output-audit", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--compact-workers", type=int, default=4)
    parser.add_argument("--max-pixels", type=int, required=True)
    parser.add_argument("--main-process-port", type=int, default=29680)
    args = parser.parse_args()
    report = build_cache(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
