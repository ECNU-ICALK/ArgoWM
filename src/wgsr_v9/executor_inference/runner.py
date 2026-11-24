"""Worker and supervisor implementations for executor inference."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable

from ..io import append_jsonl, canonical_sha256, read_jsonl, sha256_path, write_json
from .checkpoint import load_pipeline, make_transformer_view, pretrained_transformer_identity
from .constants import ALGORITHM_VERSION, CLI_SCRIPT, EXPECTED_SIZES, PROTOCOL_VERSION
from .manifest import (
    input_hash,
    load_input_rows,
    load_seed_map,
    parse_indices,
    read_optional_jsonl,
    row_seed,
    validate_input_contract,
)

sha256_file = sha256_path


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def select_gpu_ids(value: str) -> list[str]:
    ids = [token.strip() for token in value.split(",") if token.strip()]
    if not ids or any(not token.isdigit() for token in ids):
        raise ValueError(f"invalid GPU list: {value!r}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate GPU in list: {value!r}")
    return ids


def row_source_path(row: dict[str, Any]) -> Path:
    return Path(str(row["edit_image"])).expanduser().resolve()


def expected_source_size(row: dict[str, Any], source: Any) -> tuple[int, int]:
    width, height = map(int, source.size)
    allowed_sizes = EXPECTED_SIZES
    if row.get("species") == "maize":
        from ..maize.experiment import EXECUTOR_PROTOCOL, SIZE
        if row.get("executor_prompt_protocol_version", row.get("protocol_version")) != EXECUTOR_PROTOCOL:
            raise ValueError("maize source geometry requires its versioned executor protocol")
        allowed_sizes = {SIZE}
    if (width, height) not in allowed_sizes:
        raise ValueError(f"unsupported source size for {row['request_id']}: {(width, height)}")
    declared = row.get("source_size")
    if declared is not None and [width, height] != [int(declared[0]), int(declared[1])]:
        raise ValueError(
            f"source size differs from manifest for {row['request_id']}: "
            f"{(width, height)} vs {declared}"
        )
    return width, height


def existing_successes(ledger: Path) -> dict[int, dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    for record in read_optional_jsonl(ledger):
        if record.get("status") != "completed":
            continue
        index = int(record["row_index"])
        output = Path(str(record.get("output_path", "")))
        if output.is_file():
            latest[index] = record
    return latest


def fixed_ablation_inputs(row: dict[str, Any], mode: str) -> tuple[str, tuple[int, int] | None]:
    from ..executor_ablation import transform_prompt
    prompt = transform_prompt(str(row["prompt"]), mode)
    if mode != "no_image":
        return prompt, None
    declared = row.get("source_size")
    if not isinstance(declared, (list, tuple)) or len(declared) != 2:
        raise ValueError("no_image requires frozen canvas dimensions; source pixels must not be read")
    size = tuple(int(x) for x in declared)
    if size not in EXPECTED_SIZES:
        raise ValueError("unsupported no_image canvas size")
    return prompt, size


def run_identity(args: argparse.Namespace, rows: list[dict[str, Any]],
                 input_sha: str, seed_sha: str | None) -> dict[str, Any]:
    return {
        "variant": args.variant, "input_manifest_sha256": input_sha,
        "checkpoint_sha256": sha256_file(args.checkpoint) if args.checkpoint is not None else None,
        **({"pretrained_transformer": pretrained_transformer_identity(args.model_path)}
           if args.checkpoint is None else {}),
        "model_path": str(args.model_path.resolve()),
        "selected_indices": [int(row["inference_row_index"]) for row in rows],
        "steps": args.steps, "dtype": args.dtype, "use_kv_cache": bool(args.use_kv_cache),
        "seed_base": args.seed_base, "seed_manifest_sha256": seed_sha,
        "numeric_condition": None,  # Keep pre-cleanup text-run identity hashes resumable.
        "executor_prompt_protocol_version": getattr(args, "executor_prompt_protocol_version", None),
        **({"condition_dropout": args.condition_dropout}
           if getattr(args, "condition_dropout", "keep") != "keep" else {}),
        **({"condition_ablation": args.condition_ablation,
            "condition_ablation_version": "wgsr_v9_executor_input_ablation_v1"}
           if getattr(args, "condition_ablation", "none") != "none" else {}),
    }


def validate_resume(previous: dict[str, Any], identity: dict[str, Any]) -> None:
    # Check identity before the completed-run shortcut, including sampling and
    # selection. Older manifests must be regenerated in a new output directory.
    if previous.get("run_identity") != identity:
        raise RuntimeError("inference inputs/checkpoint/sampling changed or legacy identity is missing; use a new output root")


def worker_main(args: argparse.Namespace) -> int:
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for executor inference")
    torch.cuda.set_device(0)
    rows, input_sha = load_input_rows(args.input_manifest, parse_indices(args.row_indices))
    validate_input_contract(args, rows)
    seed_map, seed_sha = load_seed_map(args.seed_manifest)
    args.run_identity = run_identity(args, rows, input_sha, seed_sha)
    args.run_identity_sha256 = canonical_sha256(args.run_identity)
    view = make_transformer_view(args.model_path, args.checkpoint, args.output_root)
    pipe, loading, dtype = load_pipeline(args.model_path, args.checkpoint, view, args.dtype)
    rank_dir = args.output_root / f"rank-{args.rank:02d}"
    ledger = rank_dir / "results.jsonl"
    images = args.output_root / "images"
    images.mkdir(parents=True, exist_ok=True)
    completed = existing_successes(ledger)
    assigned = rows[args.rank :: args.world_size]
    print(
        f"rank={args.rank}/{args.world_size} assigned={len(assigned)} "
        f"resume_completed={len(completed)}",
        flush=True,
    )
    generated = 0
    failed = 0
    for row in assigned:
        index = int(row["inference_row_index"])
        row_hash = input_hash(row)
        output_path = images / f"sample-{index:04d}.png"
        old = completed.get(index)
        if (not args.overwrite and old is not None and old.get("input_sha256") == row_hash
                and old.get("run_identity_sha256") == args.run_identity_sha256 and output_path.is_file()):
            print(f"rank={args.rank} skip row={index}", flush=True)
            continue
        source_path = row_source_path(row)
        base: dict[str, Any] = {
            "algorithm_version": ALGORITHM_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "variant": args.variant,
            "row_index": index,
            "request_id": row["request_id"],
            "pair_group_id": row.get("pair_group_id"),
            "split": row.get("split"),
            "view": row.get("view"),
            "source_image_path": str(source_path),
            "target_image_path": str(Path(str(row["image"])).expanduser().resolve()),
            "target_usage": "eval_reference_only",
            "output_path": str(output_path.resolve()),
            "prompt": row["prompt"],
            "prompt_sha256": sha256_bytes(str(row["prompt"]).encode("utf-8")),
            "prompt_variant": args.variant,
            "seed": row_seed(row, seed_map, args.seed_base),
            "num_inference_steps": args.steps,
            "use_kv_cache": bool(args.use_kv_cache),
            "dtype": args.dtype,
            "source_size_declared": row.get("source_size"),
            "input_sha256": row_hash,
            "run_identity_sha256": args.run_identity_sha256,
            "input_path": str(source_path),
            "cot_source": row.get("cot_source", "teacher_reference" if args.variant == "teacher_cot" else None),
            "executor_prompt_protocol_version": row.get("executor_prompt_protocol_version", row.get("protocol_version")),
            "time_condition_unit": row.get("time_condition_unit"),
            "prediction_interval_weeks": row.get("prediction_interval_weeks"),
            "source_sha256": None,
            "retry_count": 0,
            "started_at_utc": utc_now(),
        }
        started = time.time()
        action = getattr(args, "condition_dropout", "keep")
        ablation = getattr(args, "condition_ablation", "none")
        effective_prompt = str(row["prompt"])
        canvas_size = None
        if ablation != "none":
            effective_prompt, canvas_size = fixed_ablation_inputs(row, ablation)
            base.update(condition_ablation=ablation,
                        condition_ablation_version="wgsr_v9_executor_input_ablation_v1",
                        original_prompt_sha256=base["prompt_sha256"],
                        prompt=effective_prompt, prompt_sha256=sha256_bytes(effective_prompt.encode()),
                        source_image_conditioning=ablation != "no_image",
                        source_image_opened_by_runner=ablation != "no_image")
        if action == "drop_text":
            from ..executor_dropout import drop_cot
            effective_prompt = drop_cot(effective_prompt)
        if action != "keep":
            base.update(condition_dropout=action, original_prompt_sha256=base["prompt_sha256"],
                        prompt=effective_prompt, prompt_sha256=sha256_bytes(effective_prompt.encode()))
        try:
            if ablation == "no_image":
                source = None
                width, height = canvas_size
            else:
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                base["source_sha256"] = sha256_file(source_path)
                with Image.open(source_path) as source_file:
                    source = source_file.convert("RGB")
                width, height = expected_source_size(row, source)
            generator = torch.Generator(device="cuda").manual_seed(int(base["seed"]))
            with torch.inference_mode():
                result = pipe(
                    prompt=effective_prompt,
                    image=None if action == "drop_image" else source,
                    width=width,
                    height=height,
                    output_resolution=round(math.sqrt(width * height)),
                    num_inference_steps=args.steps,
                    true_cfg_scale=1.0,
                    generator=generator,
                    use_kv_cache=args.use_kv_cache,
                )
            generated_image = result.images[0]
            if generated_image.size != (width, height):
                raise ValueError(
                    f"pipeline returned {generated_image.size}, expected {(width, height)}"
                )
            temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
            generated_image.save(temporary, format="PNG")
            os.replace(temporary, output_path)
            base.update(
                status="completed",
                error=None,
                output_size=list(generated_image.size),
                output_mode=generated_image.mode,
                output_bytes=output_path.stat().st_size,
                output_sha256=sha256_file(output_path),
                elapsed_seconds=round(time.time() - started, 4),
                completed_at_utc=utc_now(),
            )
            generated += 1
            print(
                f"rank={args.rank} completed row={index} "
                f"sec={base['elapsed_seconds']:.1f}",
                flush=True,
            )
        except Exception as exc:  # preserve a resumable failure record
            temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
            if temporary.exists():
                temporary.unlink()
            base.update(
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=round(time.time() - started, 4),
                completed_at_utc=utc_now(),
            )
            failed += 1
            print(f"rank={args.rank} FAILED row={index}: {base['error']}", flush=True)
        append_jsonl(ledger, base)
    write_json(
        rank_dir / "worker_manifest.json",
        {
            "algorithm_version": ALGORITHM_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "variant": args.variant,
            "rank": args.rank,
            "world_size": args.world_size,
            "input_manifest": str(args.input_manifest),
            "input_manifest_sha256": input_sha,
            "seed_manifest": str(args.seed_manifest) if args.seed_manifest else None,
            "seed_manifest_sha256": seed_sha,
            "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
            "checkpoint_sha256": args.run_identity["checkpoint_sha256"],
            **({"pretrained_transformer": args.run_identity["pretrained_transformer"]}
               if args.checkpoint is None else {}),
            "transformer_loading_info": loading,
            "dtype": str(dtype),
            "assigned_rows": len(assigned),
            "generated_rows": generated,
            "failed_rows": failed,
            "finished_at_utc": utc_now(),
        },
    )
    return 1 if failed else 0


def nvidia_memory() -> dict[str, int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    values: dict[str, int] = {}
    for line in output.splitlines():
        if line.strip():
            index, used = line.split(",", 1)
            values[index.strip()] = int(used.strip())
    return values


def latest_records(paths: Iterable[Path]) -> dict[int, dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    for path in paths:
        for record in read_optional_jsonl(path):
            latest[int(record["row_index"])] = record
    return latest


def consolidate(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    input_sha: str,
    seed_sha: str | None,
) -> dict[str, Any]:
    ledgers = [
        args.output_root / f"rank-{rank:02d}" / "results.jsonl"
        for rank in range(args.world_size)
    ]
    records_by_index = latest_records(ledgers)
    expected = {int(row["inference_row_index"]): input_hash(row) for row in rows}
    records_by_index = {index: record for index, record in records_by_index.items() if index in expected}
    for index, record in list(records_by_index.items()):
        stale = record.get("input_sha256") != expected[index]
        if getattr(args, "run_identity_sha256", None) is not None:
            stale |= record.get("run_identity_sha256") != args.run_identity_sha256
        missing_output = not Path(record.get("output_path") or "").is_file()
        if stale or missing_output:
            records_by_index[index] = {**record, "status": "failed",
                "error": "stale input/run identity or missing output: result does not match this run"}
    records = [records_by_index[index] for index in sorted(records_by_index)]
    successful = [record for record in records if record.get("status") == "completed"]
    failed = [record for record in records if record.get("status") != "completed"]
    expected_indices = {int(row["inference_row_index"]) for row in rows}
    successful_indices = {int(record["row_index"]) for record in successful}
    missing = sorted(expected_indices - successful_indices)
    results_path = args.output_root / "results.jsonl"
    records_json = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for record in records
    )
    temporary = results_path.with_name(f".{results_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(records_json)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, results_path)
    status = (
        "complete"
        if not missing and not failed and len(successful) == len(expected_indices)
        else "complete_with_failures"
    )
    manifest = {
        "algorithm_version": ALGORITHM_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "variant": args.variant,
        "split": "val",
        "input_manifest": str(args.input_manifest.resolve()),
        "input_manifest_sha256": input_sha,
        "selected_rows": len(rows),
        "successful_rows": len(successful),
        "failed_rows": len(failed),
        "missing_rows": missing,
        "results": str(results_path.resolve()),
        "results_sha256": sha256_file(results_path),
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint is not None else None,
        "checkpoint_sha256": sha256_file(args.checkpoint) if args.checkpoint is not None else None,
        "model_path": str(args.model_path.resolve()),
        "world_size": args.world_size,
        "gpu_ids": args.gpu_ids,
        "steps": args.steps,
        "seed_base": args.seed_base,
        "seed_manifest": str(args.seed_manifest.resolve()) if args.seed_manifest else None,
        "seed_manifest_sha256": seed_sha,
        "dtype": args.dtype,
        "use_kv_cache": bool(args.use_kv_cache),
        "source_only": True,
        "target_usage": "eval_reference_only",
        "target_image_opened_by_runner": False,
        "conditioning_source": "teacher_reference" if args.variant == "teacher_cot" else args.variant,
        "target_informed_conditioning": args.variant == "teacher_cot",
        "run_identity": getattr(args, "run_identity", None),
        "run_identity_sha256": getattr(args, "run_identity_sha256", None),
        "finished_at_utc": utc_now(),
    }
    write_json(args.output_root / "inference_manifest.json", manifest)
    return manifest


def supervisor_main(args: argparse.Namespace) -> int:
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_root / "inference.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another inference supervisor owns {lock_path}") from exc
        return _supervisor_locked(args)


def _supervisor_locked(args: argparse.Namespace) -> int:
    rows, input_sha = load_input_rows(args.input_manifest, parse_indices(args.row_indices))
    validate_input_contract(args, rows)
    _, seed_sha = load_seed_map(args.seed_manifest)
    args.run_identity = run_identity(args, rows, input_sha, seed_sha)
    args.run_identity_sha256 = canonical_sha256(args.run_identity)
    if args.steps != 20:
        raise ValueError("the reviewed Qwen Image 2.1 inference contract requires exactly 20 steps")
    gpu_ids = select_gpu_ids(args.gpu_ids)
    if len(gpu_ids) != args.world_size:
        raise ValueError(f"world size {args.world_size} does not match GPU list {gpu_ids}")
    manifest_path = args.output_root / "inference_manifest.json"
    checkpoint_sha = args.run_identity["checkpoint_sha256"]
    previous_path = manifest_path if manifest_path.exists() else args.output_root / "supervisor_manifest.json"
    if previous_path.exists() and not args.overwrite:
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        validate_resume(previous, args.run_identity)
        if previous.get("status") == "complete":
            checked = consolidate(args, rows, input_sha, seed_sha)
            if checked["status"] == "complete":
                print(f"inference already complete: {manifest_path}", flush=True)
                return 0
    memory = nvidia_memory()
    if any(gpu not in memory for gpu in gpu_ids):
        raise RuntimeError("selected GPU is unavailable")
    busy = {gpu: memory[gpu] for gpu in gpu_ids if memory[gpu] >= 1024}
    if busy:
        raise RuntimeError(f"selected GPUs are occupied above 1 GiB: {busy}")
    view = make_transformer_view(args.model_path, args.checkpoint, args.output_root)
    planned = {
        "algorithm_version": ALGORITHM_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "status": "running",
        "variant": args.variant,
        "input_manifest": str(args.input_manifest.resolve()),
        "input_manifest_sha256": input_sha,
        "selected_rows": len(rows),
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint is not None else None,
        "checkpoint_sha256": checkpoint_sha,
        "model_path": str(args.model_path.resolve()),
        "transformer_view": str(view.resolve()),
        "world_size": args.world_size,
        "gpu_ids": args.gpu_ids,
        "steps": args.steps,
        "seed_base": args.seed_base,
        "seed_manifest": str(args.seed_manifest.resolve()) if args.seed_manifest else None,
        "seed_manifest_sha256": seed_sha,
        "dtype": args.dtype,
        "use_kv_cache": bool(args.use_kv_cache),
        "source_only": True,
        "target_usage": "eval_reference_only",
        "target_image_opened_by_runner": False,
        "conditioning_source": "teacher_reference" if args.variant == "teacher_cot" else args.variant,
        "target_informed_conditioning": args.variant == "teacher_cot",
        "run_identity": args.run_identity,
        "run_identity_sha256": args.run_identity_sha256,
        "started_at_utc": utc_now(),
    }
    write_json(args.output_root / "supervisor_manifest.json", planned)
    processes: list[tuple[int, subprocess.Popen[Any]]] = []
    jobs: list[dict[str, Any]] = []
    for rank, gpu_id in enumerate(gpu_ids):
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": gpu_id,
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DIFFSYNTH_SKIP_DOWNLOAD": "True",
        }
        log_path = args.output_root / "logs" / f"worker-{rank:02d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("a", encoding="utf-8")
        command = [
            sys.executable,
            str(CLI_SCRIPT),
            "--worker",
            "--variant",
            args.variant,
            "--input-manifest",
            str(args.input_manifest),
            "--model-path",
            str(args.model_path),
            "--output-root",
            str(args.output_root),
            "--rank",
            str(rank),
            "--world-size",
            str(args.world_size),
            "--gpu-ids",
            args.gpu_ids,
            "--steps",
            str(args.steps),
            "--seed-base",
            str(args.seed_base),
            "--dtype",
            args.dtype,
        ]
        if args.checkpoint is None:
            command.append("--pretrained")
        else:
            command.extend(("--checkpoint", str(args.checkpoint)))
        if getattr(args, "config", None) is not None:
            command.extend(("--config", str(args.config)))
        if getattr(args, "condition_dropout", "keep") != "keep":
            command.extend(("--condition-dropout", args.condition_dropout))
        if getattr(args, "condition_ablation", "none") != "none":
            command.extend(("--condition-ablation", args.condition_ablation))
        if args.seed_manifest:
            command.extend(("--seed-manifest", str(args.seed_manifest)))
        if args.row_indices:
            command.extend(("--row-indices", args.row_indices))
        command.append("--use-kv-cache" if args.use_kv_cache else "--no-use-kv-cache")
        if args.overwrite:
            command.append("--overwrite")
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log.close()
        processes.append((rank, process))
        jobs.append(
            {
                "rank": rank,
                "gpu": gpu_id,
                "pid": process.pid,
                "log": str(log_path),
                "status": "running",
                "command": command,
            }
        )
    while True:
        for job, (_, process) in zip(jobs, processes):
            code = process.poll()
            if code is not None:
                job.update(
                    status="completed" if code == 0 else "failed",
                    exit_code=code,
                    finished_at_utc=utc_now(),
                )
        write_json(args.output_root / "jobs.json", {"updated_at_utc": utc_now(), "jobs": jobs})
        if all(process.poll() is not None for _, process in processes):
            break
        time.sleep(10)
    manifest = consolidate(args, rows, input_sha, seed_sha)
    planned.update(manifest, status=manifest["status"], jobs=jobs)
    write_json(args.output_root / "supervisor_manifest.json", planned)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0 if manifest["status"] == "complete" else 1
