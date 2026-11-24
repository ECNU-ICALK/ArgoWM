#!/usr/bin/env python3
"""Launch four independent planner replicas and export their completed CoT rows."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wgsr_v9.executor_training import executor_prompt, NAMED_WEEKS_PROTOCOL_VERSION, prediction_interval_weeks
from wgsr_v9.planner_adapter import growth_cot_body
from wgsr_v9.planner_inference import (
    ALGORITHM_VERSION, atomic_json, atomic_text, read_rows, sha256, protocol_version_for_cot, validate_completion,
)


def export_results(config: dict, root: Path, shard_count: int) -> dict:
    protocol_version = protocol_version_for_cot(str(config.get("cot_version", "v2")),
                                                config.get("numeric_evidence_format"), config.get("time_condition_unit"),
                                                enable_thinking=config.get("enable_thinking", True),
                                                output_wrapper=config.get("output_wrapper", "think"))
    dataset = Path(config["data_root"]) / "val.jsonl"
    rows = read_rows(dataset)
    canonical = {row["pair_id"]: row for row in read_rows(Path(config["evaluation_manifest"]))}
    manifests = [json.loads((root / f"shard_{i:02d}" / "run_manifest.json").read_text()) for i in range(shard_count)]
    identity_fields = ["dataset_sha256", "adapter_sha256", "config_sha256", "max_new_tokens", "temperature"]
    if config.get("output_wrapper") == "GROWTH_COT":
        identity_fields.extend(["protocol_version", "enable_thinking", "output_wrapper"])
        for manifest in manifests:
            if manifest.get("enable_thinking") is not False or manifest.get("output_wrapper") != "GROWTH_COT":
                raise ValueError("Shard thinking/output contract mismatch")
    for field in identity_fields:
        if len({manifest[field] for manifest in manifests}) != 1:
            raise ValueError(f"Shard provenance mismatch: {field}")
    results = {}
    for index in range(shard_count):
        for result in read_rows(root / f"shard_{index:02d}" / "results.jsonl"):
            if result["pair_id"] in results:
                raise ValueError(f"Duplicate result ID: {result['pair_id']}")
            results[result["pair_id"]] = result
    if set(results) != {row["id"] for row in rows}:
        raise ValueError("Generated IDs do not exactly match the validation manifest")
    if config.get("numeric_evidence_format") == "maize_sv1_v1":
        from wgsr_v9.maize.planner_export import export_maize_results
        return export_maize_results(config, root, rows, canonical, manifests, results)
    ordered = []
    exports = []
    rejected = []
    executor_version = NAMED_WEEKS_PROTOCOL_VERSION
    protocol = ROOT / "protocols/qwen_image21_executor_v9_v3_metric_named_weeks"
    prompt_template = (protocol / "EXECUTOR_PROMPT_TEMPLATE.txt").read_text()
    cot_suffix = (protocol / "TEACHER_COT_SUFFIX.txt").read_text()
    for index, row in enumerate(rows):
        result = results[row["id"]]
        reference = canonical[row["id"]]
        planner_source = Path(result["input_path"]).resolve()
        canonical_source = Path(reference["source_image_path"]).resolve()
        if planner_source != canonical_source and sha256(planner_source) != sha256(canonical_source):
            raise ValueError(f"Canonical source content mismatch: {row['id']}")
        ordered.append(result)
        if not result["format_valid"]:
            rejected.append(result)
            continue
        metadata = row["planner_metadata"]
        if config.get("output_wrapper") == "GROWTH_COT":
            checked, error = validate_completion(result["growth_cot"], result["finish_reason"],
                numeric_evidence_format=config.get("numeric_evidence_format"), view=metadata.get("view"),
                output_wrapper="GROWTH_COT")
            if error:
                raise ValueError(f"Invalid direct-output result marked valid: {row['id']}: {error}")
        prompt = executor_prompt(metadata, prompt_template=prompt_template, cot_suffix=cot_suffix,
                                 variant="teacher_cot", teacher_cot=growth_cot_body(result["growth_cot"]),
                                 protocol_version=executor_version)
        exports.append({
            "algorithm_version": ALGORITHM_VERSION, "protocol_version": protocol_version,
            "executor_prompt_protocol_version": executor_version,
            "request_id": row["id"], "pair_id": row["id"], **metadata,
            "image": reference["target_image_path"], "edit_image": str(canonical_source),
            "planner_source_image_path": str(planner_source),
            "time_condition_unit": "weeks", "prediction_interval_weeks": prediction_interval_weeks(metadata),
            "numeric_evidence_format": "metric_named_v1",
            "target_usage": "eval_reference_only", "source_size": reference["source_size"],
            "prompt": prompt, "executor_variant": "predicted_cot", "cot_source": "planner_prediction", "adapter_sha256": manifests[0]["adapter_sha256"],
            "input_path": result["output_path"], "output_path": str(root / "executor_prompts.jsonl"),
            "status": "ready", "error": None, "retry_count": 0, "original_row_index": index,
        })
    for name, values in [("results.jsonl", ordered), ("executor_prompts.jsonl", exports), ("rejected.jsonl", rejected)]:
        atomic_text(root / name, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values))
    summary = {
        "algorithm_version": ALGORITHM_VERSION, "protocol_version": protocol_version,
        "input_path": str(dataset.resolve()), "output_path": str(root),
        "status": "completed", "error": None, "retry_count": 0,
        "generated_rows": len(ordered), "valid_rows": len(exports), "rejected_rows": len(rejected),
        "truncated_rows": sum(row["finish_reason"] == "length" for row in ordered),
        "teacher_fallback_used": False, "dataset_sha256": sha256(dataset),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(root / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--gpu-groups", default="0,1;2,3;4,5;6,7")
    args = parser.parse_args()
    os.chdir(ROOT)
    config = json.loads(args.config.read_text())
    protocol_version = protocol_version_for_cot(str(config.get("cot_version", "v2")),
                                                config.get("numeric_evidence_format"), config.get("time_condition_unit"),
                                                enable_thinking=config.get("enable_thinking", True),
                                                output_wrapper=config.get("output_wrapper", "think"))
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    groups = args.gpu_groups.split(";")
    devices = [int(gpu) for group in groups for gpu in group.split(",")]
    if len(set(devices)) != len(devices):
        raise ValueError("GPU groups overlap")
    probe = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits",
         "-i", ",".join(map(str, devices))], capture_output=True, text=True, check=True,
    )
    if any(float(line.split(",")[1]) > 1024 for line in probe.stdout.splitlines()):
        raise RuntimeError("Selected GPUs are occupied; refusing to compete with another job")
    state = {"algorithm_version": ALGORITHM_VERSION, "protocol_version": protocol_version,
             "input_path": str(args.config.resolve()), "output_path": str(root),
             "status": "running", "error": None, "retry_count": 0, "pid": os.getpid(),
             "started_at_utc": datetime.now(timezone.utc).isoformat(), "jobs": []}
    atomic_json(root / "pipeline_status.json", state)
    processes = []
    for index, group in enumerate(groups):
        output = root / f"shard_{index:02d}"
        output.mkdir(exist_ok=True)
        env = os.environ.copy()
        env.update({"PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1",
                    "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"})
        command = [sys.executable, "scripts/run_planner_inference.py", "--config", str(args.config.resolve()),
                   "--output-dir", str(output), "--gpus", group, "--batch-size", str(args.batch_size),
                   "--shard-index", str(index), "--num-shards", str(len(groups))]
        with (output / "worker.log").open("a") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL)
        processes.append(process)
        state["jobs"].append({"pid": process.pid, "gpus": group, "command": command,
                              "output_path": str(output), "status": "running"})
        atomic_json(root / "pipeline_status.json", state)
    while any(process.poll() is None for process in processes):
        state["generated_rows"] = 0
        for job, process in zip(state["jobs"], processes, strict=True):
            progress = Path(job["output_path"]) / "progress.json"
            job["progress"] = json.loads(progress.read_text()) if progress.exists() else None
            state["generated_rows"] += job["progress"]["completed"] if job["progress"] else 0
            code = process.poll()
            job["status"] = "running" if code is None else "completed" if code == 0 else "failed"
            job["exit_code"] = code
        atomic_json(root / "pipeline_status.json", state)
        time.sleep(15)
    failures = [process.returncode for process in processes if process.returncode != 0]
    if failures:
        state.update(status="failed", error=f"Worker exit codes: {failures}")
        atomic_json(root / "pipeline_status.json", state)
        return 1
    try:
        summary = export_results(config, root, len(groups))
    except Exception as exc:
        state.update(status="failed", error=f"Export failed: {type(exc).__name__}: {exc}")
        atomic_json(root / "pipeline_status.json", state)
        raise
    state.update(status="completed", summary=summary, completed_at_utc=datetime.now(timezone.utc).isoformat())
    for job in state["jobs"]:
        job.update(status="completed", exit_code=0)
    atomic_json(root / "pipeline_status.json", state)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
