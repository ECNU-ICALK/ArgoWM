#!/usr/bin/env python3
"""Benchmark or run a planner replica with source-only validation rows."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from wgsr_v9.planner_inference import (
    ALGORITHM_VERSION, GpuMonitor, append_jsonl, atomic_json,
    atomic_text, read_rows, sha256, source_request, validate_completion,
    protocol_version_for_cot, model_device_map,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, help="Adapter saved by the selected local training run.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Invalid batch size or shard selection")
    config = json.loads(args.config.read_text())
    if args.adapter is not None:
        config["adapter"] = str(args.adapter.resolve())
    if not config.get("adapter"):
        parser.error("Specify --adapter or configure an adapter path after training completes")
    protocol_version = protocol_version_for_cot(str(config.get("cot_version", "v2")),
                                                config.get("numeric_evidence_format"), config.get("time_condition_unit"),
                                                enable_thinking=config.get("enable_thinking", True),
                                                output_wrapper=config.get("output_wrapper", "think"))
    devices = [int(value) for value in args.gpus.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["MAX_PIXELS"] = str(config["max_pixels"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["MODELSCOPE_OFFLINE"] = "1"
    package = Path(config["data_root"]).resolve()
    dataset = package / "val.jsonl"
    adapter = Path(config["adapter"]).resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    all_rows = read_rows(dataset)
    package_manifest = json.loads((package / "package_manifest.json").read_text())
    for key, default in (("enable_thinking", True), ("output_wrapper", "think")):
        if config.get(key, default) != package_manifest.get(key, default):
            raise ValueError(f"Dataset {key} does not match the inference configuration")
    if config.get("planner_protocol_version"):
        if any(row.get("protocol_version") != config["planner_protocol_version"] for row in all_rows):
            raise ValueError("Dataset does not match configured planner protocol")
    ids = [row["id"] for row in all_rows]
    if len(set(ids)) != len(ids) or len(ids) != config["expected_rows"]:
        raise ValueError("Validation dataset count or IDs are invalid")
    rows = [(index, row) for index, row in enumerate(all_rows) if index % args.num_shards == args.shard_index]
    if args.limit:
        rows = rows[:args.limit]
    requests = {index: source_request(row, package) for index, row in rows}
    identity = {
        "algorithm_version": ALGORITHM_VERSION, "protocol_version": protocol_version,
        "dataset_sha256": sha256(dataset), "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
        "model_path": config["model"], "adapter_path": str(adapter),
        "config_sha256": sha256(args.config), "shard_index": args.shard_index, "num_shards": args.num_shards,
        "max_new_tokens": args.benchmark_tokens if args.benchmark else config["max_new_tokens"],
        "temperature": 0, "benchmark": args.benchmark,
        "enable_thinking": config.get("enable_thinking", True),
        "output_wrapper": config.get("output_wrapper", "think"),
    }
    manifest_path = output / "run_manifest.json"
    ledger = output / "results.jsonl"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["identity"] != identity:
            raise ValueError("Output directory belongs to another inference configuration")
    elif ledger.exists():
        raise ValueError("Ledger without a run manifest")
    existing = read_rows(ledger) if ledger.exists() else []
    done = {row["pair_id"] for row in existing if row["status"] == "completed" and Path(row["output_path"]).is_file()}
    pending = [(index, row) for index, row in rows if row["id"] not in done]
    manifest = {
        **identity, "identity": identity, "input_path": str(dataset), "output_path": str(output),
        "status": "loading", "error": None, "retry_count": 0, "pid": os.getpid(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "gpus": devices,
        "batch_size_per_replica": args.batch_size, "selected_rows": len(rows),
        "resumed_rows": len(done), "source_only": True,
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps({"status": "loading", "gpus": devices, "batch_size": args.batch_size}), flush=True)
    import torch
    from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
    from swift.template import get_template
    torch.set_num_threads(4)
    torch.manual_seed(42)
    with GpuMonitor(devices, output / "gpu_samples.jsonl", protocol_version) as monitor:
        engine = TransformersEngine(
            config["model"], adapters=[str(adapter)], torch_dtype=torch.bfloat16,
            device_map="balanced", model_type="qwen3_5", template_type="qwen3_8",
            attn_impl="sdpa", max_batch_size=args.batch_size,
            model_kwargs={"max_memory": {
                index: config.get("max_memory_per_gpu", "44GiB") for index in range(len(devices))}},
        )
        # Merge into the in-memory BF16 model; the saved training adapter remains unchanged.
        engine.model = engine.model.merge_and_unload()
        engine.engine = engine.model
        # BaseArguments.from_pretrained does not restore every training template field.
        # Set the reviewed training contract explicitly instead of inheriting defaults.
        engine.template = get_template(
            engine.template.processor, template_type="qwen3_8", enable_thinking=config.get("enable_thinking", True),
            max_length=config["max_length"], max_pixels=config["max_pixels"],
            padding_side="left", template_backend="swift",
        )
        engine.template.set_mode("transformers")
        template_type = str(engine.template.template_meta.template_type)
        manifest.update(status="running", device_map=model_device_map(engine.model),
                        template=template_type,
                        enable_thinking=bool(engine.template.enable_thinking),
                        thinking_prefix=engine.template.template_meta.thinking_prefix,
                        dtype="bfloat16", adapter_merged_in_memory=True)
        if not template_type.lower().endswith("qwen3_8") or manifest["enable_thinking"] != config.get("enable_thinking", True):
            raise ValueError(
                f"Training template mode was not restored: template={template_type!r}, "
                f"enable_thinking={manifest['enable_thinking']!r}"
            )
        atomic_json(manifest_path, manifest)
        print(json.dumps({"status": "loaded", "template": manifest["template"]}), flush=True)
        audit = []
        for index, row in rows[:2]:
            encoded = engine.template.encode(InferRequest(**requests[index]))
            decoded = engine.tokenizer.decode(encoded["input_ids"])
            expected_suffix = "<think>\n" if manifest["enable_thinking"] else "<think>\n\n</think>\n\n"
            if not decoded.endswith(expected_suffix) or row["messages"][-1]["content"] in decoded:
                raise ValueError("Encoded prompt violates the source-only thinking contract")
            audit.append({"algorithm_version": ALGORITHM_VERSION, "protocol_version": protocol_version,
                          "input_path": requests[index]["images"][0], "output_path": str(output / "encoding_audit.json"),
                          "status": "verified", "error": None, "retry_count": 0,
                          "row_index": index, "pair_id": row["id"], "prompt_tokens": len(encoded["input_ids"]),
                          "image_grid_thw": encoded["image_grid_thw"].tolist(), "prompt_suffix": decoded[-80:],
                          "assistant_teacher_in_input": False})
        atomic_json(output / "encoding_audit.json", audit)
        generation = RequestConfig(max_tokens=identity["max_new_tokens"], temperature=0,
                                   repetition_penalty=1, seed=42, return_details=True)
        if args.benchmark:
            batch = rows[:args.batch_size]
            if len(batch) != args.batch_size:
                raise ValueError("Insufficient benchmark rows")
            for repeat in range(2):
                for index in range(len(devices)):
                    torch.cuda.reset_peak_memory_stats(index)
                start = time.time()
                responses = engine.infer([InferRequest(**requests[index]) for index, _ in batch],
                                         generation, use_tqdm=False)
                elapsed = time.time() - start
                tokens = sum(response.usage.completion_tokens for response in responses)
                record = {**identity, "input_path": str(dataset), "output_path": str(output),
                          "status": "completed", "error": None, "retry_count": 0,
                          "repeat": repeat, "batch_size": args.batch_size,
                          "elapsed_seconds": elapsed, "output_tokens": tokens,
                          "tokens_per_second": tokens / elapsed,
                          "gpu": monitor.summary(start, time.time()),
                          "torch_peak_allocated_gib": [torch.cuda.max_memory_allocated(i) / 2**30 for i in range(len(devices))]}
                append_jsonl(output / "benchmark.jsonl", record)
                print(json.dumps(record), flush=True)
        else:
            for start_index in range(0, len(pending), args.batch_size):
                batch = pending[start_index:start_index + args.batch_size]
                start = time.time()
                oom_retries = 0
                while True:
                    try:
                        responses = engine.infer([InferRequest(**requests[index]) for index, _ in batch],
                                                 generation, use_tqdm=False)
                        break
                    except torch.cuda.OutOfMemoryError:
                        if engine.max_batch_size <= 1:
                            raise
                        engine.max_batch_size = max(1, engine.max_batch_size // 2)
                        oom_retries += 1
                        import gc
                        gc.collect()
                        for device in range(len(devices)):
                            with torch.cuda.device(device):
                                torch.cuda.empty_cache()
                        print(json.dumps({"status": "retrying", "reason": "cuda_oom",
                                          "new_batch_size": engine.max_batch_size}), flush=True)
                elapsed = time.time() - start
                for (index, row), response in zip(batch, responses, strict=True):
                    choice = response.choices[0]
                    text = choice.message.content or ""
                    cot, error = validate_completion(
                        text, choice.finish_reason, numeric_evidence_format=config.get("numeric_evidence_format"),
                        view=row["planner_metadata"].get("view"), output_wrapper=config.get("output_wrapper", "think"))
                    raw_path = output / "raw" / f"{index:05d}.txt"
                    atomic_text(raw_path, text)
                    cot_path = output / "growth_cot" / f"{index:05d}.txt"
                    if cot:
                        atomic_text(cot_path, cot)
                    record = {
                        "algorithm_version": ALGORITHM_VERSION, "protocol_version": protocol_version,
                        "row_index": index, "pair_id": row["id"], **row["planner_metadata"],
                        "input_path": requests[index]["images"][0], "output_path": str(raw_path),
                        "growth_cot_path": str(cot_path) if cot else None,
                        "status": "completed", "error": error, "retry_count": oom_retries,
                        "format_valid": cot is not None, "finish_reason": choice.finish_reason,
                        "generated_text": text, "growth_cot": cot, "usage": asdict(response.usage),
                        "batch_elapsed_seconds": elapsed, "batch_size": len(batch),
                        "effective_batch_size": engine.max_batch_size,
                        "prompt_sha256": hashlib.sha256(json.dumps(requests[index],sort_keys=True).encode()).hexdigest(),
                    }
                    append_jsonl(ledger, record)
                progress = {"status": "running", "completed": len(done) + start_index + len(batch),
                            "total": len(rows), "last_batch_seconds": elapsed,
                            "last_batch_tokens_per_second": sum(r.usage.completion_tokens for r in responses) / elapsed,
                            "gpu": monitor.summary(start, time.time())}
                atomic_json(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        manifest.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        if "--output-dir" in sys.argv:
            path = Path(sys.argv[sys.argv.index("--output-dir") + 1]) / "run_manifest.json"
            if path.exists():
                manifest = json.loads(path.read_text())
                if manifest.get("pid") == os.getpid():
                    manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                                    completed_at_utc=datetime.now(timezone.utc).isoformat())
                    atomic_json(path, manifest)
        raise
