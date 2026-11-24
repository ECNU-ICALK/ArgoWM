"""Export only complete, source-only maize planner predictions to the executor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from wgsr_v9.io import sha256_path, write_json, write_jsonl
from wgsr_v9.planner_inference import source_request
from .experiment import (EXECUTOR_PROTOCOL, INFERENCE_PROTOCOL, executor_prompt,
                         validate_executor_row, validate_growth_cot)
from .training import provenance


def export_maize_results(config: dict[str, Any], root: Path, rows: list[dict[str, Any]],
                         canonical: dict[str, dict[str, Any]], manifests: list[dict[str, Any]],
                         results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    package = Path(config["data_root"]).resolve()
    dataset = package / "val.jsonl"
    expected_ids = {row["id"] for row in rows}
    if len(rows) != config["expected_rows"] or set(canonical) != expected_ids or set(results) != expected_ids:
        raise ValueError("Maize planner must cover the full canonical validation set")
    for manifest in manifests:
        if (manifest.get("dataset_sha256") != sha256_path(dataset)
                or manifest.get("protocol_version") != INFERENCE_PROTOCOL
                or manifest.get("source_only") is not True
                or manifest.get("benchmark") is not False
                or manifest.get("status") != "completed"
                or not manifest.get("adapter_sha256")):
            raise ValueError("Planner shard provenance is incompatible with formal maize inference")
    exports, rejected = [], []
    for index, row in enumerate(rows):
        result, reference = results[row["id"]], canonical[row["id"]]
        source = Path(reference["source_image_path"]).resolve()
        expected_request = source_request(row, package)
        expected_hash = hashlib.sha256(json.dumps(expected_request, sort_keys=True).encode()).hexdigest()
        if (Path(result["input_path"]).resolve() != source
                or result.get("prompt_sha256") != expected_hash
                or sha256_path(source) != reference["source_image_sha256"]):
            raise ValueError(f"Planner source-only request mismatch: {row['id']}")
        if not result.get("format_valid") or result.get("finish_reason") == "length":
            rejected.append(result)
            continue
        cot = validate_growth_cot(result["growth_cot"])
        metadata = row["planner_metadata"]
        export = {**metadata, **provenance(Path(result["output_path"]), root / "executor_prompts.jsonl",
                                           protocol=INFERENCE_PROTOCOL),
                  "executor_prompt_protocol_version": EXECUTOR_PROTOCOL,
                  "image": reference["target_image_path"], "edit_image": str(source),
                  "target_usage": "eval_reference_only", "prompt": executor_prompt(metadata, cot),
                  "executor_variant": "predicted_cot", "cot_source": "planner_prediction",
                  "adapter_sha256": manifests[0]["adapter_sha256"], "original_row_index": index}
        validate_executor_row(export, "predicted_cot")
        exports.append(export)
    write_jsonl(root / "rejected.jsonl", rejected)
    if rejected:
        raise ValueError(f"{len(rejected)} invalid maize predictions; fix/retry before export, no teacher fallback")
    write_jsonl(root / "results.jsonl", [results[row["id"]] for row in rows])
    write_jsonl(root / "executor_prompts.jsonl", exports)
    summary = {**provenance(dataset, root, protocol=INFERENCE_PROTOCOL, status="completed"),
               "generated_rows": len(rows), "valid_rows": len(exports), "rejected_rows": 0,
               "teacher_fallback_used": False, "dataset_sha256": sha256_path(dataset)}
    write_json(root / "summary.json", summary)
    return summary
