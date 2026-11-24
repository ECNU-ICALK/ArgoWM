"""Audit maize teachers and atomically package matched four-method inputs."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from PIL import Image

from wgsr_v9.io import canonical_sha256, read_jsonl, sha256_path, write_json, write_jsonl
from .cot import ANNOTATION_QC_VERSION, build_request, protocol_sha256, render_cot
from .experiment import (EXECUTOR_PROTOCOL, PLANNER_FORMAT, PLANNER_PROTOCOL, PROTOCOL,
                         PROTOCOL_ROOT, ROOT, SIZE, executor_prompt, planner_messages,
                         validate_growth_cot)

ALGORITHM = "wgsr_maize_four_way_package_v1"
EXECUTOR_ALGORITHM = "wgsr_v9_qwen_image21_executor_sft_v1"


def provenance(input_path: Path, output_path: Path, *, protocol: str = PROTOCOL,
               status: str = "ready") -> dict[str, Any]:
    return {"algorithm_version": ALGORITHM, "protocol_version": protocol,
            "input_path": str(input_path.resolve()), "output_path": str(output_path.resolve()),
            "status": status, "error": None, "retry_count": 0}


def unique_index(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    indexed = {str(row[key]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate {key}")
    return indexed


def prepare(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())

    def path(key: str) -> Path:
        return (ROOT / config[key]).resolve()

    pairs_path, requests_path, teacher, teacher_protocol, output = (
        path(key) for key in ("pairs", "requests", "teacher_root", "teacher_protocol_root", "output_root"))
    contract = json.loads((PROTOCOL_ROOT / "CONTRACT.json").read_text())
    status = json.loads((teacher / "status.json").read_text())
    if status["status"] not in {"complete", "completed"} or status["succeeded"] != 8000 or status["failed"]:
        raise ValueError("Teacher synthesis must finish successfully for all 8000 queries before packaging")
    pairs = read_jsonl(pairs_path)
    requests = unique_index(read_jsonl(requests_path), "request_id")
    pair_index = unique_index(pairs, "pair_id")
    # A resumed run may append another result for a previously failed request.
    results = {row["request_id"]: row for row in read_jsonl(teacher / "results.jsonl")}
    if set(pair_index) != set(requests) or set(pair_index) != set(results):
        raise ValueError("Pair, request, and teacher result IDs differ")
    counts = dict(Counter(pair["split"] for pair in pairs))
    expected = {"train": contract["train_rows"], "validation": contract["validation_rows"]}
    if counts != expected:
        raise ValueError(f"Unexpected split counts: {counts}")
    plants = {split: {p["plant_id"] for p in pairs if p["split"] == split} for split in expected}
    if plants["train"] & plants["validation"]:
        raise ValueError("Plant leakage across train/validation")
    teacher_hash = protocol_sha256(teacher_protocol)
    file_hashes: dict[str, str] = {}
    cots: dict[str, str] = {}
    audit = []
    for pair in pairs:
        key = pair["pair_id"]
        request, result = requests[key], results[key]
        if (pair.get("species"), pair.get("view"), pair.get("training_ready")) != ("maize", "sv1", True):
            raise ValueError(f"Ineligible pair: {key}")
        if (pair["actual_interval_days"] <= 0 or pair["stage_delta"] <= 0
                or pair["source"]["plant_id"] != pair["target"]["plant_id"]
                or pair["source"]["plant_id"] != pair["plant_id"]):
            raise ValueError(f"Nonforward or cross-plant pair: {key}")
        rebuilt = build_request(pair, protocol_root=teacher_protocol, repo_root=ROOT, check_images=False)
        if rebuilt != request:
            raise ValueError(f"Teacher request differs from frozen pair: {key}")
        if result.get("status") != "success" or result.get("protocol_sha256") != teacher_hash:
            raise ValueError(f"Teacher not valid: {key}")
        for endpoint in ("source", "target"):
            image_path = (ROOT / request[endpoint]["path"]).resolve()
            if str(image_path) not in file_hashes:
                with Image.open(image_path) as image:
                    if image.size != SIZE or image.mode != "RGB":
                        raise ValueError(f"Noncanonical maize image: {image_path}")
                file_hashes[str(image_path)] = sha256_path(image_path)
            if file_hashes[str(image_path)] != request[endpoint]["image_sha256"]:
                raise ValueError(f"Image hash mismatch: {key}: {endpoint}")
        annotation_path, rendered_path = Path(result["annotation_path"]), Path(result["output_path"])
        if sha256_path(annotation_path) != result["annotation_sha256"] or sha256_path(rendered_path) != result["rendered_sha256"]:
            raise ValueError(f"Teacher artifact hash mismatch: {key}")
        rendered = render_cot(json.loads(annotation_path.read_text()), request,
                              protocol_root=teacher_protocol, repo_root=ROOT, check_images=False)
        if rendered != rendered_path.read_text():
            raise ValueError(f"Teacher rendered content mismatch: {key}")
        cots[key] = validate_growth_cot(rendered)
        audit.append({**provenance(annotation_path, output / "teacher_audit.jsonl"), "request_id": key,
                      "annotation_sha256": result["annotation_sha256"], "rendered_sha256": result["rendered_sha256"],
                      "teacher_fingerprint": result["fingerprint"], "split": pair["split"]})
    inputs = {"pairs": sha256_path(pairs_path), "requests": sha256_path(requests_path),
              "teacher_audit": canonical_sha256(audit), "teacher_protocol": teacher_hash,
              "experiment_protocol": protocol_sha256(PROTOCOL_ROOT), "images": canonical_sha256(file_hashes),
              "annotation_qc_version": ANNOTATION_QC_VERSION}
    manifest_path = output / "package_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous.get("input_sha256") != inputs:
            raise ValueError("Existing package has different inputs; use a new versioned output_root")
        for relative, digest in previous["generated_file_sha256"].items():
            if sha256_path(output / relative) != digest:
                raise ValueError(f"Existing package artifact changed: {relative}")
        return previous
    executor = {variant: {split: [] for split in ("train", "val")} for variant in ("simple", "teacher_cot")}
    planner: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    evaluation, api_inputs, copies, seeds = [], [], [], []
    for pair in pairs:
        key = pair["pair_id"]
        split = "val" if pair["split"] == "validation" else "train"
        index = len(planner[split])
        source, target = (Path(pair[side]["path"]).resolve() for side in ("source", "target"))
        metadata = {"request_id": key, "pair_id": key, "pair_group_id": pair["pair_group_id"],
                    "plant_id": pair["plant_id"], "split": split, "species": "maize", "native_view": "sv1",
                    "view": "side0", "actual_interval_days": pair["actual_interval_days"],
                    "time_condition_unit": "days", "source_size": list(SIZE), "original_row_index": index}
        for variant in executor:
            executor[variant][split].append({**provenance(source, target, protocol=EXECUTOR_PROTOCOL), **metadata,
                "image": str(target), "edit_image": str(source), "executor_variant": variant,
                "cot_source": "teacher_annotation" if variant == "teacher_cot" else None,
                "prompt": executor_prompt(metadata, cots[key] if variant == "teacher_cot" else None)})
        planner[split].append({**provenance(source, output / "planner" / f"{split}.jsonl", protocol=PLANNER_PROTOCOL),
            "id": key, "messages": [*planner_messages(metadata, teacher_protocol),
                                    {"role": "assistant", "content": cots[key]}],
            "images": [str(source)], "planner_metadata": metadata})
        if split != "val":
            continue
        canonical = {**provenance(source, output / "evaluation/manifest.jsonl"), **metadata,
                     "row_index": index, "source_image_path": str(source), "target_image_path": str(target),
                     "source_image_sha256": file_hashes[str(source)], "target_image_sha256": file_hashes[str(target)],
                     "source_stage_id": pair["source_stage_id"], "target_stage_id": pair["target_stage_id"],
                     "source_stage": pair["source_stage_id"], "target_stage": pair["target_stage_id"],
                     "source_image_id": pair["source"]["image_id"], "target_image_id": pair["target"]["image_id"],
                     "horizon_bucket": pair["interval_bin"], "seed": 20261108 + index,
                     "target_usage": "eval_reference_only"}
        evaluation.append(canonical)
        api_inputs.append({**canonical, "output_path": str(output / "evaluation/gpt_image2_inputs.jsonl"),
                           "prompt": executor_prompt(metadata), "prompt_variant": "simple",
                           "model": "gpt-image-2", "source_only": True})
        copies.append({**canonical, "output_path": str(source), "status": "completed", "model": "source_copy",
                       "algorithm_version": "wgsr_maize_source_copy_v1", "prediction_sha256": file_hashes[str(source)]})
        seeds.append({**provenance(source, output / "evaluation/seeds.jsonl"), "pair_id": key, "seed": canonical["seed"]})
    write_jsonl(output / "teacher_audit.jsonl", audit)
    variants = {}
    for variant, splits in executor.items():
        for split, rows in splits.items():
            write_jsonl(output / "executor" / variant / f"{split}.jsonl", rows)
        variants[variant] = {"status": "complete", "rows_by_split": {k: len(v) for k, v in splits.items()},
                             "generated_file_sha256": {f"{s}.jsonl": sha256_path(output / "executor" / variant / f"{s}.jsonl") for s in splits}}
    write_json(output / "executor/package_manifest.json", {
        **provenance(pairs_path, output / "executor", protocol=EXECUTOR_PROTOCOL, status="complete"),
        "algorithm_version": EXECUTOR_ALGORITHM, "species": "maize", "variants": variants})
    for split, rows in planner.items():
        write_jsonl(output / "planner" / f"{split}.jsonl", rows)
    write_json(output / "planner/package_manifest.json", {
        **provenance(pairs_path, output / "planner", protocol=PLANNER_PROTOCOL, status="complete"),
        "planner_variant": "proposed_cot", "active_no_cot_planner": False, "cot_version": "v3",
        "numeric_evidence_format": PLANNER_FORMAT, "time_condition_unit": "days", "enable_thinking": False,
        "output_wrapper": "GROWTH_COT", "rows_by_split": {k: len(v) for k, v in planner.items()},
        "generated_file_sha256": {f"{s}.jsonl": sha256_path(output / "planner" / f"{s}.jsonl") for s in planner}})
    for name, rows in (("manifest", evaluation), ("gpt_image2_inputs", api_inputs), ("source_copy/results", copies), ("seeds", seeds)):
        write_jsonl(output / "evaluation" / f"{name}.jsonl", rows)
    generated = {p.relative_to(output).as_posix(): sha256_path(p) for p in sorted(output.rglob("*.json*")) if p != manifest_path}
    summary = {**provenance(pairs_path, output, status="complete"), "rows_by_split": counts,
               "plants_by_split": {k: len(v) for k, v in plants.items()}, "plant_overlap": 0,
               "unique_images": len(file_hashes), "validated_teacher_rows": len(cots),
               "source_copy_rows": len(copies), "image_size": list(SIZE), "input_sha256": inputs,
               "generated_file_sha256": generated, "training_started": False, "image_api_started": False}
    write_json(manifest_path, summary)
    return summary
