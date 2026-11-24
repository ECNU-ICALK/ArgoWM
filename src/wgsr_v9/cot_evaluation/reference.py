"""Frozen input alignment and reference scoring, independent of teacher phrasing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..evaluation_io import read_jsonl, sha256_file
from . import ALGORITHM_VERSION, PROTOCOL_VERSION
from .parser import parse_cot
from ..cot_metric_names import FORMAT, name_cot_evidence


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def indexed(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate {key}")
    return result



def alias_fields(values: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
    """Map versioned field IDs to evaluation IDs without touching model text."""
    result = {aliases.get(key, key): value for key, value in values.items()}
    if len(result) != len(values):
        raise ValueError("field alias collision")
    return result


def same_image(a: Path, b: Path, *, allow_hash_match: bool,
               cache: dict[Path, str]) -> bool:
    if a == b:
        return True
    if not allow_hash_match:
        return False
    for path in (a, b):
        if path not in cache:
            cache[path] = sha256_file(path)
    return cache[a] == cache[b]


def score_fields(parsed: dict[str, Any], request: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, evidence in request["metric_evidence"].items():
        predicted = parsed["fields"][key]
        binary = key == "panicle"
        spec = None if binary else contract["continuous_metrics"][key]
        if spec and evidence["metric_key"] != spec["metric_key"]:
            raise ValueError(f"reference metric mismatch: {key}")
        multiplier = 1 if binary else spec["display_multiplier"]
        reference: dict[str, float | None] = {}
        for role in ("source", "target"):
            observation = evidence[role]
            reference[role] = (float(observation["value"]) * multiplier
                               if observation["status"] == "available" and evidence["qc"].get("valid") is True else None)
        both = all(reference[role] is not None for role in ("source", "target"))
        delta = reference["target"] - reference["source"] if both else None
        reference.update(declared_change=delta, derived_change=delta)
        output[key] = {
            "reference": reference,
            "prediction": {role: predicted[role] for role in reference},
            "reference_status": {role: evidence[role]["status"] for role in ("source", "target")},
            "reference_qc": evidence["qc"], "metric_key": evidence["metric_key"],
            "unit": "binary" if binary else "percentage_points", "parse_errors": predicted["parse_errors"],
            "arithmetic_residual": predicted["arithmetic_residual"],
            "arithmetic_consistent": (abs(predicted["arithmetic_residual"]) <= (0 if binary else .02) + 1e-10
                                      if predicted["arithmetic_residual"] is not None else None),
            "unscorable_assertions": [role for role in ("source", "target") if reference[role] is None and predicted[role] is not None],
        }
    return output


def load_inputs(config: dict[str, Any], root: Path, output: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[str]]:
    contract = json.loads((root / config["protocol"]).read_text())
    protocol_root = root / config["cot_protocol_root"]
    bindings = json.loads((protocol_root / "VIEW_VARIABLE_BINDINGS.json").read_text())
    aliases = config.get("field_aliases", {})
    for view in ("front", "top"):
        spec = bindings[view]
        for key, value in list(spec.items()):
            if isinstance(value, dict):
                spec[key] = alias_fields(value, aliases)
        spec["ordered_keys"] = [aliases.get(key, key) for key in spec["ordered_keys"]]
    wrapper = config.get("output_wrapper", "think")
    allow_hash_match = config.get("allow_image_hash_match", False)
    image_hash_cache: dict[Path, str] = {}
    validation_root = (root / config["planner_validation"]).parent
    stages = json.loads((protocol_root / "OUTPUT_SCHEMA_FRONT.json").read_text())["$defs"]["stage_id"]["enum"]
    predictions = indexed(read_jsonl(root / config["prediction_ledger"]), "request_id")
    validation = indexed(read_jsonl(root / config["planner_validation"]), "id")
    canonical = read_jsonl(root / config["canonical_manifest"])
    canonical_index = indexed(canonical, "pair_id")
    if set(validation) != set(canonical_index) or not set(predictions).issubset(canonical_index):
        raise ValueError("input ID sets do not align")
    requests: dict[str, Any] = {}
    for row in read_jsonl(root / config["reference_requests"]):
        if row["request_id"] in canonical_index:
            if row["request_id"] in requests:
                raise ValueError("duplicate reference request")
            row["metric_evidence"] = alias_fields(row["metric_evidence"], aliases)
            requests[row["request_id"]] = row
    if set(requests) != set(canonical_index):
        raise ValueError("missing reference requests")
    if len(canonical) != config["expected_validation_rows"]:
        raise ValueError("unexpected validation count")
    scored = []
    teacher_max_rounding_error = 0.0
    reference_digest = hashlib.sha256()
    for index, canonical_row in enumerate(canonical):
        request_id = canonical_row["pair_id"]
        request, training = requests[request_id], validation[request_id]
        prediction = predictions.get(request_id)
        if request["split"] != "val" or training["planner_metadata"]["cot_version"] != config["cot_version"]:
            raise ValueError(f"wrong split/version: {request_id}")
        if request["source"]["path"] != canonical_row["source_image_path"] or request["target"]["path"] != canonical_row["target_image_path"]:
            raise ValueError(f"image identity mismatch: {request_id}")
        training_image = Path(training["input_path"])
        if not training_image.is_absolute():
            training_image = validation_root / training_image
        if not same_image(training_image, Path(request["source"]["path"]),
                          allow_hash_match=allow_hash_match, cache=image_hash_cache):
            raise ValueError(f"training image identity mismatch: {request_id}")
        if prediction and (not same_image(Path(prediction["input_path"]), training_image,
                                          allow_hash_match=allow_hash_match, cache=image_hash_cache) or prediction["view"] != request["view"] or prediction["cot_version"] != config["cot_version"]):
            raise ValueError(f"prediction identity mismatch: {request_id}")
        if training["messages"][-1]["role"] != "assistant":
            raise ValueError("teacher message not found")
        teacher_text = training["messages"][-1]["content"]
        provider = root / config["teacher_provider_root"]
        annotation_path = provider / "annotations" / f"{request_id}.json"
        rendered_path = provider / "rendered" / f"{request_id}.txt"
        annotation = json.loads(annotation_path.read_text())
        rendered = rendered_path.read_text().strip()
        teacher_body = teacher_text.strip().removeprefix(f"<{wrapper}>").removesuffix(f"</{wrapper}>").strip()
        expected_body = rendered.removeprefix("<GROWTH_COT>").removesuffix("</GROWTH_COT>").strip()
        if bindings[request["view"]].get("numeric_evidence_format") == FORMAT:
            expected_body = name_cot_evidence(expected_body, request["view"], bindings)
        if expected_body != teacher_body:
            raise ValueError(f"teacher renderer differs: {request_id}")
        for role, ref_side in (("current", "source"), ("future", "target")):
            if annotation["stage_reasoning"][f"{role}_stage"] != request[ref_side]["stage_id"]:
                raise ValueError("teacher stage/reference mismatch")
        reference_digest.update(request_id.encode() + b"\0" + annotation_path.read_bytes() + b"\0" + rendered_path.read_bytes())
        teacher = parse_cot(teacher_text, request["view"], bindings, stages, output_wrapper=wrapper)
        if not teacher["structure_valid"]:
            raise ValueError(f"teacher structure invalid: {request_id}: {teacher['issues']}")
        teacher_fields = score_fields(teacher, request, contract)
        for key, field in teacher_fields.items():
            for role, ref in field["reference"].items():
                if ref is None:
                    continue
                pred = field["prediction"][role]
                tolerance = 0 if key == "panicle" else (.010001 if role == "derived_change" else .005001)
                if pred is None or abs(pred - ref) > tolerance:
                    raise ValueError(f"teacher numeric self-check failed: {request_id}/{key}/{role}")
                teacher_max_rounding_error = max(teacher_max_rounding_error, abs(pred - ref))
        raw = prediction[config["text_field"]] if prediction else ""
        parsed = parse_cot(raw, request["view"], bindings, stages,
                           finish_reason=prediction.get("finish_reason", "unknown") if prediction else "missing",
                           output_wrapper=wrapper)
        if config.get("validate_direct_contract", False):
            from ..planner_direct import validate_direct_body
            for text, result in ((teacher_text, teacher), (raw, parsed)):
                try:
                    # Wrapper validity is checked separately and never repaired.
                    body = text.split("<GROWTH_COT>", 1)[-1].split("</GROWTH_COT>", 1)[0]
                    validate_direct_body(body.strip(), request["view"])
                except ValueError as exc:
                    result["issues"].append(f"direct_contract:{exc}")
                    result["structure_valid"] = False
            if not teacher["structure_valid"]:
                raise ValueError(f"teacher direct contract invalid: {request_id}")
        fields = score_fields(parsed, request, contract)
        numeric_issues = [f"numeric:{key}:{role}:{error}" for key, field in fields.items()
                          for role, error in field["parse_errors"].items()
                          if field["reference"][role] is not None and error]
        # Numeric evidence omitted by reference QC is allowed by the rendering contract.
        strict = parsed["structure_valid"] and not numeric_issues
        scored.append({
            "algorithm_version": ALGORITHM_VERSION, "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id, "pair_group_id": request["pair_group_id"], "plant_id": request["plant_id"],
            "row_index": index, "view": request["view"], "actual_interval_days": request["actual_interval_days"],
            "horizon_bucket": canonical_row["horizon_bucket"], "source_path": request["source"]["path"],
            "target_path": request["target"]["path"], "source_stage": request["source"]["stage_id"],
            "target_stage": request["target"]["stage_id"],
            "same_stage_or_cross_stage": "same" if request["source"]["stage_id"] == request["target"]["stage_id"] else "cross",
            "panicle_emergence": fields["panicle"]["reference"]["derived_change"] == 1,
            "input_path": str(root / config["prediction_ledger"]), "output_path": str(output / "per_sample.jsonl"),
            "status": "scored" if prediction else "missing_prediction", "error": None if prediction else "prediction missing",
            "retry_count": 0, "prediction_sha256": text_hash(raw), "reference_sha256": text_hash(json.dumps(request, sort_keys=True)),
            "teacher_sha256": text_hash(teacher_text), "raw_text": raw, "teacher_text": teacher_text,
            "parse_status": "valid" if strict else "partial_or_invalid", "strict_structure_valid": strict,
            "issues": parsed["issues"] + numeric_issues, "parsed": parsed, "teacher_parsed": teacher,
            "numeric": fields, "denominator_eligibility": {key: {role: value is not None for role, value in field["reference"].items()} for key, field in fields.items()},
        })
    if len({r["plant_id"] for r in scored}) != config["expected_plants"] or len({r["pair_group_id"] for r in scored}) != config["expected_pair_groups"]:
        raise ValueError("unexpected plant or pair-group count")
    audit = {"status": "passed", "rows": len(scored), "predictions": len(predictions),
             "missing_predictions": len(canonical) - len(predictions), "teacher_selfcheck_rows": len(scored),
             "teacher_max_rounding_error_pp_including_derived_change": teacher_max_rounding_error,
             "provider_reference_sha256": reference_digest.hexdigest(),
             "input_hashes": {key: sha256_file(root / config[key]) for key in ("prediction_ledger", "planner_validation", "canonical_manifest", "reference_requests", "protocol")},
             "binding_sha256": sha256_file(protocol_root / "VIEW_VARIABLE_BINDINGS.json"),
             "output_wrapper": wrapper, "field_aliases": aliases,
             "image_identity_mode": "path_or_sha256" if allow_hash_match else "exact_path",
             "image_files_hashed": len(image_hash_cache)}
    return scored, audit, contract, stages
