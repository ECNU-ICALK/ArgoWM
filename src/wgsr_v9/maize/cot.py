"""Offline maize sv1 CoT request construction and deterministic rendering.

The provider request schema is kept separate from the operational request
ledger.  The former remains a closed teacher contract; the latter records
status, paths, errors, and retry information needed for resumable runs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Mapping


MAIZE_COT_ALGORITHM_VERSION = "wgsr_maize_sv1_cot_request_renderer_v2"
MAIZE_COT_PROTOCOL_VERSION = "wgsr_maize_sv1_cot_v2_en_v2"
PUBLIC_KEYS = (
    "culm",
    "leaves_area",
    "plant_density",
    "tassel",
    "ear",
    "plant_color_development",
)
METRIC_KEYS = ("culm", "leaves_area", "plant_density", "plant_color_development")
STATUSES = {"available", "provisional", "pending", "invalid", "not_applicable"}
MANUAL_STATUSES = {"visible", "not_visible", "pending"}
CONDITIONS = {"", "枯萎或干枯", "倒伏或残株"}
CONDITION_TRANSLATIONS = {
    "枯萎或干枯": "wilted or dried",
    "倒伏或残株": "lodged or plant remnant",
}
_HAN_TEXT = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
ANNOTATION_QC_VERSION = "wgsr_maize_annotation_qc_v2_no_refusal"
REFUSAL_TEXT = re.compile(
    r"\b(?:I|we)\s+(?:(?:therefore|also|still)\s+)?(?:cannot|can['’]t|am unable|are unable)\b"
    r"|\b(?:refusal|service boundary|ongoing restriction|this service does not support|as an AI)\b",
    re.IGNORECASE,
)
_BANNED_TEXT = re.compile(
    r"\d|[%/\\]|\b(?:pixels?|masks?|coordinates?|LAI|LAD|SPAD|chlorophyll|biomass)\b"
    r"|\b(?:pending|invalid|not_applicable)\b|Source\s*=|Target\s*=|Change\s*="
    r"|\b(?:fertili[sz]e|pesticide|irrigate)\b"
    r"|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+leaves\b"
    r"|[零一二三四五六七八九十百]+[片张枚]?[叶穗]|像素|掩膜|叶面积指数|叶绿素|施肥|灌溉",
    re.IGNORECASE,
)
_METRIC_RANGES = {
    "culm": (0.0, 100.0),
    "leaves_area": (0.0, 100.0),
    "plant_density": (0.0, 1.0),
    "plant_color_development": (0.0, 1.0),
}
_LABELS = {
    "culm": "Culm",
    "leaves_area": "Visible foreground area",
    "plant_density": "Projected fill",
    "tassel": "Tassel",
    "ear": "Ear",
    "plant_color_development": "Visible yellow-brown composition",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def protocol_version(protocol_root: Path) -> str:
    return "wgsr_" + str(read_json(protocol_root / "COT_RENDERER_TEMPLATE.json")["template_version"])


def _stage_description(stage: Mapping[str, Any]) -> str:
    return str(stage.get("visible_description_en", stage.get("visible_description_zh", "")))


def protocol_sha256(protocol_root: Path) -> str:
    """Hash every protocol file by relative name and bytes."""

    digest = hashlib.sha256()
    for path in sorted(item for item in protocol_root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(protocol_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _stages(protocol_root: Path) -> dict[str, dict[str, Any]]:
    value = read_json(protocol_root / "STAGES.json")
    rows = value.get("stages") if isinstance(value, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("maize CoT stage table is empty")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("maize CoT stage row must be an object")
        stage_id = str(row.get("stage_id", "")).strip()
        if not stage_id or stage_id in result:
            raise ValueError(f"duplicate or missing maize stage: {stage_id!r}")
        result[stage_id] = row
    return result


def _bindings(protocol_root: Path) -> dict[str, Any]:
    value = read_json(protocol_root / "VIEW_VARIABLE_BINDINGS.json")
    fields = value.get("fields") if isinstance(value, dict) else None
    if not isinstance(fields, dict):
        raise ValueError("maize CoT bindings have no fields object")
    return fields


def _schema_issues(value: Any, filename: str, protocol_root: Path) -> list[str]:
    """Validate the local JSON-schema subset used by the maize package."""

    cache: dict[Path, Any] = {}
    root = protocol_root.resolve()
    issues: list[str] = []

    def load(name: str) -> Any:
        path = (root / name).resolve()
        if path.parent != root:
            raise ValueError("maize schema references must stay inside the protocol root")
        if path not in cache:
            cache[path] = read_json(path)
        return cache[path]

    def matches(item: Any, kind: str) -> bool:
        if kind == "null":
            return item is None
        if kind == "boolean":
            return isinstance(item, bool)
        if kind == "integer":
            return isinstance(item, int) and not isinstance(item, bool)
        if kind == "number":
            return isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item))
        if kind == "object":
            return isinstance(item, dict)
        if kind == "array":
            return isinstance(item, list)
        if kind == "string":
            return isinstance(item, str)
        return False

    def visit(item: Any, schema: Mapping[str, Any], document: Any, location: str) -> None:
        reference = schema.get("$ref")
        if reference:
            filename_ref, _, fragment = str(reference).partition("#")
            referenced_document = load(filename_ref) if filename_ref else document
            resolved: Any = referenced_document
            for part in fragment.strip("/").split("/") if fragment else []:
                resolved = resolved[part.replace("~1", "/").replace("~0", "~")]
            visit(item, resolved, referenced_document, location)
            return
        kinds = schema.get("type")
        if kinds:
            allowed = [kinds] if isinstance(kinds, str) else kinds
            if not any(matches(item, kind) for kind in allowed):
                issues.append(f"{location}: expected {kinds}")
                return
        if "enum" in schema and item not in schema["enum"]:
            issues.append(f"{location}: invalid enum value")
        if "const" in schema and item != schema["const"]:
            issues.append(f"{location}: unexpected constant")
        if isinstance(item, str) and len(item) < int(schema.get("minLength", 0)):
            issues.append(f"{location}: text too short")
        if isinstance(item, dict):
            for key in schema.get("required", []):
                if key not in item:
                    issues.append(f"{location}: missing {key}")
            properties = schema.get("properties", {})
            for key, child in item.items():
                if key in properties:
                    visit(child, properties[key], document, f"{location}.{key}")
                elif schema.get("additionalProperties") is False:
                    issues.append(f"{location}: extra {key}")
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            if not math.isfinite(float(item)):
                issues.append(f"{location}: non-finite number")
            if float(item) < float(schema.get("minimum", -math.inf)):
                issues.append(f"{location}: below minimum")
            if float(item) > float(schema.get("maximum", math.inf)):
                issues.append(f"{location}: above maximum")
        if isinstance(item, list):
            if schema.get("uniqueItems") and any(item[i] in item[:i] for i in range(len(item))):
                issues.append(f"{location}: duplicate items")
            child_schema = schema.get("items")
            if isinstance(child_schema, dict):
                for index, child in enumerate(item):
                    visit(child, child_schema, document, f"{location}[{index}]")

    document = load(filename)
    visit(value, document, document, "$")
    return issues


def _resolved_path(value: Any, repo_root: Path) -> Path:
    if value in (None, ""):
        raise ValueError("maize request image path is missing")
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _portable_or_absolute(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _endpoint(pair_endpoint: Mapping[str, Any], stages: Mapping[str, Mapping[str, Any]], *, repo_root: Path, check_images: bool) -> dict[str, Any]:
    stage_id = str(pair_endpoint.get("stage_id", ""))
    stage = stages.get(stage_id)
    if stage is None:
        raise ValueError(f"unknown maize stage in pair endpoint: {stage_id}")
    image_id = str(pair_endpoint.get("image_id", ""))
    plant_id = str(pair_endpoint.get("plant_id", ""))
    if image_id != f"{plant_id}/{stage_id}_sv1":
        raise ValueError(f"invalid maize image identity: {image_id}")
    image_path = _resolved_path(pair_endpoint.get("path"), repo_root)
    if check_images and not image_path.is_file():
        raise FileNotFoundError(image_path)
    image_sha256 = str(pair_endpoint.get("image_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", image_sha256):
        raise ValueError(f"malformed image hash for {image_id}")
    return {
        "image_id": image_id,
        "plant_id": plant_id,
        "stage_id": stage_id,
        "days_after_sowing": int(stage["days_after_sowing"]),
        "maize_stage": str(stage["maize_stage"]),
        "stage_description": _stage_description(stage),
        "path": _portable_or_absolute(image_path, repo_root),
        "image_sha256": image_sha256,
        "plant_condition": (
            CONDITION_TRANSLATIONS.get(pair_endpoint.get("plant_condition"), pair_endpoint.get("plant_condition"))
            if "visible_description_en" in stage else pair_endpoint.get("plant_condition")
        ) or None,
    }


def _manual_value(value: Any) -> str:
    result = str(value or "pending")
    return result if result in MANUAL_STATUSES else "pending"


def _observation(value: Mapping[str, Any] | None) -> dict[str, Any]:
    value = value or {}
    status = str(value.get("status", "pending"))
    if status not in STATUSES:
        status = "invalid"
    observation_value = value.get("value") if status in {"available", "provisional"} else None
    return {
        "value": observation_value,
        "status": status,
        "image_sha256": value.get("image_sha256"),
        "mask_sha256": value.get("mask_sha256"),
    }


def _request_metric(key: str, pair_metric: Mapping[str, Any] | None, bindings: Mapping[str, Any]) -> dict[str, Any]:
    pair_metric = pair_metric or {}
    source = _observation(pair_metric.get("source") if isinstance(pair_metric.get("source"), Mapping) else None)
    target = _observation(pair_metric.get("target") if isinstance(pair_metric.get("target"), Mapping) else None)
    versions = sorted(
        {
            str(item.get("algorithm_version"))
            for item in (
                pair_metric.get("source", {}),
                pair_metric.get("target", {}),
            )
            if isinstance(item, Mapping) and item.get("algorithm_version")
        }
    )
    reasons: set[str] = set()
    for item in (pair_metric.get("source"), pair_metric.get("target")):
        if isinstance(item, Mapping):
            reasons.update(str(reason) for reason in item.get("reason_codes", []))
    return {
        "metric_key": str(bindings[key]["metric_key"]),
        "source": source,
        "target": target,
        "unit": str(bindings[key]["unit"]),
        "algorithm_version": ",".join(versions) or "unavailable",
        "qc": {
            "valid": source["status"] == "available" and target["status"] == "available",
            "reason_codes": sorted(reasons),
        },
    }


def build_request(
    pair: Mapping[str, Any],
    *,
    protocol_root: Path,
    repo_root: Path,
    check_images: bool = True,
) -> dict[str, Any]:
    """Convert one species-specific pair row into a strict provider request."""

    if pair.get("species") != "maize" or pair.get("view") != "sv1":
        raise ValueError("maize CoT requests require species=maize and view=sv1")
    stages = _stages(protocol_root)
    bindings = _bindings(protocol_root)
    source = _endpoint(pair.get("source", {}), stages, repo_root=repo_root, check_images=check_images)
    target = _endpoint(pair.get("target", {}), stages, repo_root=repo_root, check_images=check_images)
    if source["plant_id"] != target["plant_id"]:
        raise ValueError("maize CoT pair endpoints must have the same plant_id")
    source_stage = int(source["stage_id"][1:])
    target_stage = int(target["stage_id"][1:])
    if target_stage <= source_stage:
        raise ValueError("maize CoT target stage must follow source stage")
    actual_interval_days = int(target["days_after_sowing"]) - int(source["days_after_sowing"])
    if actual_interval_days <= 0:
        raise ValueError("maize CoT interval must be positive")
    manual = pair.get("manual_summary", {})
    manual_summary = {
        organ: {
            "source": _manual_value(manual.get(organ, {}).get("source")),
            "target": _manual_value(manual.get(organ, {}).get("target")),
        }
        for organ in ("tassel", "ear")
    }
    metric_evidence = {
        key: _request_metric(key, pair.get("metric_evidence", {}).get(key), bindings)
        for key in METRIC_KEYS
    }
    return {
        "request_id": str(pair.get("pair_id") or f"{source['plant_id']}__{source['stage_id']}__{target['stage_id']}__sv1"),
        "view": "sv1",
        "protocol_sha256": protocol_sha256(protocol_root),
        "source": source,
        "target": target,
        "actual_interval_days": actual_interval_days,
        "prediction_interval_weeks": actual_interval_days / 7.0,
        "stage_delta": target_stage - source_stage,
        "metric_evidence": metric_evidence,
        "manual_summary": manual_summary,
    }


def validate_request(request: Mapping[str, Any], *, protocol_root: Path, repo_root: Path, check_images: bool = True) -> list[str]:
    """Validate schema, stage identity, hashes, and local metric provenance."""

    issues = _schema_issues(dict(request), "PAIR_REQUEST_SCHEMA.json", protocol_root)
    if issues:
        return issues
    stages = _stages(protocol_root)
    bindings = _bindings(protocol_root)
    if request.get("view") != "sv1":
        issues.append("request view must be sv1")
    if request.get("protocol_sha256") != protocol_sha256(protocol_root):
        issues.append("protocol fingerprint mismatch")
    endpoints: dict[str, Mapping[str, Any]] = {}
    for side in ("source", "target"):
        endpoint = request[side]
        endpoints[side] = endpoint
        stage_id = str(endpoint["stage_id"])
        stage = stages.get(stage_id)
        if stage is None:
            issues.append(f"{side}: unknown stage {stage_id}")
            continue
        for field in ("days_after_sowing", "maize_stage"):
            if endpoint[field] != stage[field]:
                issues.append(f"{side}: stage metadata mismatch for {field}")
        if endpoint["stage_description"] != _stage_description(stage):
            issues.append(f"{side}: stage description mismatch")
        if endpoint["image_id"] != f"{endpoint['plant_id']}/{stage_id}_sv1":
            issues.append(f"{side}: image identity mismatch")
        condition = endpoint.get("plant_condition")
        allowed_conditions = set(CONDITION_TRANSLATIONS.values()) if "visible_description_en" in stage else CONDITIONS - {""}
        if condition is not None and condition not in allowed_conditions:
            issues.append(f"{side}: invalid plant_condition")
        if not re.fullmatch(r"[0-9a-f]{64}", str(endpoint["image_sha256"])):
            issues.append(f"{side}: malformed image hash")
        if check_images:
            try:
                if not _resolved_path(endpoint["path"], repo_root).is_file():
                    issues.append(f"{side}: image path does not exist")
            except ValueError as exc:
                issues.append(f"{side}: {exc}")
    source, target = endpoints["source"], endpoints["target"]
    if source["plant_id"] != target["plant_id"]:
        issues.append("source and target must be the same plant")
    delta = int(target["stage_id"][1:]) - int(source["stage_id"][1:])
    days = int(target["days_after_sowing"]) - int(source["days_after_sowing"])
    expected_request_id = f"{source['plant_id']}__{source['stage_id']}__{target['stage_id']}__sv1"
    if request["request_id"] != expected_request_id:
        issues.append("request_id must match same-plant endpoint identity")
    if delta <= 0 or request["stage_delta"] != delta:
        issues.append("stage_delta mismatch")
    if request["actual_interval_days"] != days or days <= 0:
        issues.append("actual_interval_days mismatch")
    if not math.isclose(float(request["prediction_interval_weeks"]), days / 7.0, rel_tol=1e-12):
        issues.append("prediction_interval_weeks mismatch")
    for key in METRIC_KEYS:
        metric = request["metric_evidence"][key]
        if metric["metric_key"] != bindings[key]["metric_key"] or metric["unit"] != bindings[key]["unit"]:
            issues.append(f"{key}: metric binding mismatch")
        observations = [metric["source"], metric["target"]]
        if metric["qc"]["valid"] != all(item["status"] == "available" for item in observations):
            issues.append(f"{key}: inconsistent qc.valid")
        lower, upper = _METRIC_RANGES[key]
        for side, observation in zip(("source", "target"), observations):
            value = observation["value"]
            status = observation["status"]
            if status in {"available", "provisional"}:
                if value is None or not isinstance(value, (int, float)) or not lower <= float(value) <= upper:
                    issues.append(f"{key}.{side}: missing or out-of-range metric")
                if observation["image_sha256"] != request[side]["image_sha256"]:
                    issues.append(f"{key}.{side}: stale metric image hash")
            elif value is not None:
                issues.append(f"{key}.{side}: unavailable value must be null")
    return issues


def organ_text(organ: str, status: str, side: str | None = None) -> str:
    if status not in MANUAL_STATUSES:
        status = "pending"
    subject = f"{side.capitalize()} {organ}" if side else f"The {organ}"
    suffix = {
        "visible": "is visible",
        "not_visible": "is not visible",
        "pending": "visibility has not been manually established",
    }[status]
    return f"{subject} {suffix}."


def _manual_annotation(request: Mapping[str, Any], organ: str, section: str) -> str:
    evidence = request["manual_summary"][organ]
    if section == "target_phenotype":
        return organ_text(organ, str(evidence["target"]))
    return " ".join(organ_text(organ, str(evidence[side]), side) for side in ("source", "target"))


def validate_annotation(annotation: Mapping[str, Any], request: Mapping[str, Any], *, protocol_root: Path, repo_root: Path, check_images: bool = True) -> list[str]:
    issues = validate_request(request, protocol_root=protocol_root, repo_root=repo_root, check_images=check_images)
    issues.extend(_schema_issues(dict(annotation), "OUTPUT_SCHEMA_FRONT.json", protocol_root))
    if issues:
        return issues
    if list(annotation) != ["stage_reasoning", "growth_plan", "target_phenotype"]:
        issues.append("annotation root order mismatch")
    source_stage = request["source"]["stage_id"]
    target_stage = request["target"]["stage_id"]
    if annotation["stage_reasoning"] != {"current_stage": source_stage, "future_stage": target_stage}:
        issues.append("annotation stages must match request")
    english_only = read_json(protocol_root / "TEACHER_PROMPT.json").get("output_language") == "en"
    for section in ("growth_plan", "target_phenotype"):
        if list(annotation[section]) != list(PUBLIC_KEYS):
            issues.append(f"{section}: field order mismatch")
        for key in PUBLIC_KEYS:
            entry = annotation[section].get(key)
            text = entry.get("text") if isinstance(entry, Mapping) else None
            if not isinstance(text, str) or not text.strip() or _BANNED_TEXT.search(text):
                issues.append(f"{section}.{key}: forbidden or empty text")
            if isinstance(text, str) and (REFUSAL_TEXT.search(text) or "\n" in text or "\r" in text):
                issues.append(f"{section}.{key}: refusal or multiline meta-response instead of phenotype text")
            if english_only and isinstance(text, str) and _HAN_TEXT.search(text):
                issues.append(f"{section}.{key}: Chinese text is not allowed in the English protocol")
            if key in {"tassel", "ear"} and text != _manual_annotation(request, key, section):
                issues.append(f"{section}.{key}: manual visibility text mismatch")
    return issues


def _display(value: Decimal) -> str:
    rounded = value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return f"{abs(rounded) if rounded == 0 else rounded:.1f}"


def _numeric_evidence(key: str, metric: Mapping[str, Any], *, include_provisional: bool) -> str:
    if not isinstance(metric, Mapping) or metric.get("qc", {}).get("valid") is not True:
        return ""
    source = metric.get("source", {})
    target = metric.get("target", {})
    if not isinstance(source, Mapping) or not isinstance(target, Mapping):
        return ""
    statuses = (source.get("status"), target.get("status"))
    if any(status not in {"available", "provisional"} for status in statuses):
        return ""
    if "provisional" in statuses and not include_provisional:
        return ""
    try:
        source_value = Decimal(str(source["value"]))
        target_value = Decimal(str(target["value"]))
    except (KeyError, TypeError, ValueError):
        return ""
    if metric.get("unit") == "fraction":
        source_value *= Decimal(100)
        target_value *= Decimal(100)
    change = target_value - source_value
    if key == "culm" and source_value != 0 and abs(change / source_value) < Decimal("0.02"):
        return "Stable within the registered two-percent proxy tolerance; Source/Target/Change suppressed."
    prefix = "Diagnostic provisional evidence; " if "provisional" in statuses else ""
    return f"{prefix}Source={_display(source_value)}%; Target={_display(target_value)}%; Change={_display(change)} pp."


def render_cot(annotation: Mapping[str, Any], request: Mapping[str, Any], *, protocol_root: Path, repo_root: Path, include_provisional: bool = False, check_images: bool = True) -> str:
    """Render validated annotation text and locally owned numeric evidence."""

    issues = validate_annotation(annotation, request, protocol_root=protocol_root, repo_root=repo_root, check_images=check_images)
    if issues:
        raise ValueError("maize annotation validation failed: " + "; ".join(issues))
    template = read_json(protocol_root / "COT_RENDERER_TEMPLATE.json")
    lines = [template["wrapper"][0], template["sections"][0]]
    labels = annotation["stage_reasoning"]
    lines.extend(line.format_map(labels) for line in template["stage_reasoning_lines"])
    lines.append(template["sections"][1])
    for key in PUBLIC_KEYS:
        text = str(annotation["growth_plan"][key]["text"]).strip().rstrip(".")
        evidence = _numeric_evidence(key, request["metric_evidence"].get(key, {}), include_provisional=include_provisional)
        pattern = template["growth_plan_line"]
        line = pattern.format(section_label=_LABELS[key], text=text, numeric_evidence=evidence).rstrip()
        lines.append(line)
    lines.append(template["sections"][2])
    for key in PUBLIC_KEYS:
        text = str(annotation["target_phenotype"][key]["text"]).strip().rstrip(".")
        lines.append(template["target_phenotype_line"].format(section_label=_LABELS[key], text=text).rstrip())
    lines.append(template["wrapper"][1])
    return "\n".join(lines) + "\n"


def build_provider_input(request: Mapping[str, Any], *, protocol_root: Path, repo_root: Path, check_images: bool = True) -> dict[str, Any]:
    """Build an inspectable multimodal prompt bundle without contacting a provider."""

    issues = validate_request(request, protocol_root=protocol_root, repo_root=repo_root, check_images=check_images)
    if issues:
        raise ValueError("maize request validation failed: " + "; ".join(issues))
    prompt = read_json(protocol_root / "TEACHER_PROMPT.json")
    metric_summary: dict[str, Any] = {}
    for key in METRIC_KEYS:
        metric = request["metric_evidence"][key]
        trusted = metric["qc"]["valid"] and all(metric[side]["status"] == "available" for side in ("source", "target"))
        metric_summary[key] = (
            {"source": metric["source"]["value"], "target": metric["target"]["value"], "unit": metric["unit"]}
            if trusted
            else {"status": "unavailable"}
        )
    dump = lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    source, target = request["source"], request["target"]
    placeholders = {
        "actual_interval_days": request["actual_interval_days"],
        "prediction_interval_weeks": request["prediction_interval_weeks"],
        "stage_delta": request["stage_delta"],
        "source_stage": source["stage_id"],
        "target_stage": target["stage_id"],
        "source_das": source["days_after_sowing"],
        "target_das": target["days_after_sowing"],
        "source_stage_description": source["stage_description"],
        "target_stage_description": target["stage_description"],
        "metric_summary": dump(metric_summary),
        "manual_summary": dump(request["manual_summary"]),
        "plant_condition_summary": dump({side: request[side]["plant_condition"] for side in ("source", "target")}),
    }
    user_prompt = "\n".join(line.format_map(placeholders) for line in prompt["user_prompt_lines"])
    canonical = {
        section: {
            organ: {"text": _manual_annotation(request, organ, section)}
            for organ in ("tassel", "ear")
        }
        for section in ("growth_plan", "target_phenotype")
    }
    user_prompt += "\nCopy these deterministic manual organ text entries exactly into the matching sections: " + dump(canonical)
    return {
        "request_id": request["request_id"],
        "protocol_sha256": request["protocol_sha256"],
        "transport": "offline_prompt_bundle_not_api_request",
        "provider_called": False,
        "system_prompt": "\n".join([
            *prompt["system_prompt_lines"],
            "Strict output rules:",
            *prompt.get("strict_output_rules", []),
        ]),
        "user_prompt": user_prompt,
        "required_organ_text": canonical,
        "images": [
            {
                "role": f"{side.upper()}_IMAGE",
                "path": str(_resolved_path(request[side]["path"], repo_root)),
                "sha256": request[side]["image_sha256"],
            }
            for side in ("source", "target")
        ],
        "output_schema": read_json(protocol_root / "OUTPUT_SCHEMA_FRONT.json"),
    }


def request_ledger_row(request: Mapping[str, Any], *, input_path: Path, output_path: Path, status: str = "ready", error: str | None = None, protocol_root: Path | None = None) -> dict[str, Any]:
    """Return operational metadata for one strict request row."""

    return {
        "algorithm_version": MAIZE_COT_ALGORITHM_VERSION,
        "protocol_version": protocol_version(protocol_root) if protocol_root else MAIZE_COT_PROTOCOL_VERSION,
        "request_id": request.get("request_id"),
        "input_path": str(input_path.resolve()),
        "output_path": str(output_path.resolve()),
        "status": status,
        "error": error,
        "retry_count": 0,
        "provider_calls": False,
        "training_started": False,
        "inference_started": False,
    }
