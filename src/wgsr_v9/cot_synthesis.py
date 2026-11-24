"""Deterministic validation and rendering for the WGSR V9 GPT contract."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ROOT = REPO_ROOT / "protocols/qwen_image21_executor_v9_v3_metric_named_weeks/cot"
_COLOR_PROXY_PUBLIC_KEYS = {"yellow_brown_fraction", "plant_color_development"}
_COLOR_PROXY_METRIC_KEYS = {"yellow_brown_fraction"}
_STATUSES = {"available", "pending", "invalid", "not_applicable"}
_STAGE_ORDER = (
    "S1_tillering",
    "S2_jointing",
    "S3_booting",
    "S4_heading_flowering",
    "S5_early_grain_filling",
    "S6_milk_waxy",
    "S7_yellow_ripening",
    "S8_maturity_senescence",
)
_STAGE_ALIASES = {
    "s1tillering": "S1_tillering",
    "分蘖期": "S1_tillering",
    "tillering": "S1_tillering",
    "s2jointing": "S2_jointing",
    "拔节期": "S2_jointing",
    "jointing": "S2_jointing",
    "stemelongationjointing": "S2_jointing",
    "s3booting": "S3_booting",
    "孕穗期": "S3_booting",
    "booting": "S3_booting",
    "s4headingflowering": "S4_heading_flowering",
    "抽穗开花期": "S4_heading_flowering",
    "headingflowering": "S4_heading_flowering",
    "heading": "S4_heading_flowering",
    "s5earlygrainfilling": "S5_early_grain_filling",
    "灌浆期": "S5_early_grain_filling",
    "grainfilling": "S5_early_grain_filling",
    "earlygrainfilling": "S5_early_grain_filling",
    "s6milkwaxy": "S6_milk_waxy",
    "乳熟蜡熟期": "S6_milk_waxy",
    "milkdoughripening": "S6_milk_waxy",
    "milkwaxy": "S6_milk_waxy",
    "s7yellowripening": "S7_yellow_ripening",
    "黄熟期": "S7_yellow_ripening",
    "yellowripening": "S7_yellow_ripening",
    "s8maturitysenescence": "S8_maturity_senescence",
    "成熟衰老期": "S8_maturity_senescence",
    "maturitysenescence": "S8_maturity_senescence",
}
_TEXT_BANNED_RE = re.compile(
    r"(?:%|\bpp\b|source\s*=|target\s*=|change\s*=|[\\/]"
    r"|\b(?:pixel|pixels|mask|coordinate|coordinates|bottom_gap|boundary_contact)\b"
    r"|\b(?:s_color|canopy/architecture|coverage|pending|invalid|not_applicable)\b)",
    re.IGNORECASE,
)
_TEXT_NUMBER_RE = re.compile(r"(?<![A-Za-z])[+-]?\d+(?:\.\d+)?(?:\s*(?:%|pp))?")
_EVIDENCE_PHRASES = {
    "plant remains inside frame",
    "plant reaches image edge",
    "one-sided lean is visible",
    "off-center projected plant is visible",
    "possible truncation is visible",
    "frame completeness is uncertain",
    "no obvious tilt or clipping",
}
_SIDES = {"top", "bottom", "left", "right"}


def _load_registry(protocol_root: Path | None = None) -> dict[str, Any]:
    path = (protocol_root or PROTOCOL_ROOT) / "VIEW_VARIABLE_BINDINGS.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _view_spec(view: str, protocol_root: Path | None = None) -> tuple[str, Mapping[str, Any]]:
    normalized = normalize_view(view)
    return normalized, _load_registry(protocol_root)[normalized]


def _legacy_annotation_contract(protocol_root: Path | None = None) -> bool:
    """Return whether a protocol snapshot still uses the v1 annotation fields."""

    root = protocol_root or PROTOCOL_ROOT
    try:
        schema = json.loads((root / "OUTPUT_SCHEMA_FRONT.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Keep the historical default safe when a caller supplies an incomplete
        # protocol root; the API runner will report the missing schema separately.
        return True
    required = schema.get("required", [])
    return "image_quality_review" in required


def normalize_view(raw_view: str) -> Literal["front", "top"]:
    """Normalize the view labels accepted by pair manifests."""

    value = str(raw_view).strip().lower()
    if value in {"side0", "side90", "front", "front_view"}:
        return "front"
    if value in {"top", "top_view"}:
        return "top"
    raise ValueError(f"Unknown view label: {raw_view!r}")


def _stage_alias_key(value: str) -> str:
    text = str(value).strip().lower()
    if text in _STAGE_ALIASES:
        return text
    return re.sub(r"[\s_\-/]+", "", text)


def stage_id_from_label(value: str) -> str:
    """Return the canonical V9 stage ID for an English or Chinese label."""

    key = _stage_alias_key(value)
    try:
        return _STAGE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown stage label: {value!r}") from exc


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _finite_or_none(value: Any) -> bool:
    return value is None or _is_number(value)


def _metric_range(key: str, value: Any) -> bool:
    if value is None:
        return True
    if not _is_number(value):
        return False
    if key in {"culm", "leaves_area"}:
        return 0.0 <= float(value) <= 100.0
    if key in {"plant_density", "canopy"} or is_color_proxy_key(key):
        return 0.0 <= float(value) <= 1.0
    if key == "panicle":
        return isinstance(value, int) and not isinstance(value, bool) and value in {0, 1}
    return False


def is_color_proxy_key(public_key: str, metric_key: str | None = None) -> bool:
    """Return whether a public or local metric key is the plant-color proxy."""

    return public_key in _COLOR_PROXY_PUBLIC_KEYS or metric_key in _COLOR_PROXY_METRIC_KEYS


def _optional_color_value(container: Mapping[str, Any], name: str) -> Any:
    for candidate in (name, f"raw_{name}"):
        if candidate in container:
            return container[candidate]
    return None


def _validate_color_fractions(metric: Mapping[str, Any], issues: list[str]) -> None:
    containers: list[Mapping[str, Any]] = []
    for key in ("source", "target", "qc"):
        value = metric.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
    for container in containers:
        yellow = _optional_color_value(container, "yellow_fraction")
        brown = _optional_color_value(container, "brown_fraction")
        if yellow is None and brown is None:
            continue
        if yellow is not None and (not _is_number(yellow) or not 0 <= float(yellow) <= 1):
            issues.append("yellow_fraction must be finite and in [0, 1]")
        if brown is not None and (not _is_number(brown) or not 0 <= float(brown) <= 1):
            issues.append("brown_fraction must be finite and in [0, 1]")
        if _is_number(yellow) and _is_number(brown) and float(yellow) + float(brown) > 1:
            issues.append("yellow_fraction + brown_fraction must be <= 1")


def validate_metric_evidence(
    view: str,
    evidence: Mapping[str, Any],
    *,
    protocol_root: Path | None = None,
) -> list[str]:
    """Validate a view's complete metric evidence and return readable issues."""

    issues: list[str] = []
    try:
        normalized, spec = _view_spec(view, protocol_root)
    except (TypeError, ValueError) as exc:
        return [str(exc)]
    if not isinstance(evidence, Mapping):
        return ["metric evidence must be an object"]
    expected_keys = list(spec["ordered_keys"])
    actual_keys = list(evidence.keys())
    for key in expected_keys:
        if key not in evidence:
            issues.append(f"missing metric evidence: {key}")
    for key in actual_keys:
        if key not in expected_keys:
            issues.append(f"unexpected metric evidence: {key}")
    for key in expected_keys:
        metric = evidence.get(key)
        if not isinstance(metric, Mapping):
            issues.append(f"{key} must be an object")
            continue
        required = {"metric_key", "source", "target", "unit", "algorithm_version", "qc"}
        missing = required - set(metric)
        extra = set(metric) - required
        if missing:
            issues.append(f"{key} missing fields: {sorted(missing)}")
        if extra:
            issues.append(f"{key} has unexpected fields: {sorted(extra)}")
        expected_metric_key = spec["metric_keys"][key]
        if metric.get("metric_key") != expected_metric_key:
            issues.append(f"{key}.metric_key must be {expected_metric_key}")
        if not isinstance(metric.get("unit"), str) or not metric.get("unit"):
            issues.append(f"{key}.unit must be non-empty")
        elif metric.get("unit") != spec["display_units"][key]:
            issues.append(f"{key}.unit must be {spec['display_units'][key]}")
        if not isinstance(metric.get("algorithm_version"), str) or not metric.get("algorithm_version"):
            issues.append(f"{key}.algorithm_version must be non-empty")
        qc = metric.get("qc")
        if not isinstance(qc, Mapping):
            issues.append(f"{key}.qc must be an object")
        else:
            allowed_qc = {"valid", "reason_codes", "color_coverage", "boundary_contact", "bottom_gap"}
            unexpected_qc = set(qc) - allowed_qc
            if unexpected_qc:
                issues.append(f"{key}.qc has unexpected fields: {sorted(unexpected_qc)}")
            if not isinstance(qc.get("valid"), bool):
                issues.append(f"{key}.qc.valid must be boolean")
            if not isinstance(qc.get("reason_codes"), list) or not all(isinstance(item, str) for item in qc.get("reason_codes", [])):
                issues.append(f"{key}.qc.reason_codes must be a string list")
            coverage = qc.get("color_coverage")
            if coverage is not None and (not _is_number(coverage) or not 0 <= float(coverage) <= 1):
                issues.append(f"{key}.qc.color_coverage must be in [0, 1]")
        for side in ("source", "target"):
            observation = metric.get(side)
            if not isinstance(observation, Mapping):
                issues.append(f"{key}.{side} must be an object")
                continue
            missing_obs = {"value", "status"} - set(observation)
            extra_obs = set(observation) - {"value", "status"}
            if missing_obs:
                issues.append(f"{key}.{side} missing fields: {sorted(missing_obs)}")
            if extra_obs:
                issues.append(f"{key}.{side} has unexpected fields: {sorted(extra_obs)}")
            value = observation.get("value")
            status = observation.get("status")
            if not _finite_or_none(value):
                issues.append(f"{key}.{side}.value must be finite or null")
            if status not in _STATUSES:
                issues.append(f"{key}.{side}.status is invalid")
            if not _metric_range(key, value):
                issues.append(f"{key}.{side}.value is outside the allowed range")
            if status == "available" and value is None:
                issues.append(f"{key}.{side} available value cannot be null")
            if key == "panicle" and value is not None and not _metric_range(key, value):
                issues.append(f"{key}.{side} panicle value must be integer 0 or 1")
        if is_color_proxy_key(key, str(metric.get("metric_key", ""))) and isinstance(qc, Mapping):
            coverage = qc.get("color_coverage")
            if _is_number(coverage) and float(coverage) < 0.70:
                for side in ("source", "target"):
                    if isinstance(metric.get(side), Mapping) and metric[side].get("status") == "available":
                        issues.append(f"{key}.{side} must be pending when color coverage is below 0.70")
        _validate_color_fractions(metric, issues)
    return issues


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _group_is_clean(group: Mapping[str, Any]) -> bool:
    if group.get("quarantined") is True or group.get("excluded") is True:
        return False
    if group.get("is_clean") is False:
        return False
    for key in ("clean_status", "status", "qc_status"):
        if key in group:
            value = str(group[key]).strip().lower()
            return value in {"clean", "ok", "eligible", "available", "keep"}
    return True


def _lookup_group(groups: Mapping[str, Mapping[str, Any]], identifier: Any) -> Mapping[str, Any] | None:
    if identifier is None:
        return None
    value = str(identifier)
    return groups.get(value) or groups.get(identifier)


def validate_pair_metadata(
    row: Mapping[str, Any],
    *,
    clean_groups: Mapping[str, Mapping[str, Any]],
    panicle_groups: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Validate the hard QC fields needed before sending a pair to GPT."""

    issues: list[str] = []
    if not isinstance(row, Mapping):
        return ["pair row must be an object"]
    try:
        view = normalize_view(_first(row, "view", "view_label"))
    except (TypeError, ValueError) as exc:
        issues.append(str(exc))
        view = "front"
    for label, keys in {
        "plant identity": ("plant_id", "plant", "single_plant_id"),
        "accession identity": ("accession_id", "accession", "canonical_accession_id"),
    }.items():
        if _first(row, *keys) in (None, ""):
            issues.append(f"missing {label}")
    source_group_id = _first(row, "source_group_id", "source_group")
    target_group_id = _first(row, "target_group_id", "target_group")
    if source_group_id in (None, "") or target_group_id in (None, ""):
        issues.append("missing source/target group identity")
    source_group = _lookup_group(clean_groups, source_group_id)
    target_group = _lookup_group(clean_groups, target_group_id)
    if source_group is None:
        issues.append("source group is missing from clean manifest")
    elif not _group_is_clean(source_group):
        issues.append("source group is not clean")
    if target_group is None:
        issues.append("target group is missing from clean manifest")
    elif not _group_is_clean(target_group):
        issues.append("target group is not clean")
    source_path = _first(row, "source_path", "source_image_path")
    target_path = _first(row, "target_path", "target_image_path")
    if source_path in (None, "") or target_path in (None, ""):
        issues.append("missing source/target image path")
    else:
        source_text = str(source_path)
        target_text = str(target_path)
        if source_text == target_text:
            issues.append("source and target image paths must differ")
        for label, path_value in (("source", source_path), ("target", target_path)):
            if not Path(str(path_value)).is_file():
                issues.append(f"{label} image path does not exist")
    source_date = _parse_date(_first(row, "source_date", "source_capture_date"))
    target_date = _parse_date(_first(row, "target_date", "target_capture_date"))
    if source_date is None or target_date is None:
        issues.append("source and target dates must be valid")
    elif source_date >= target_date:
        issues.append("source date must precede target date")
    source_stage_value = _first(row, "source_stage_id", "source_stage", "source_stage_en")
    target_stage_value = _first(row, "target_stage_id", "target_stage", "target_stage_en")
    try:
        source_stage = stage_id_from_label(source_stage_value)
        target_stage = stage_id_from_label(target_stage_value)
        if _STAGE_ORDER.index(source_stage) > _STAGE_ORDER.index(target_stage):
            issues.append("target stage must not precede source stage")
    except (TypeError, ValueError) as exc:
        issues.append(str(exc))
    for group_id in (source_group_id, target_group_id):
        group = _lookup_group(panicle_groups, group_id)
        if group is None:
            continue
        value = _first(group, "panicle_visibility", "panicle", "value")
        if value is not None and not _metric_range("panicle", value):
            issues.append(f"panicle visibility for {group_id} must be 0 or 1")
    source_panicle = _first(row, "source_panicle_visibility", "source_panicle")
    target_panicle = _first(row, "target_panicle_visibility", "target_panicle")
    if source_panicle is not None and not _metric_range("panicle", source_panicle):
        issues.append("source panicle visibility must be 0 or 1")
    if target_panicle is not None and not _metric_range("panicle", target_panicle):
        issues.append("target panicle visibility must be 0 or 1")
    if source_panicle == 1 and target_panicle == 0:
        issues.append("panicle visibility cannot reverse from 1 to 0")
    if view == "top" and "culm" in row:
        issues.append("top pair must not contain culm metric")
    return issues


def _clean_text_issues(text: Any, location: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return [f"{location} text must be non-empty"]
    issues: list[str] = []
    if _TEXT_NUMBER_RE.search(text) or _TEXT_BANNED_RE.search(text):
        issues.append(f"{location} contains numeric or renderer-owned text")
    if re.search(r"\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b", text):
        issues.append(f"{location} contains a date")
    return issues


def _validate_quality_item(item: Any, location: str) -> list[str]:
    issues: list[str] = []
    if not isinstance(item, Mapping):
        return [f"{location} must be an object"]
    expected = {"tilt", "clipping", "tilt_clipping_risk", "review_recommendation", "evidence"}
    if set(item) != expected:
        issues.append(f"{location} keys must be {sorted(expected)}")
    tilt = item.get("tilt")
    if not isinstance(tilt, Mapping) or set(tilt) != {"label", "confidence"}:
        issues.append(f"{location}.tilt has invalid shape")
    else:
        if tilt.get("label") not in {"none", "mild", "marked", "uncertain"}:
            issues.append(f"{location}.tilt.label is invalid")
        if tilt.get("confidence") not in {"high", "medium", "low"}:
            issues.append(f"{location}.tilt.confidence is invalid")
    clipping = item.get("clipping")
    if not isinstance(clipping, Mapping) or set(clipping) != {"label", "sides", "confidence"}:
        issues.append(f"{location}.clipping has invalid shape")
    else:
        if clipping.get("label") not in {"none", "minor", "substantial", "uncertain"}:
            issues.append(f"{location}.clipping.label is invalid")
        sides = clipping.get("sides")
        if (
            not isinstance(sides, list)
            or not all(isinstance(side, str) for side in sides)
            or len(sides) != len(set(sides))
            or not set(sides).issubset(_SIDES)
        ):
            issues.append(f"{location}.clipping.sides is invalid")
        if clipping.get("confidence") not in {"high", "medium", "low"}:
            issues.append(f"{location}.clipping.confidence is invalid")
    if item.get("tilt_clipping_risk") not in {"low", "medium", "high", "uncertain"}:
        issues.append(f"{location}.tilt_clipping_risk is invalid")
    if item.get("review_recommendation") not in {"keep", "manual_review", "exclude_candidate"}:
        issues.append(f"{location}.review_recommendation is invalid")
    evidence = item.get("evidence")
    if (
        not isinstance(evidence, list)
        or not evidence
        or not all(isinstance(phrase, str) for phrase in evidence)
        or len(evidence) != len(set(evidence))
        or not set(evidence).issubset(_EVIDENCE_PHRASES)
    ):
        issues.append(f"{location}.evidence is invalid")
    return issues


def validate_annotation(
    annotation: Mapping[str, Any],
    *,
    view: str,
    source_stage_id: str,
    target_stage_id: str,
    protocol_root: Path | None = None,
) -> list[str]:
    """Validate provider JSON before it can be rendered or persisted as success."""

    issues: list[str] = []
    try:
        normalized, spec = _view_spec(view, protocol_root)
        expected_source = stage_id_from_label(source_stage_id)
        expected_target = stage_id_from_label(target_stage_id)
    except (TypeError, ValueError) as exc:
        return [str(exc)]
    if not isinstance(annotation, Mapping):
        return ["annotation must be an object"]
    legacy_contract = _legacy_annotation_contract(protocol_root)
    root_keys = {"stage_reasoning", "growth_plan", "target_phenotype"}
    if legacy_contract:
        root_keys.add("image_quality_review")
    if set(annotation) != root_keys:
        issues.append(f"annotation root keys must be {sorted(root_keys)}")
    reasoning = annotation.get("stage_reasoning")
    if not isinstance(reasoning, Mapping) or set(reasoning) != {"current_stage", "future_stage"}:
        issues.append("stage_reasoning must contain only current_stage and future_stage")
    else:
        try:
            current = stage_id_from_label(reasoning["current_stage"])
            future = stage_id_from_label(reasoning["future_stage"])
            if current != expected_source:
                issues.append("stage_reasoning.current_stage does not match authoritative source stage")
            if future != expected_target:
                issues.append("stage_reasoning.future_stage does not match authoritative target stage")
            if _STAGE_ORDER.index(current) > _STAGE_ORDER.index(future):
                issues.append("stage transition moves backward")
        except (TypeError, ValueError) as exc:
            issues.append(str(exc))
    expected_keys = list(spec["ordered_keys"])
    for section_name, entry_name in (("growth_plan", "plan"), ("target_phenotype", "phenotype")):
        section = annotation.get(section_name)
        if not isinstance(section, Mapping):
            issues.append(f"{section_name} must be an object")
            continue
        if list(section.keys()) != expected_keys:
            issues.append(f"{section_name} keys must exactly follow registry order: {expected_keys}")
        for key in expected_keys:
            entry = section.get(key)
            if not isinstance(entry, Mapping):
                issues.append(f"{section_name}.{key} must be an object")
                continue
            required = {"text", "guardrails"} if legacy_contract and entry_name == "plan" else {"text"}
            if set(entry) != required:
                issues.append(f"{section_name}.{key} keys must be {sorted(required)}")
            issues.extend(_clean_text_issues(entry.get("text"), f"{section_name}.{key}"))
            if legacy_contract and entry_name == "plan":
                guardrails = entry.get("guardrails")
                if not isinstance(guardrails, list) or not all(isinstance(item, str) and item.strip() for item in guardrails):
                    issues.append(f"{section_name}.{key}.guardrails must be a string list")
                else:
                    for index, guardrail in enumerate(guardrails):
                        issues.extend(_clean_text_issues(guardrail, f"{section_name}.{key}.guardrails[{index}]"))
    if legacy_contract:
        quality = annotation.get("image_quality_review")
        if not isinstance(quality, Mapping) or set(quality) != {"source", "target"}:
            issues.append("image_quality_review must contain source and target")
        else:
            issues.extend(_validate_quality_item(quality["source"], "image_quality_review.source"))
            issues.extend(_validate_quality_item(quality["target"], "image_quality_review.target"))
    return issues


def _renderable(metric: Mapping[str, Any], key: str) -> tuple[bool, float | int | None, float | int | None]:
    if not isinstance(metric, Mapping):
        return False, None, None
    qc = metric.get("qc")
    if not isinstance(qc, Mapping) or qc.get("valid") is not True:
        return False, None, None
    values: list[Any] = []
    for side in ("source", "target"):
        observation = metric.get(side)
        if not isinstance(observation, Mapping) or observation.get("status") != "available":
            return False, None, None
        value = observation.get("value")
        if not _is_number(value) or not _metric_range(key, value):
            return False, None, None
        values.append(value)
    return True, values[0], values[1]


def _numeric_evidence(key: str, metric: Mapping[str, Any]) -> str | None:
    ok, source, target = _renderable(metric, key)
    if not ok:
        return None
    if key == "panicle":
        source_int = int(source)  # type: ignore[arg-type]
        target_int = int(target)  # type: ignore[arg-type]
        source_label = "visible" if source_int else "not visible"
        target_label = "visible" if target_int else "not visible"
        return f"Source = {source_label}; Target = {target_label}; Change = {target_int - source_int:+d}."
    scale = 100.0 if is_color_proxy_key(key, str(metric.get("metric_key", ""))) or key in {"plant_density", "canopy"} else 1.0
    source_value = float(source) * scale  # type: ignore[arg-type]
    target_value = float(target) * scale  # type: ignore[arg-type]
    change = target_value - source_value
    return f"Source = {source_value:.2f}%; Target = {target_value:.2f}%; Change = {change:+.2f} pp."


def _with_sentence(text: str) -> str:
    value = text.strip()
    return value if value.endswith((".", "!", "?")) else value + "."


def render_cot(
    annotation: Mapping[str, Any],
    *,
    view: str,
    metric_evidence: Mapping[str, Any],
    protocol_root: Path | None = None,
) -> tuple[str, list[str]]:
    """Render validated biological text and locally owned metric evidence."""

    _, spec = _view_spec(view, protocol_root)
    omissions: list[str] = []
    reasoning = annotation["stage_reasoning"]
    lines = [
        "<GROWTH_COT>",
        "",
        "Stage reasoning",
        f"Current stage: {reasoning['current_stage']}.",
        f"Future stage: {reasoning['future_stage']}.",
        "",
        "Growth plan",
    ]
    for key in spec["ordered_keys"]:
        entry = annotation["growth_plan"][key]
        line = f"{spec['section_labels'][key]}: {_with_sentence(entry['text'])}"
        guardrails = entry.get("guardrails", [])
        if guardrails:
            rendered_guardrails = "; ".join(item.rstrip(".") for item in guardrails)
            line += f" Guardrails: {rendered_guardrails}."
        evidence = _numeric_evidence(key, metric_evidence.get(key, {}))
        if evidence is None:
            omissions.append(key)
        else:
            from .cot_metric_names import metric_prefix
            line += f" {metric_prefix(spec, key)}{evidence}"
        lines.append(line)
    lines.extend(["", "Target phenotype"])
    for key in spec["ordered_keys"]:
        entry = annotation["target_phenotype"][key]
        lines.append(f"{spec['section_labels'][key]}: {_with_sentence(entry['text'])}")
    lines.extend(["", "</GROWTH_COT>"])
    return "\n".join(lines) + "\n", omissions


def protocol_sha256(protocol_root: Path) -> str:
    """Hash protocol file names and bytes in sorted order."""

    digest = hashlib.sha256()
    for path in sorted(item for item in protocol_root.rglob("*") if item.is_file()):
        relative = path.relative_to(protocol_root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
