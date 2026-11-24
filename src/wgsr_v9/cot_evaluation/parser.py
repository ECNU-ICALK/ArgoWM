"""Lossless field locations and independent endpoint parsing; no text repair."""
from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import re
from typing import Any
from ..cot_metric_names import validate_metric_name

SECTIONS = ("Stage reasoning", "Growth plan", "Target phenotype")
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
TRANSLATE = str.maketrans({"％": "%", "−": "-", "＋": "+"})


def scalar(text: str, role: str, *, binary: bool = False) -> tuple[float | int | None, str | None]:
    matches = list(re.finditer(rf"\b{role}\s*=", text))
    if len(matches) != 1:
        return None, "missing" if not matches else "duplicate"
    suffix = text[matches[0].end():].strip()
    if binary and role != "Change":
        match = re.match(r"(not visible|visible)(?=\s*(?:;|\.(?:\s|$)|$))", suffix)
        return (int(match[1] == "visible"), None) if match else (None, "invalid_binary_label")
    unit = "" if binary else ("pp" if role == "Change" else "%")
    match = re.match(rf"({NUMBER})\s*{unit}(?=\s*(?:;|\.(?:\s|$)|$))", suffix)
    if not match or len(match[1]) > 32:
        return None, "invalid_number_or_unit"
    try:
        value = Decimal(match[1])
    except InvalidOperation:
        return None, "invalid_number"
    lower, upper = (-1, 1) if binary else ((-100, 100) if role == "Change" else (0, 100))
    if not value.is_finite() or not lower <= value <= upper:
        return None, "out_of_range"
    if binary and value != int(value):
        return None, "noninteger_change"
    return (int(value) if binary else float(value)), None


def parse_cot(text: str, view: str, bindings: dict[str, Any], stages: list[str],
              *, finish_reason: str = "stop", output_wrapper: str = "think") -> dict[str, Any]:
    issues: list[str] = []
    normalized_events: list[str] = []
    if "\r" in text:
        normalized_events.append("line_endings")
    if text.translate(TRANSLATE) != text:
        normalized_events.append("unicode_numeric_characters")
    if finish_reason in {"length", "max_tokens", "max_new_tokens"}:
        issues.append("truncated")
    if output_wrapper == "think":
        wrapper = re.fullmatch(r"\s*<think>(.*?)</think>\s*", text, re.DOTALL)
        wrapper_valid = (text.count("<think>") == text.count("</think>") == 1
                         and "<GROWTH_COT>" not in text and "</GROWTH_COT>" not in text)
    elif output_wrapper == "GROWTH_COT":
        wrapper = re.fullmatch(r"\s*(?:<think>\s*</think>\s*)?<GROWTH_COT>(.*?)</GROWTH_COT>\s*",
                               text, re.DOTALL)
        wrapper_valid = (text.count("<GROWTH_COT>") == text.count("</GROWTH_COT>") == 1
                         and text.count("<think>") == text.count("</think>") <= 1)
        if wrapper and ("<think>" in wrapper[1] or "</think>" in wrapper[1]):
            wrapper_valid = False
    else:
        raise ValueError(f"unsupported output wrapper: {output_wrapper}")
    if not wrapper or not wrapper_valid:
        issues.append("invalid_wrapper")
    if wrapper:
        body, offset = wrapper[1], wrapper.start(1)
    else:
        body, offset = text, 0
    section: str | None = None
    headings: list[str] = []
    points: dict[str, dict[str, list[dict[str, Any]]]] = {s: defaultdict(list) for s in SECTIONS}
    labels = {value: key for key, value in bindings[view]["section_labels"].items()}
    expected = bindings[view]["ordered_keys"]
    cursor = offset
    for raw in body.splitlines(keepends=True):
        line = raw.strip()
        start, end = cursor, cursor + len(raw)
        cursor = end
        if not line or line in {"<think>", "</think>"}:
            continue
        if line in SECTIONS:
            section = line
            headings.append(line)
            continue
        if section is None:
            issues.append("content_outside_sections")
            continue
        label, separator, value = line.partition(":")
        label, value = label.strip(), value.strip()
        valid_labels = {"Current stage": "current", "Future stage": "future"} if section == SECTIONS[0] else labels
        if not separator or label not in valid_labels:
            issues.append(f"unknown_or_wrong_view_field:{section}:{label}")
            continue
        key = valid_labels[label]
        points[section][key].append({"start": start, "end": end, "raw": raw,
                                     "label": label, "text": value})
    if headings != list(SECTIONS):
        issues.append("missing_duplicate_or_unordered_sections")
    stage_values: dict[str, str | None] = {}
    stage_errors: dict[str, str | None] = {}
    for key in ("current", "future"):
        occurrences = points[SECTIONS[0]].get(key, [])
        value = occurrences[0]["text"].removesuffix(".").strip() if len(occurrences) == 1 else None
        error = ("missing" if not occurrences else "duplicate") if len(occurrences) != 1 else (None if value in stages else "unknown_stage")
        stage_values[key] = value if error is None else None
        stage_errors[key] = error
        if error:
            issues.append(f"stage:{key}:{error}")
    fields: dict[str, Any] = {}
    qualitative: dict[str, dict[str, str | None]] = {s: {} for s in SECTIONS[1:]}
    for section in SECTIONS[1:]:
        for key in expected:
            occurrences = points[section].get(key, [])
            valid = len(occurrences) == 1 and bool(occurrences[0]["text"])
            if not valid:
                problem = "missing" if not occurrences else ("duplicate" if len(occurrences) > 1 else "empty")
                issues.append(f"point:{section}:{key}:{problem}")
            value = occurrences[0]["text"] if valid else None
            if section == "Target phenotype":
                if value and re.search(r"\b(?:Metric|Source|Target|Change)\s*=|\d\s*(?:%|％|pp\b)", value):
                    issues.append(f"numeric_phenotype:{key}")
                qualitative[section][key] = value
                continue
            normalized = value.translate(TRANSLATE) if value else ""
            metric_error = None
            if valid:
                try:
                    validate_metric_name(value, key, bindings[view])
                except ValueError as exc:
                    metric_error = str(exc)
                    issues.append(f"metric_binding:{key}:{metric_error}")
            values: dict[str, Any] = {}
            errors: dict[str, str | None] = {}
            for role, output in (("Source", "source"), ("Target", "target"), ("Change", "declared_change")):
                parsed, error = scalar(normalized, role, binary=key == "panicle") if valid else (None, "invalid_field")
                if metric_error:
                    parsed, error = None, "invalid_metric_binding"
                values[output], errors[output] = parsed, error
            derived = None
            residual = None
            if values["source"] is not None and values["target"] is not None:
                derived = float(Decimal(str(values["target"])) - Decimal(str(values["source"])))
                if values["declared_change"] is not None:
                    residual = float(Decimal(str(values["declared_change"])) - Decimal(str(derived)))
            prefix = re.split(r"\b(?:Metric|Source|Target|Change)\s*=", value or "", maxsplit=1)[0].strip()
            qualitative[section][key] = prefix if valid and prefix else None
            if valid and not prefix:
                issues.append(f"point:{section}:{key}:empty_qualitative")
            fields[key] = {**values, "derived_change": derived, "arithmetic_residual": residual,
                           "parse_errors": errors, "raw_text": value,
                           "numeric_assignment_present": bool(re.search(r"\b(?:Source|Target|Change)\s*=", normalized))}
    return {"structure_valid": not issues, "issues": sorted(set(issues)), "issue_counts": dict(Counter(issues)),
            "normalizations": normalized_events, "stages": stage_values, "stage_errors": stage_errors,
            "fields": fields, "qualitative": qualitative, "occurrences": points,
            "heading_order": headings, "finish_reason": finish_reason}
