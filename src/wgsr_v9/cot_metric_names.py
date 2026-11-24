"""Versioned display names for locally rendered CoT numeric evidence."""

from __future__ import annotations

import re
from typing import Any, Mapping

FORMAT = "metric_named_v1"
METRIC_NAMES = {
    "front_height_from_image_bottom_pct": "Height",
    "visible_foreground_area_pct": "Foreground area",
    "solidity": "Solidity",
    "canopy_density": "Canopy density",
    "panicle_visibility": "Panicle visibility",
    "yellow_brown_fraction": "Yellow-brown fraction",
}
ASSIGNMENTS = re.compile(r"\b(?:Source|Target|Change)\s*=")
MARKER = re.compile(r"\bMetric\s*=")


def metric_prefix(spec: Mapping[str, Any], key: str) -> str:
    """Keep legacy rendering byte-identical unless the protocol opts in."""
    version = spec.get("numeric_evidence_format")
    if version is None:
        return ""
    if version != FORMAT:
        raise ValueError(f"unsupported numeric evidence format: {version}")
    return f"Metric = {METRIC_NAMES[spec['metric_keys'][key]]}; "


def validate_metric_name(line: str, key: str, spec: Mapping[str, Any]) -> None:
    """Validate the binding, count and position; omission has no dangling name."""
    expected = metric_prefix(spec, key)
    markers = list(MARKER.finditer(line))
    assignments = list(ASSIGNMENTS.finditer(line))
    if not assignments:
        if markers:
            raise ValueError(f"{key}: metric name without numeric evidence")
        return
    if not expected and not markers:
        return
    expected = expected or f"Metric = {METRIC_NAMES[spec['metric_keys'][key]]}; "
    if len(markers) != 1:
        raise ValueError(f"{key}: missing or duplicate metric name")
    source = re.search(r"\bSource\s*=", line)
    if source is None or line[markers[0].start():source.start()] != expected:
        raise ValueError(f"{key}: metric name or position does not match its field")
    if assignments[0].start() != source.start():
        raise ValueError(f"{key}: metric name must precede the numeric triple")


def name_cot_evidence(body: str, view: str, bindings: Mapping[str, Any]) -> str:
    """Add names only in Growth plan; preserve all original text and numbers."""
    view = "top" if view in {"top", "top_view"} else "front"
    spec = bindings[view]
    if spec.get("numeric_evidence_format") != FORMAT:
        raise ValueError("named rendering requires an explicit metric_named_v1 protocol")
    labels = {label: key for key, label in spec["section_labels"].items()}
    section = None
    seen: set[str] = set()
    output = []
    for line in body.splitlines(keepends=True):
        if line.strip() in {"Stage reasoning", "Growth plan", "Target phenotype"}:
            section = line.strip()
        elif section == "Growth plan" and line.strip():
            label = line.partition(":")[0].strip()
            if label not in labels or label in seen:
                raise ValueError(f"unknown or duplicate Growth plan field: {label}")
            seen.add(label)
            key = labels[label]
            if MARKER.search(line):
                raise ValueError("input already contains a metric name; expected original provider text")
            if ASSIGNMENTS.search(line):
                sources = list(re.finditer(r"\bSource\s*=", line))
                if len(sources) != 1:
                    raise ValueError(f"{key}: expected exactly one source assignment")
                offset = sources[0].start()
                line = line[:offset] + metric_prefix(spec, key) + line[offset:]
            validate_metric_name(line, key, spec)
        output.append(line)
    if seen != set(labels):
        raise ValueError("missing Growth plan fields")
    return "".join(output)
