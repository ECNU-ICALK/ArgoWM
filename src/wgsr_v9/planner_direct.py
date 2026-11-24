"""Versioned non-thinking planner output validation."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "wgsr_v9_planner_v3_direct_metric_named_weeks_v1"
STAGES = ("S1_tillering", "S2_jointing", "S3_booting", "S4_heading_flowering",
          "S5_early_grain_filling", "S6_milk_waxy", "S7_yellow_ripening", "S8_maturity_senescence")
CONTINUOUS = re.compile(r"Source = (\d+\.\d{2})%; Target = (\d+\.\d{2})%; Change = ([+-]\d+\.\d{2}) pp\.")
BINARY = re.compile(r"Source = (visible|not visible); Target = (visible|not visible); Change = ([+-][01])\.")


@lru_cache(maxsize=1)
def bindings() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / "protocols/qwen_image21_executor_v9_v3_metric_named_weeks/cot/VIEW_VARIABLE_BINDINGS.json"
    return json.loads(path.read_text())


def validate_direct_body(body: str, view: str | None) -> None:
    """Enforce the complete named-metric schema without rewriting teacher numbers."""
    if view not in {"front", "side0", "side90", "top"}:
        raise ValueError("unsupported direct-output view")
    spec = bindings()["top" if view == "top" else "front"]
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    labels = [spec["section_labels"][key] for key in spec["ordered_keys"]]
    n = len(labels)
    if len(lines) != 5 + 2 * n or lines[0] != "Stage reasoning" or lines[3] != "Growth plan" or lines[4+n] != "Target phenotype":
        raise ValueError("direct output has missing, extra, or reordered sections/fields")
    stages = []
    for line, prefix in zip(lines[1:3], ("Current stage: ", "Future stage: "), strict=True):
        stage = line[len(prefix):-1] if line.startswith(prefix) and line.endswith(".") else ""
        if stage not in STAGES:
            raise ValueError("invalid canonical stage")
        stages.append(STAGES.index(stage))
    if stages[1] < stages[0]:
        raise ValueError("future stage reverses development")
    for key, label, line, phenotype in zip(spec["ordered_keys"], labels, lines[4:4+n], lines[5+n:], strict=True):
        if not line.startswith(label + ": ") or not phenotype.startswith(label + ": "):
            raise ValueError("wrong field order or view-specific field")
        marker = " Metric = " + spec["metric_display_names"][key] + "; "
        if line.count("Metric =") != 1 or marker not in line:
            raise ValueError("missing or mismatched metric name")
        description, numeric = line.split(marker)
        if not description[len(label)+2:].strip():
            raise ValueError("empty growth description")
        if key == "panicle":
            match = BINARY.fullmatch(numeric)
            if not match or int(match[3]) != int(match[2] == "visible") - int(match[1] == "visible"):
                raise ValueError("invalid panicle visibility evidence")
        else:
            match = CONTINUOUS.fullmatch(numeric)
            if not match:
                raise ValueError("invalid continuous evidence units/precision")
            source, target, change = map(float, match.groups())
            if not (0 <= source <= 100 and 0 <= target <= 100) or abs((target - source) - change) > 0.0100001:
                raise ValueError("invalid continuous evidence range/difference")
        if not phenotype[len(label)+2:].strip() or re.search(r"\b(?:Metric|Source|Target|Change)\s*=", phenotype):
            raise ValueError("invalid qualitative target phenotype")
    if re.search(r"\b\d+\s+days?\b", body):
        raise ValueError("exact-day expression in weeks-only output")
