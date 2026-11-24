"""Versioned, source-only prompts and output checks for the maize experiment."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_ROOT = ROOT / "protocols/maize_four_way_v1"
PROTOCOL = "wgsr_maize_four_way_v1"
EXECUTOR_PROTOCOL = "wgsr_maize_executor_days_v1"
PLANNER_PROTOCOL = "wgsr_maize_planner_direct_days_v1"
PLANNER_FORMAT = "maize_sv1_v1"
INFERENCE_PROTOCOL = "wgsr_maize_planner_direct_days_source_only_inference_v1"
SIZE = (768, 1280)
LABELS = ("Culm", "Visible foreground area", "Projected fill", "Tassel", "Ear",
          "Visible yellow-brown composition")
STABLE = "Stable within the registered two-percent proxy tolerance; Source/Target/Change suppressed."


def simple_prompt(row: dict[str, Any]) -> str:
    """Allowlist only identity and horizon; target fields cannot affect the prompt."""
    plant = str(row["plant_id"])
    days = row["actual_interval_days"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", plant):
        raise ValueError("Invalid plant ID")
    if isinstance(days, bool) or not isinstance(days, int) or days <= 0:
        raise ValueError("Maize prediction requires positive integral actual_interval_days")
    return (PROTOCOL_ROOT / "EXECUTOR_PROMPT_TEMPLATE.txt").read_text().strip().format(
        plant_id=plant, actual_interval_days=days)


def validate_growth_cot(text: str) -> str:
    """Check a prediction without consulting future labels or teacher text."""
    text = text.strip()
    from .cot import REFUSAL_TEXT
    if REFUSAL_TEXT.search(text):
        raise ValueError("Provider refusal is not a maize phenotype description")
    if re.search(r"[\u3400-\u9fff]|</?think>", text):
        raise ValueError("Maize output must be English direct CoT")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 19 or lines[0] != "<GROWTH_COT>" or lines[-1] != "</GROWTH_COT>":
        raise ValueError("Expected one complete maize GROWTH_COT block")
    if lines[1] != "Stage reasoning" or lines[4] != "Growth plan" or lines[11] != "Target phenotype":
        raise ValueError("Invalid maize section order")
    for index, name in ((2, "Current"), (3, "Future")):
        if not re.fullmatch(rf"{name} stage: S[1-9]\.", lines[index]):
            raise ValueError("Expected maize stage S1 through S9")
    for offset in (5, 12):
        for label, line in zip(LABELS, lines[offset:offset + 6], strict=True):
            if not line.startswith(label + ": ") or not line[len(label) + 2:].strip():
                raise ValueError(f"Missing or reordered maize field: {label}")
    for index in (5, 6, 7, 10):
        line = lines[index]
        if index == 5 and line.endswith(STABLE):
            continue
        match = re.search(r"Source=(\d+\.\d)%; Target=(\d+\.\d)%; Change=(-?\d+\.\d) pp\.$", line)
        if match is None:
            raise ValueError("Missing one-decimal maize numeric triplet")
        source, target, delta = map(float, match.groups())
        if not 0 <= source <= 100 or not 0 <= target <= 100 or abs(target - source - delta) > 0.151:
            raise ValueError("Maize numeric triplet is out of range or inconsistent")
    return text


def executor_prompt(row: dict[str, Any], cot: str | None = None) -> str:
    prompt = simple_prompt(row)
    if cot is not None:
        suffix = (PROTOCOL_ROOT / "COT_SUFFIX.txt").read_text().strip()
        prompt += "\n\n" + suffix.format(growth_cot=validate_growth_cot(cot))
    return prompt


def planner_messages(row: dict[str, Any], teacher_root: Path) -> list[dict[str, str]]:
    stages = json.loads((teacher_root / "STAGES.json").read_text())["stages"]
    guide = "\n".join(f"{s['stage_id']}: {s.get('maize_stage', '')}; {s['visible_description_en']}" for s in stages)
    system = (PROTOCOL_ROOT / "PLANNER_SYSTEM.txt").read_text().strip().format(stage_guide=guide)
    # Reuse the allowlist validation, never serialize a pair/request into the user message.
    simple_prompt(row)
    user = (f"<image>\nMaize plant ID: {row['plant_id']}. Shared view: sv1 (side view). "
            f"Prediction interval: {row['actual_interval_days']} days. Forecast the growth script.")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def validate_executor_row(row: dict[str, Any], variant: str) -> None:
    if (row.get("species"), row.get("native_view"), row.get("view"), row.get("time_condition_unit")) != (
            "maize", "sv1", "side0", "days"):
        raise ValueError("Maize executor metadata mismatch")
    if row.get("executor_variant") != variant or tuple(row.get("source_size", [])) != SIZE:
        raise ValueError("Maize executor variant or native geometry mismatch")
    cot = None
    if variant != "simple":
        prompt = row["prompt"]
        if prompt.count("<GROWTH_COT>") != 1 or prompt.count("</GROWTH_COT>") != 1:
            raise ValueError("Expected exactly one maize growth script")
        cot = "<GROWTH_COT>" + prompt.split("<GROWTH_COT>", 1)[1]
        if variant == "predicted_cot" and (row.get("cot_source") != "planner_prediction"
                                           or not row.get("adapter_sha256")):
            raise ValueError("Predicted CoT requires trained planner provenance; teacher fallback is forbidden")
    if row["prompt"] != executor_prompt(row, cot):
        raise ValueError("Maize prompt differs from the trained template")
