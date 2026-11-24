"""Contracts for the paired Qwen Image 2.1 executor SFT packages."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

ALGORITHM_VERSION = "wgsr_v9_qwen_image21_executor_sft_v1"
PROTOCOL_VERSION = "wgsr_v9_qwen_image21_executor_prompt_v1"
WEEKS_PROTOCOL_VERSION = "wgsr_v9_qwen_image21_executor_prompt_v2_weeks"
NAMED_WEEKS_PROTOCOL_VERSION = "wgsr_v9_qwen_image21_executor_prompt_v3_metric_named_weeks"
WEEKS_PROTOCOLS = {WEEKS_PROTOCOL_VERSION, NAMED_WEEKS_PROTOCOL_VERSION}
SUPPORTED_VIEWS = {"front": "front view", "side0": "front view", "top": "top view"}
EXPECTED_SIZE = {"front": (672, 896), "side0": (672, 896), "top": (896, 672)}


def prediction_interval_weeks(request: dict[str, Any]) -> int:
    """Validate canonical nearest-integer weeks, falling back for legacy requests."""
    days = request.get("actual_interval_days")
    if isinstance(days, bool) or not isinstance(days, int) or days <= 0:
        raise ValueError("actual_interval_days must be a positive integer")
    expected = max(1, (days + 3) // 7)
    weeks = request.get("prediction_interval_weeks", expected)
    if isinstance(weeks, bool) or not isinstance(weeks, int) or weeks != expected:
        raise ValueError("prediction_interval_weeks does not match canonical rounded weeks")
    return weeks


def growth_cot_body(rendered_text: str, request_id: str) -> str:
    """Return the provider-rendered CoT body without changing its contents."""
    text = rendered_text.strip()
    opening = "<GROWTH_COT>"
    closing = "</GROWTH_COT>"
    if not text.startswith(opening) or not text.endswith(closing):
        raise ValueError(f"{request_id}: rendered artifact must have one GROWTH_COT wrapper")
    body = text[len(opening) : -len(closing)].strip()
    if not body or opening in body or closing in body:
        raise ValueError(f"{request_id}: rendered artifact has an empty or nested GROWTH_COT body")
    return body


def executor_prompt(
    request: dict[str, Any],
    *,
    prompt_template: str,
    cot_suffix: str,
    variant: str,
    teacher_cot: str | None = None,
    protocol_version: str = PROTOCOL_VERSION,
) -> str:
    """Build one deployment-shaped prompt from source-known request fields."""
    request_id = str(request.get("request_id") or "<missing request_id>")
    view = str(request.get("view") or "")
    if view not in SUPPORTED_VIEWS:
        raise ValueError(f"{request_id}: unsupported view {view!r}")
    if variant not in {"simple", "teacher_cot"}:
        raise ValueError(f"unsupported executor variant: {variant!r}")
    interval_days = int(request["actual_interval_days"])
    if interval_days <= 0:
        raise ValueError(f"{request_id}: prediction interval must be positive")
    if protocol_version in WEEKS_PROTOCOLS:
        weeks = prediction_interval_weeks(request)
        if "{interval_days}" in prompt_template:
            raise ValueError("weeks-only protocol cannot expose exact days in the prompt")
    elif protocol_version == PROTOCOL_VERSION:
        weeks = max(1, math.ceil(interval_days / 7))
    else:
        raise ValueError(f"unsupported executor prompt protocol: {protocol_version}")
    prompt = prompt_template.format(
        accession_id=str(request["accession_id"]),
        view_label=SUPPORTED_VIEWS[view],
        interval_weeks=weeks,
        interval_days=interval_days,
    ).strip()
    if variant == "simple":
        if teacher_cot is not None:
            raise ValueError(f"{request_id}: simple prompt cannot receive teacher CoT")
        return prompt
    if teacher_cot is None or not teacher_cot.strip():
        raise ValueError(f"{request_id}: teacher_cot prompt requires a nonempty CoT body")
    return prompt + cot_suffix.format(growth_cot=teacher_cot.strip())


def make_executor_row(
    request: dict[str, Any],
    *,
    prompt: str,
    variant: str,
    original_row_index: int,
    protocol_version: str = PROTOCOL_VERSION,
) -> dict[str, Any]:
    """Map a V9 request to the DiffSynth image-edit row contract."""
    request_id = str(request["request_id"])
    source = request.get("source")
    target = request.get("target")
    if not isinstance(source, dict) or not isinstance(target, dict):
        raise ValueError(f"{request_id}: source/target objects are required")
    input_path = Path(str(source.get("path") or "")).resolve()
    output_path = Path(str(target.get("path") or "")).resolve()
    if not input_path.is_file() or not output_path.is_file():
        raise FileNotFoundError(f"{request_id}: missing source or target image")
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "protocol_version": protocol_version,
        "request_id": request_id,
        "pair_group_id": str(request["pair_group_id"]),
        "split": str(request["split"]),
        "plant_id": str(request["plant_id"]),
        "view": str(request["view"]),
        "actual_interval_days": int(request["actual_interval_days"]),
        **({"prediction_interval_weeks": prediction_interval_weeks(request),
            "time_condition_unit": "weeks"} if protocol_version in WEEKS_PROTOCOLS else {}),
        "executor_variant": variant,
        "image": str(output_path),
        "edit_image": str(input_path),
        "prompt": prompt,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "status": "ready",
        "error": None,
        "retry_count": 0,
        "original_row_index": original_row_index,
    }
