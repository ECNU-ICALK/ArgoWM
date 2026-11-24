"""Reusable helpers for the WGSR V9 paper data pipeline."""

from .cot_synthesis import (
    normalize_view,
    protocol_sha256,
    render_cot,
    stage_id_from_label,
    validate_annotation,
    validate_metric_evidence,
    validate_pair_metadata,
)
from .planner_adapter import growth_cot_body, to_growth_cot

__version__ = "0.1.0"

__all__ = [
    "normalize_view",
    "protocol_sha256",
    "render_cot",
    "stage_id_from_label",
    "validate_annotation",
    "validate_metric_evidence",
    "validate_pair_metadata",
    "growth_cot_body",
    "to_growth_cot",
]

