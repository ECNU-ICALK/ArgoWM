"""Reusable input-contract helpers for the species-isolated maize pipeline."""

from .contract import (
    CANONICAL_SOURCE_SIZE,
    CANONICAL_TARGET_SIZE,
    MAIZE_ALGORITHM_VERSION,
    MAIZE_PROTOCOL_VERSION,
    MAIZE_VIEW,
    load_catalog,
    load_stage_table,
)
from .geometry import normalize_geometry
from .naming import OUTPUT_NAMING_VERSION, maize_legacy_output_stem, maize_output_stem
from .cot import (
    MAIZE_COT_ALGORITHM_VERSION,
    MAIZE_COT_PROTOCOL_VERSION,
    build_request,
    render_cot,
)

__all__ = [
    "CANONICAL_SOURCE_SIZE",
    "CANONICAL_TARGET_SIZE",
    "MAIZE_ALGORITHM_VERSION",
    "MAIZE_PROTOCOL_VERSION",
    "MAIZE_VIEW",
    "load_catalog",
    "load_stage_table",
    "normalize_geometry",
    "OUTPUT_NAMING_VERSION",
    "maize_output_stem",
    "maize_legacy_output_stem",
    "MAIZE_COT_ALGORITHM_VERSION",
    "MAIZE_COT_PROTOCOL_VERSION",
    "build_request",
    "render_cot",
]
