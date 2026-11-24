"""Image-result height and density using the existing CoT geometry contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .v9_image_metrics import _measure_geometry, foreground_mask, normalize_view


ALGORITHM_VERSION = "wgsr_v9_image_height_density_augmentation_v1"
PROTOCOL_VERSION = "wgsr_v9_image_height_density_v1"
METRICS = (
    "target_front_height_pct",
    "prediction_front_height_pct",
    "front_height_abs_error_pp",
    "target_projection_density_fraction",
    "prediction_projection_density_fraction",
    "projection_density_abs_error_pp",
)


def measure_geometry_mask(mask: np.ndarray, view: str) -> dict[str, Any]:
    """Reuse CoT's pixel-center convex hull and bottom-anchored front height."""
    view = normalize_view(view)
    geometry = _measure_geometry(mask, view)
    values = geometry["metrics"]
    density_key = "plant_density" if view == "front" else "canopy"
    top, bottom = geometry["top_row"], geometry["bottom_row"]
    return {
        "normalized_view": view,
        "front_height_pct": values.get("culm"),
        "projection_density_fraction": values[density_key],
        "height_metric_status": (
            geometry["metric_status"]["culm"] if view == "front" else "not_applicable"
        ),
        "height_reason_codes": (
            geometry["metric_reason_codes"]["culm"] if view == "front" else ["top_view"]
        ),
        "density_metric_status": geometry["metric_status"][density_key],
        "density_reason_codes": geometry["metric_reason_codes"][density_key],
        "foreground_area_pct": values["leaves_area"],
        "visible_extent_y_pct": (
            None if top is None else 100.0 * (bottom - top + 1) / mask.shape[0]
        ),
        **{key: geometry[key] for key in (
            "visible_foreground_pixels", "convex_hull_area_pixels", "top_row", "bottom_row",
            "bottom_gap_pct", "touches_image_boundary", "measurement_status",
        )},
    }


def measure_geometry_image(path: Path, view: str) -> dict[str, Any]:
    """Match the pixel evaluator's white alpha compositing and foreground mask."""
    with Image.open(path) as image:
        if "A" in image.getbands():
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba)
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return {
        "image_width": int(rgb.shape[1]),
        "image_height": int(rgb.shape[0]),
        **measure_geometry_mask(foreground_mask(rgb), view),
    }


def compare_geometry(target: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    if target["normalized_view"] != prediction["normalized_view"]:
        raise ValueError("target and prediction geometry views differ")
    output: dict[str, Any] = {}
    for name, field, multiplier in (
        ("height", "front_height_pct", 1.0),
        ("density", "projection_density_fraction", 100.0),
    ):
        left, right = target[field], prediction[field]
        error_field = "front_height_abs_error_pp" if name == "height" else "projection_density_abs_error_pp"
        output[f"target_{field}"] = left
        output[f"prediction_{field}"] = right
        output[error_field] = None if left is None or right is None else multiplier * abs(left - right)
        reasons = [
            f"{side}:{reason}"
            for side, geometry in (("target", target), ("prediction", prediction))
            for reason in geometry[f"{name}_reason_codes"]
        ]
        output[f"{name}_metric_status"] = (
            "available" if not reasons else
            "not_applicable" if name == "height" and target["normalized_view"] == "top" else "unavailable"
        )
        output[f"{name}_reason_codes"] = reasons
    return output
