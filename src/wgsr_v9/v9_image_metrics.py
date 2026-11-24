"""Compute the image-level numeric evidence used by the WGSR V9 contract.

The calculator consumes the exact-white PNG representation produced by the V9
background-removal stage.  It deliberately keeps the public planning names
(``culm``, ``leaves_area``, ``plant_density``, ``canopy``, and
``yellow_brown_fraction``) alongside the canonical metric names used by the
protocol registry.  No pair-level delta is computed here; deltas belong to the
request/renderer stage.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .io import read_jsonl, sha256_path, write_json, write_jsonl


ALGORITHM_VERSION = "wgsr_v9_image_metrics_v1_pure_white_hsv_baseline"
COLOR_ALGORITHM_VERSION = "wgsr_v9_color_hsv_baseline_v1"
MIN_COLOR_COVERAGE = 0.70
BACKGROUND_MIN_VALUE = 220
BACKGROUND_MAX_CHROMA = 20


def _numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("NumPy is required for V9 image metrics") from exc
    return np


def normalize_view(raw_view: str) -> str:
    value = str(raw_view or "").strip().lower()
    if value in {"side0", "side90", "front", "front_view", "side"}:
        return "front"
    if value in {"top", "top_view"}:
        return "top"
    raise ValueError(f"unsupported V9 view: {raw_view!r}")


def _view_label(raw_view: str) -> str:
    value = str(raw_view or "").strip().lower()
    if value in {"side0", "side90", "front", "front_view", "side"}:
        return "side0" if value != "side90" else "side90"
    if value in {"top", "top_view"}:
        return "top"
    raise ValueError(f"unsupported V9 view: {raw_view!r}")


def _normalize_rgb(rgb: Any) -> Any:
    np = _numpy()
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"expected HxWx3 RGB data, got {array.shape}")
    normalized = array[..., :3].copy()
    work = normalized.astype(np.int16, copy=False)
    low = work.min(axis=2)
    background = (low >= BACKGROUND_MIN_VALUE) & (
        work.max(axis=2) - low <= BACKGROUND_MAX_CHROMA
    )
    normalized[background] = 255
    return normalized


def foreground_mask(rgb: Any) -> Any:
    """Return the V9 visible mask after near-white normalization."""

    np = _numpy()
    normalized = _normalize_rgb(rgb)
    return np.any(normalized != 255, axis=2)


def _cross(origin: tuple[int, int], left: tuple[int, int], right: tuple[int, int]) -> int:
    return (left[0] - origin[0]) * (right[1] - origin[1]) - (
        left[1] - origin[1]
    ) * (right[0] - origin[0])


def _monotonic_hull(points: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted(set(points))
    if len(ordered) <= 1:
        return list(ordered)
    lower: list[tuple[int, int]] = []
    for point in ordered:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[int, int]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _convex_hull(points: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return hull vertices, using SciPy when available for large masks."""

    if len(points) <= 2:
        return list(dict.fromkeys(points))
    try:
        import numpy as np  # type: ignore
        from scipy.spatial import ConvexHull, QhullError  # type: ignore

        coordinates = np.asarray(points, dtype=np.float64)
        try:
            hull = ConvexHull(coordinates)
        except QhullError:
            return _monotonic_hull(points)
        return [tuple(int(value) for value in coordinates[index]) for index in hull.vertices]
    except ImportError:  # pragma: no cover - fallback for minimal environments
        return _monotonic_hull(points)


def _rasterized_convex_hull_area(vertices: Sequence[tuple[int, int]]) -> int:
    """Count integer pixel centers covered by a convex hull."""

    if len(vertices) < 3:
        return 0
    min_y = min(y for _, y in vertices)
    max_y = max(y for _, y in vertices)
    area = 0
    epsilon = 1e-9
    for y in range(min_y, max_y + 1):
        intersections: list[float] = []
        for index, point in enumerate(vertices):
            next_point = vertices[(index + 1) % len(vertices)]
            x1, y1 = point
            x2, y2 = next_point
            if y1 == y2:
                if y == y1:
                    intersections.extend((float(x1), float(x2)))
                continue
            if min(y1, y2) <= y <= max(y1, y2):
                intersections.append(x1 + (y - y1) * (x2 - x1) / float(y2 - y1))
        if intersections:
            left = math.ceil(min(intersections) - epsilon)
            right = math.floor(max(intersections) + epsilon)
            if right >= left:
                area += right - left + 1
    return area


def _hsv_from_rgb(rgb: Any) -> tuple[Any, Any, Any]:
    np = _numpy()
    array = np.asarray(rgb, dtype=np.float32)
    work = array[..., :3]
    if float(work.max(initial=0.0)) > 1.0:
        work = work / 255.0
    red, green, blue = work[..., 0], work[..., 1], work[..., 2]
    maximum = work.max(axis=2)
    minimum = work.min(axis=2)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    nonzero = delta > 1e-8
    red_max = (maximum == red) & nonzero
    green_max = (maximum == green) & nonzero
    blue_max = (maximum == blue) & nonzero
    hue[red_max] = (60.0 * (green[red_max] - blue[red_max]) / delta[red_max]) % 360.0
    hue[green_max] = 60.0 * ((blue[green_max] - red[green_max]) / delta[green_max] + 2.0)
    hue[blue_max] = 60.0 * ((red[blue_max] - green[blue_max]) / delta[blue_max] + 4.0)
    saturation = np.zeros_like(maximum)
    positive_maximum = maximum > 1e-8
    saturation[positive_maximum] = delta[positive_maximum] / maximum[positive_maximum]
    return hue, saturation, maximum


def classify_colors(rgb: Any, mask: Any, *, min_color_coverage: float = MIN_COLOR_COVERAGE) -> dict[str, Any]:
    """Classify visible pixels into disjoint green, yellow, and brown classes."""

    np = _numpy()
    array = np.asarray(rgb)
    foreground = np.asarray(mask, dtype=bool)
    if array.ndim != 3 or array.shape[:2] != foreground.shape or array.shape[2] < 3:
        raise ValueError(f"RGB/mask shape mismatch: {array.shape} and {foreground.shape}")
    foreground_pixels = int(foreground.sum())
    if foreground_pixels == 0:
        return {
            "green_fraction": None,
            "yellow_fraction": None,
            "brown_fraction": None,
            "unclassified_fraction": None,
            "color_coverage": None,
            "yellow_brown_fraction": None,
            "classified_color_pixels": 0,
            "status": "invalid",
            "reason_codes": ["empty_foreground"],
            "algorithm_version": COLOR_ALGORITHM_VERSION,
        }

    hue, saturation, value = _hsv_from_rgb(array)
    hue = hue[foreground]
    saturation = saturation[foreground]
    value = value[foreground]

    # Keep the audited exploratory baseline.  Brown is checked first because
    # low-value orange/brown pixels overlap the lower yellow hue interval.
    brown = ((hue < 35.0) | (hue >= 330.0)) & (saturation >= 0.12) & (value < 0.75)
    yellow = (
        (hue >= 25.0)
        & (hue < 75.0)
        & (saturation >= 0.15)
        & (value >= 0.25)
        & ~brown
    )
    green = (
        (hue >= 35.0)
        & (hue < 170.0)
        & (saturation >= 0.15)
        & (value >= 0.12)
        & ~brown
        & ~yellow
    )
    classified = green | yellow | brown
    color_coverage = float(classified.mean())
    yellow_brown_fraction = float((yellow | brown).mean())
    status = "available" if color_coverage >= float(min_color_coverage) else "pending"
    reason_codes = [] if status == "available" else ["low_color_coverage"]
    return {
        "green_fraction": float(green.mean()),
        "yellow_fraction": float(yellow.mean()),
        "brown_fraction": float(brown.mean()),
        "unclassified_fraction": float((~classified).mean()),
        "color_coverage": color_coverage,
        "yellow_brown_fraction": yellow_brown_fraction,
        "classified_color_pixels": int(classified.sum()),
        "status": status,
        "reason_codes": reason_codes,
        "algorithm_version": COLOR_ALGORITHM_VERSION,
    }


def _empty_geometry(height: int, width: int, view: str) -> dict[str, Any]:
    metrics: dict[str, float | None] = {"leaves_area": 0.0}
    if view == "front":
        metrics.update({"culm": None, "plant_density": None})
    else:
        metrics["canopy"] = None
    return {
        "metrics": metrics,
        "metric_status": {key: "invalid" for key in metrics},
        "metric_reason_codes": {key: ["empty_foreground"] for key in metrics},
        "visible_foreground_pixels": 0,
        "image_pixels": int(height * width),
        "top_row": None,
        "bottom_row": None,
        "bottom_gap_pct": None,
        "touches_top": False,
        "touches_bottom": False,
        "touches_left": False,
        "touches_right": False,
        "touches_image_boundary": False,
        "boundary_contact": False,
        "measurement_status": "empty_mask",
        "convex_hull_area_pixels": None,
    }


def _measure_geometry(mask: Any, view: str) -> dict[str, Any]:
    np = _numpy()
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError(f"expected HxW mask, got {array.shape}")
    height, width = (int(array.shape[0]), int(array.shape[1]))
    foreground_pixels = int(array.sum())
    if foreground_pixels == 0:
        return _empty_geometry(height, width, view)

    ys, xs = np.where(array)
    top_row = int(ys.min())
    bottom_row = int(ys.max())
    hull_vertices = _convex_hull([(int(x), int(y)) for y, x in zip(ys, xs)])
    hull_area = _rasterized_convex_hull_area(hull_vertices)
    density = None if hull_area <= 0 else foreground_pixels / float(hull_area)
    leaves_area = 100.0 * foreground_pixels / float(height * width)
    touches_top = top_row == 0
    touches_bottom = bottom_row == height - 1
    touches_left = bool(array[:, 0].any())
    touches_right = bool(array[:, -1].any())
    touches_boundary = touches_top or touches_bottom or touches_left or touches_right
    geometry: dict[str, Any] = {
        "metrics": {"leaves_area": leaves_area},
        "metric_status": {"leaves_area": "available"},
        "metric_reason_codes": {"leaves_area": []},
        "visible_foreground_pixels": foreground_pixels,
        "image_pixels": int(height * width),
        "top_row": top_row,
        "bottom_row": bottom_row,
        "bottom_gap_pct": 100.0 * (height - 1 - bottom_row) / float(height),
        "touches_top": bool(touches_top),
        "touches_bottom": bool(touches_bottom),
        "touches_left": touches_left,
        "touches_right": touches_right,
        "touches_image_boundary": bool(touches_boundary),
        "boundary_contact": bool(touches_boundary),
        "convex_hull_area_pixels": int(hull_area) if hull_area > 0 else None,
    }
    if view == "front":
        geometry["metrics"]["culm"] = 100.0 * (height - top_row) / float(height)
        geometry["metrics"]["plant_density"] = density
        geometry["metric_status"]["culm"] = "available"
        geometry["metric_reason_codes"]["culm"] = []
        geometry["metric_status"]["plant_density"] = "available" if density is not None else "invalid"
        geometry["metric_reason_codes"]["plant_density"] = [] if density is not None else ["degenerate_hull"]
        geometry["boundary_contact"] = bool(touches_top or touches_left or touches_right)
        if touches_top:
            geometry["measurement_status"] = "top_censored"
        elif not touches_bottom:
            geometry["measurement_status"] = "bottom_gap"
        else:
            geometry["measurement_status"] = "valid"
    else:
        geometry["metrics"]["canopy"] = density
        geometry["metric_status"]["canopy"] = "available" if density is not None else "invalid"
        geometry["metric_reason_codes"]["canopy"] = [] if density is not None else ["degenerate_hull"]
        geometry["measurement_status"] = "boundary_contact" if touches_boundary else "valid"
        geometry["bottom_gap_pct"] = None
    return geometry


def measure_rgb(rgb: Any, *, view: str) -> dict[str, Any]:
    """Measure one decoded RGB image according to the V9 view contract."""

    np = _numpy()
    array = np.asarray(rgb)
    canonical_view = normalize_view(view)
    mask = foreground_mask(array)
    geometry = _measure_geometry(mask, canonical_view)
    colors = classify_colors(array, mask)
    metrics = dict(geometry["metrics"])
    metric_status = dict(geometry["metric_status"])
    metric_reason_codes = dict(geometry["metric_reason_codes"])
    metrics["yellow_brown_fraction"] = colors["yellow_brown_fraction"]
    metric_status["yellow_brown_fraction"] = colors["status"]
    metric_reason_codes["yellow_brown_fraction"] = list(colors["reason_codes"])
    valid_geometry = all(status in {"available", "pending"} for status in geometry["metric_status"].values())
    row_status = "ready" if valid_geometry else "invalid"
    canonical_metrics = {
        "visible_foreground_area_pct": metrics.get("leaves_area"),
        "solidity": metrics.get("plant_density", metrics.get("canopy")),
        "front_height_from_image_bottom_pct": metrics.get("culm"),
        "yellow_brown_fraction": metrics.get("yellow_brown_fraction"),
    }
    return {
        **{
            key: value
            for key, value in geometry.items()
            if key not in {"metrics", "metric_status", "metric_reason_codes"}
        },
        "algorithm_version": ALGORITHM_VERSION,
        "view": _view_label(view),
        "status": row_status,
        "error": None,
        "retry_count": 0,
        "metrics": metrics,
        "metric_status": metric_status,
        "metric_reason_codes": metric_reason_codes,
        "canonical_metrics": canonical_metrics,
        "color": colors,
    }


def _pair_rows(pair_paths: Sequence[Path]) -> Iterable[dict[str, Any]]:
    for path in pair_paths:
        candidates = sorted(path.glob("**/pairs.jsonl")) if path.is_dir() else [path]
        for candidate in candidates:
            yield from read_jsonl(candidate)


def build_pair_image_map(pair_paths: Sequence[Path], pure_white_dir: Path) -> dict[str, dict[str, Any]]:
    """Build an immutable image-id to pure-white PNG map from V9 pair rows."""

    mapping: dict[str, dict[str, Any]] = {}
    references: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"roles": set(), "splits": set(), "groups": set()})
    for row in _pair_rows(pair_paths):
        raw_view = str(row.get("view", row.get("view_label", "")))
        view = _view_label(raw_view)
        for role in ("source", "target"):
            image_id = row.get(f"{role}_image_id")
            if not image_id:
                raise ValueError(f"pair row is missing {role}_image_id: {row.get('pair_id')}")
            image_id = str(image_id)
            filename = row.get(f"{role}_filename")
            if not filename:
                raw_path = row.get(f"{role}_input_path", row.get(f"{role}_image_path"))
                filename = Path(str(raw_path)).name
            filename = str(filename)
            png_name = Path(filename).with_suffix(".png").name
            candidate = {
                "image_id": image_id,
                "view": view,
                "filename": filename,
                "raw_input_path": row.get(f"{role}_input_path", row.get(f"{role}_image_path")),
                "pure_white_path": str((pure_white_dir / png_name).resolve()),
                "plant_id": row.get("plant_id"),
                "accession_id": row.get("canonical_accession_id", row.get("accession_id")),
                "capture_date": row.get(f"{role}_capture_date", row.get("source_capture_date" if role == "source" else "target_capture_date")),
                "stage": row.get(f"{role}_stage"),
                "stage_en": row.get(f"{role}_stage_en"),
                "DAS": row.get(f"{role}_DAS"),
            }
            previous = mapping.get(image_id)
            if previous is not None:
                if previous["pure_white_path"] != candidate["pure_white_path"] or previous["view"] != view:
                    raise ValueError(f"image id maps to conflicting records: {image_id}")
                for field in ("plant_id", "accession_id", "capture_date", "stage", "stage_en", "DAS"):
                    if previous.get(field) not in (None, "") and candidate.get(field) not in (None, "") and str(previous[field]) != str(candidate[field]):
                        raise ValueError(f"image id has conflicting {field}: {image_id}")
                    if previous.get(field) in (None, "") and candidate.get(field) not in (None, ""):
                        previous[field] = candidate[field]
            else:
                mapping[image_id] = candidate
            references[image_id]["roles"].add(role)
            if row.get("split"):
                references[image_id]["splits"].add(str(row["split"]))
            if row.get("pair_group_id"):
                references[image_id]["groups"].add(str(row["pair_group_id"]))
    for image_id, record in mapping.items():
        record["roles"] = sorted(references[image_id]["roles"])
        record["splits"] = sorted(references[image_id]["splits"])
        record["pair_group_ids"] = sorted(references[image_id]["groups"])
    return dict(sorted(mapping.items()))


def _error_row(record: Mapping[str, Any], *, output_path: Path, error: Exception) -> dict[str, Any]:
    return {
        **dict(record),
        "algorithm_version": ALGORITHM_VERSION,
        "input_path": record.get("pure_white_path"),
        "output_path": str(output_path),
        "image_sha256": None,
        "status": "invalid",
        "error": f"{type(error).__name__}: {error}",
        "retry_count": 0,
        "metrics": {},
        "metric_status": {},
        "metric_reason_codes": {},
    }


def compute_image_metrics(
    pair_paths: Sequence[Path],
    *,
    pure_white_dir: Path,
    output: Path,
    offset: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Measure all unique pair images and atomically write a JSONL sidecar."""

    pure_white_dir = pure_white_dir.resolve()
    output = output.resolve()
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    mapping = build_pair_image_map(pair_paths, pure_white_dir)
    selected = list(mapping.items())[offset:] if limit is None else list(mapping.items())[offset : offset + int(limit)]
    mapping_output = output.with_name(f"{output.stem}.image_path_map.jsonl")
    mapping_rows = [
        {
            "algorithm_version": f"{ALGORITHM_VERSION}_pair_image_map",
            **record,
            "output_path": str(mapping_output),
            "status": "ready",
            "error": None,
            "retry_count": 0,
        }
        for _, record in selected
    ]
    write_jsonl(mapping_output, mapping_rows)
    rows: list[dict[str, Any]] = []
    for image_id, record in selected:
        image_path = Path(str(record["pure_white_path"]))
        try:
            from PIL import Image  # type: ignore

            with Image.open(image_path) as image:
                array = _numpy().asarray(image.convert("RGB"))
            measured = measure_rgb(array, view=str(record["view"]))
            measured.update(
                {
                    **record,
                    "image_id": image_id,
                    "input_path": str(image_path),
                    "output_path": str(output),
                    "image_sha256": sha256_path(image_path),
                }
            )
            rows.append(measured)
        except (OSError, ValueError, RuntimeError) as exc:
            rows.append(_error_row({**record, "image_id": image_id}, output_path=output, error=exc))
    write_jsonl(output, rows)
    summary = {
        "algorithm_version": ALGORITHM_VERSION,
        "output": str(output),
        "image_path_map": str(mapping_output),
        "pure_white_dir": str(pure_white_dir),
        "pair_paths": [str(path) for path in pair_paths],
        "requested_image_count": len(mapping),
        "offset": int(offset),
        "limit": limit,
        "selected_image_count": len(selected),
        "measured_image_count": len(rows),
        "ready_count": sum(row.get("status") == "ready" for row in rows),
        "invalid_count": sum(row.get("status") == "invalid" for row in rows),
        "view_counts": {
            view: sum(row.get("view") == view for row in rows)
            for view in sorted({str(record["view"]) for _, record in selected})
        },
        "color_status_counts": {
            status: sum(row.get("metric_status", {}).get("yellow_brown_fraction") == status for row in rows)
            for status in ("available", "pending", "invalid")
        },
    }
    write_json(output.with_suffix(output.suffix + ".summary.json"), summary)
    return summary
