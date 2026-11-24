"""Maize input contract and provenance joins.

The source archive and the design package are read-only inputs.  This module
only parses their manifests and returns normalized records for downstream
materialization.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from zipfile import ZipFile


MAIZE_ALGORITHM_VERSION = "wgsr_maize_input_materialization_v1"
MAIZE_PROTOCOL_VERSION = "wgsr_maize_sv1_input_v1"
MAIZE_VIEW = "sv1"
EXCLUDED_VIEW = "sv2"
CANONICAL_SOURCE_SIZE = (768, 1270)
CANONICAL_TARGET_SIZE = (768, 1280)
DESIGN_IMAGE_MANIFEST = "COT_Maize_Design_v2/data/image_manifest.csv"
DESIGN_MANUAL_REVIEW = "COT_Maize_Design_v2/data/manual_organ_review.csv"
DESIGN_STAGE_TABLE = "COT_Maize_Design_v2/protocol/STAGES.json"
RGB_MEMBER_PREFIX = "rgb/"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_rows(data: bytes, source: str) -> list[dict[str, str]]:
    text = data.decode("utf-8-sig")
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise ValueError(f"CSV has no rows: {source}")
    if any(row.get(None) for row in rows):
        raise ValueError(f"CSV contains extra unnamed columns: {source}")
    return [{str(key): (value or "").strip() for key, value in row.items()} for row in rows]


def load_csv_file(path: Path) -> tuple[list[dict[str, str]], str]:
    data = path.read_bytes()
    return _csv_rows(data, str(path)), sha256_bytes(data)


def load_design_package(path: Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    with ZipFile(path) as archive:
        manifest_bytes = archive.read(DESIGN_IMAGE_MANIFEST)
        manual_bytes = archive.read(DESIGN_MANUAL_REVIEW)
        stage_bytes = archive.read(DESIGN_STAGE_TABLE)
    return _csv_rows(manifest_bytes, DESIGN_IMAGE_MANIFEST), {
        "image_manifest_sha256": sha256_bytes(manifest_bytes),
        "manual_review_sha256": sha256_bytes(manual_bytes),
        "stage_table_sha256": sha256_bytes(stage_bytes),
    }


def load_stage_table(path: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    stages = value.get("stages") if isinstance(value, dict) else None
    if not isinstance(stages, list) or not stages:
        raise ValueError(f"Invalid maize stage table: {path}")
    result: dict[str, dict[str, Any]] = {}
    for stage in stages:
        if not isinstance(stage, dict):
            raise ValueError(f"Invalid stage row in {path}")
        stage_id = str(stage.get("stage_id", "")).strip()
        if not stage_id or stage_id in result:
            raise ValueError(f"Duplicate or missing stage_id in {path}: {stage_id!r}")
        days = stage.get("days_after_sowing")
        if not isinstance(days, int) or days < 0:
            raise ValueError(f"Invalid days_after_sowing for {stage_id}")
        result[stage_id] = dict(stage)
    return result


def _required(row: dict[str, str], keys: Iterable[str], source: str) -> None:
    missing = [key for key in keys if key not in row]
    if missing:
        raise ValueError(f"Missing columns {missing} in {source}")


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not value:
        raise ValueError(f"Unsafe relative image path: {value!r}")
    return path.as_posix()


def load_catalog(
    *,
    design_zip: Path,
    manual_path: Path,
    stages_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join the design manifest, manual observations, and stage table.

    The returned rows include both included and excluded manifest entries.  A
    caller must filter ``selection_status == included`` before materializing
    images.
    """

    manifest_rows, design_hashes = load_design_package(design_zip)
    manual_rows, manual_sha256 = load_csv_file(manual_path)
    if manual_sha256 != design_hashes["manual_review_sha256"]:
        raise ValueError("External manual-review table differs from the design-package copy")
    stages_sha256 = sha256_file(stages_path)
    if stages_sha256 != design_hashes["stage_table_sha256"]:
        raise ValueError("External stage table differs from the design-package copy")
    stage_table = load_stage_table(stages_path)
    _required(
        manifest_rows[0],
        (
            "image_id",
            "plant_id",
            "stage_id",
            "days_after_sowing",
            "maize_stage",
            "view",
            "relative_image_path",
            "aligned_sha256",
            "alignment_flags",
            "alignment_needs_review",
            "selection_status",
            "selection_reason",
        ),
        DESIGN_IMAGE_MANIFEST,
    )
    _required(
        manual_rows[0],
        (
            "image_id",
            "plant_id",
            "stage_id",
            "days_after_sowing",
            "maize_stage",
            "tassel_presence",
            "ear_presence",
            "plant_condition",
        ),
        str(manual_path),
    )
    manifest_by_id = {row["image_id"]: row for row in manifest_rows}
    manual_by_id = {row["image_id"]: row for row in manual_rows}
    if len(manifest_by_id) != len(manifest_rows):
        raise ValueError("Duplicate image_id in maize image manifest")
    if len(manual_by_id) != len(manual_rows):
        raise ValueError("Duplicate image_id in maize manual review")

    included_ids = {
        row["image_id"] for row in manifest_rows if row["selection_status"] == "included"
    }
    if included_ids != set(manual_by_id):
        missing = sorted(included_ids.difference(manual_by_id))[:3]
        extra = sorted(set(manual_by_id).difference(included_ids))[:3]
        raise ValueError(f"Manual/image join mismatch; missing={missing}, extra={extra}")

    joined: list[dict[str, Any]] = []
    for manifest in manifest_rows:
        image_id = manifest["image_id"]
        stage_id = manifest["stage_id"]
        stage = stage_table.get(stage_id)
        if stage is None:
            raise ValueError(f"Unknown maize stage {stage_id!r} for {image_id}")
        try:
            days = int(manifest["days_after_sowing"])
        except ValueError as exc:
            raise ValueError(f"Invalid DAS for {image_id}") from exc
        if days != stage["days_after_sowing"]:
            raise ValueError(f"DAS mismatch for {image_id}: {days} != {stage['days_after_sowing']}")
        if manifest["maize_stage"] != str(stage.get("maize_stage", "")):
            raise ValueError(f"Maize stage mismatch for {image_id}: {manifest['maize_stage']}")
        relative = _safe_relative_path(manifest["relative_image_path"])
        view = manifest["view"]
        if view != MAIZE_VIEW:
            status = "excluded_view" if view == EXCLUDED_VIEW else "excluded_unknown_view"
        else:
            status = manifest["selection_status"]
        manual = manual_by_id.get(image_id, {})
        for field in ("tassel_presence", "ear_presence"):
            value = manual.get(field, "")
            if value not in {"", "visible", "not_visible"}:
                raise ValueError(f"Invalid {field}={value!r} for {image_id}")
        review = str(manifest.get("alignment_needs_review", "")).strip() == "1"
        flags = str(manifest.get("alignment_flags", "")).strip()
        quality_stratum = (
            "excluded" if status != "included" else ("alignment_review" if review or flags else "strict_alignment")
        )
        joined.append(
            {
                "species": "maize",
                "image_id": image_id,
                "plant_id": manifest["plant_id"],
                "stage_id": stage_id,
                "days_after_sowing": days,
                "maize_stage": manifest["maize_stage"],
                "view": view,
                "relative_image_path": relative,
                "source_archive_member": RGB_MEMBER_PREFIX + relative,
                "manifest_aligned_sha256": manifest["aligned_sha256"],
                "alignment_flags": flags,
                "alignment_needs_review": review,
                "selection_status": status,
                "selection_reason": manifest.get("selection_reason", ""),
                "tassel_presence": manual.get("tassel_presence", "") or None,
                "ear_presence": manual.get("ear_presence", "") or None,
                "plant_condition": manual.get("plant_condition", "") or None,
                "quality_stratum": quality_stratum,
            }
        )
    provenance = {
        "design_package": str(design_zip.resolve()),
        "design_package_sha256": sha256_file(design_zip),
        "design_inner_sha256": design_hashes,
        "manual_review_path": str(manual_path.resolve()),
        "manual_review_sha256": manual_sha256,
        "stage_table_path": str(stages_path.resolve()),
        "stage_table_sha256": stages_sha256,
        "stage_ids": sorted(stage_table, key=lambda item: int(item[1:])),
    }
    return joined, provenance
