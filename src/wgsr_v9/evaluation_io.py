"""Shared, read-only loaders for WGSR V9 baseline evaluation helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ALGORITHM_VERSION = "wgsr_v9_inference_result_alignment_v1"
PROTOCOL_VERSION = "wgsr_v9_image_result_evaluation_v1"
SUCCESS_STATUSES = {"completed", "success", "succeeded", "ready"}


def _normalized_view(value: Any) -> str:
    """Normalize the dataset's ``side0`` spelling to the executor's front view."""
    text = str(value or "").strip().lower()
    if text in {"side0", "side", "front", "front_view"}:
        return "front"
    if text in {"top", "top_view"}:
        return "top"
    return text


@dataclass(frozen=True)
class Prediction:
    row: dict[str, Any]
    output_path: Path
    result_file: Path


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-manifest",
        required=True,
        type=Path,
        help="JSONL dataset manifest with canonical source_image_path and target_image_path.",
    )
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Model result JSONL or directory containing result JSONL ledgers; repeat per model.",
    )
    parser.add_argument(
        "--path-root",
        action="append",
        default=[],
        type=Path,
        help="Additional root for resolving relative image paths; may be repeated.",
    )


def parse_result_specs(specs: Iterable[str]) -> dict[str, Path]:
    results: dict[str, Path] = {}
    for spec in specs:
        name, separator, raw_path = spec.partition("=")
        name = name.strip()
        raw_path = raw_path.strip()
        if not separator or not name or not raw_path:
            raise ValueError(f"result must have the form NAME=PATH: {spec!r}")
        if name in results:
            raise ValueError(f"duplicate result model name: {name}")
        results[name] = Path(raw_path).expanduser()
    return results


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL ledger or a JSON array/object containing record rows."""

    if path.suffix.lower() != ".json":
        return read_jsonl(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict):
        rows = value.get("rows", value.get("data", value.get("results")))
    else:
        rows = None
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected a JSON array of objects in {path}")
    return [dict(row) for row in rows]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_file(path: Path, bases: Iterable[Path]) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    candidates = [base / expanded for base in bases]
    candidates.append(expanded)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def load_dataset_manifest(path: Path, extra_roots: Iterable[Path]) -> tuple[Path, list[dict[str, Any]], str]:
    manifest_path = path.expanduser().resolve()
    rows = read_records(manifest_path)
    roots = [root.expanduser().resolve() for root in extra_roots]
    project_root = Path(__file__).resolve().parents[2]
    for row_index, row in enumerate(rows):
        if not row.get("pair_id"):
            if row.get("row_index") is None:
                raise ValueError(f"dataset row {row_index} is missing pair_id and row_index")
        for field in ("source_image_path", "target_image_path"):
            raw_path = row.get(field)
            if not raw_path:
                raise ValueError(f"dataset row {row_index} is missing {field}")
            resolved = resolve_file(
                Path(str(raw_path)),
                [*roots, manifest_path.parent, Path.cwd(), project_root],
            )
            row[field] = str(resolved)
    return manifest_path, rows, sha256_file(manifest_path)


def discover_result_files(path: Path) -> list[Path]:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return [resolved]
    if not resolved.is_dir():
        raise FileNotFoundError(f"result path does not exist: {path}")
    files = sorted(
        candidate
        for candidate in resolved.rglob("results*")
        if candidate.is_file() and candidate.suffix.lower() in {".jsonl", ".json"}
    )
    if not files:
        raise FileNotFoundError(f"no results*.jsonl/results*.json ledgers found under {resolved}")
    return files


def _timestamp_value(row: dict[str, Any]) -> float | None:
    for key in ("updated_at", "completed_at", "finished_at", "timestamp", "created_at"):
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            normalized = value.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
            return parsed.timestamp()
        except ValueError:
            continue
    return None


def _record_key(row: dict[str, Any]) -> tuple[str, str] | None:
    if row.get("pair_id") not in (None, ""):
        return ("pair_id", str(row["pair_id"]))
    # Executor training/inference rows use request_id; in the V9 temporal
    # dataset this identifier is the same stable pair key as pair_id.
    if row.get("request_id") not in (None, ""):
        return ("pair_id", str(row["request_id"]))
    if row.get("row_index") is not None:
        return ("row_index", str(row["row_index"]))
    return None


def load_predictions(
    result_path: Path,
    *,
    extra_roots: Iterable[Path],
    dataset_rows: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], Prediction], dict[str, Any]]:
    result_files = discover_result_files(result_path)
    project_root = Path(__file__).resolve().parents[2]
    roots = [root.expanduser().resolve() for root in extra_roots]
    dataset_keys = {
        ("pair_id", str(row["pair_id"])): row
        for row in dataset_rows
        if row.get("pair_id") not in (None, "")
    }
    dataset_index_keys = {
        ("row_index", str(row["row_index"])): row
        for row in dataset_rows
        if row.get("row_index") is not None
    }
    candidates: dict[tuple[str, str], list[tuple[tuple[float, int, int], Prediction]]] = {}
    file_records: list[dict[str, Any]] = []
    malformed = 0
    unkeyed = 0
    failure_count = 0
    missing_output_count = 0
    mismatch_rows = 0
    mismatch_examples: list[dict[str, Any]] = []

    for file_order, result_file in enumerate(result_files):
        rows = read_records(result_file)
        file_records.append(
            {
                "path": str(result_file),
                "sha256": sha256_file(result_file),
                "row_count": len(rows),
            }
        )
        file_mtime = result_file.stat().st_mtime
        for line_order, row in enumerate(rows):
            key = _record_key(row)
            if key is None:
                unkeyed += 1
                continue
            status = str(row.get("status", "")).strip().lower()
            if status not in SUCCESS_STATUSES:
                failure_count += 1
                continue
            raw_output = row.get("output_path") or row.get("generated_image_path") or row.get("image_path")
            if not raw_output:
                missing_output_count += 1
                continue
            output_path = resolve_file(
                Path(str(raw_output)),
                [*roots, Path.cwd(), project_root, result_file.parent],
            )
            if not output_path.is_file():
                missing_output_count += 1
                continue

            dataset_row = dataset_keys.get(key) or dataset_index_keys.get(key)
            if dataset_row is not None:
                fields_to_check = ("pair_group_id", "split", "view")
                mismatches = {
                    field: {"dataset": dataset_row.get(field), "result": row.get(field)}
                    for field in fields_to_check
                    if row.get(field) not in (None, "")
                    and dataset_row.get(field) not in (None, "")
                    and (
                        _normalized_view(row[field]) != _normalized_view(dataset_row[field])
                        if field == "view"
                        else str(row[field]) != str(dataset_row[field])
                    )
                }
                if mismatches:
                    mismatch_rows += 1
                    if len(mismatch_examples) < 20:
                        mismatch_examples.append({"key": list(key), "fields": mismatches, "result_file": str(result_file)})

            timestamp = _timestamp_value(row)
            ranking = (timestamp if timestamp is not None else file_mtime, file_order, line_order)
            prediction = Prediction(row=dict(row), output_path=output_path, result_file=result_file)
            candidates.setdefault(key, []).append((ranking, prediction))

    best: dict[tuple[str, str], Prediction] = {}
    for key, options in candidates.items():
        best[key] = max(options, key=lambda option: option[0])[1]

    duplicate_key_count = sum(len(options) > 1 for options in candidates.values())
    missing_dataset_keys = []
    for row in dataset_rows:
        key = _record_key(row)
        if key is not None and key not in best:
            missing_dataset_keys.append(key)

    report = {
        "result_path": str(result_path.expanduser().resolve()),
        "result_files": file_records,
        "ledger_row_count": sum(item["row_count"] for item in file_records),
        "successful_existing_records": sum(len(options) for options in candidates.values()),
        "matched_key_count": len(best),
        "duplicate_key_count": duplicate_key_count,
        "failed_or_non_success_record_count": failure_count,
        "success_record_missing_output_count": missing_output_count,
        "unkeyed_record_count": unkeyed,
        "dataset_rows_without_prediction_count": len(missing_dataset_keys),
        "dataset_rows_without_prediction_examples": [list(key) for key in missing_dataset_keys[:20]],
        "metadata_mismatch_row_count": mismatch_rows,
        "metadata_mismatch_examples": mismatch_examples,
        "selection_policy": "latest successful record with an existing output file; timestamp fields first, then ledger mtime and line order",
    }
    return best, report


def pair_key_for_dataset_row(row: dict[str, Any]) -> tuple[str, str] | None:
    return _record_key(row)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def ensure_new_output_dir(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists; choose a new path: {output}")
    output.mkdir(parents=True)
    return output
