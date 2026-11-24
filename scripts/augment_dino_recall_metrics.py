#!/usr/bin/env python3
"""Add DINOv2 similarity and temporal Recall metrics to image evaluations.

The script uses the canonical source/target paths in the V9 input manifest to
build an image gallery.  Gallery candidates are restricted to the same plant
and normalized view as a query.  Short plant/view sequences can be excluded
with ``--min-gallery-images-per-plant-view``; the default is five images.

Evaluation directories are updated atomically when ``--merge-evaluation`` is
provided.  The input image manifest and prediction files are never modified.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import multiprocessing as mp
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wgsr_v9.dino_checkpoint import hf_dinov2_to_timm

ALGORITHM_VERSION = "wgsr_v9_dino_recall_augmentation_v1"
PROTOCOL_VERSION = "wgsr_v9_dino_temporal_recall_v1"
DEFAULT_DINO_MODEL = "vit_base_patch14_dinov2.lvd142m"
DEFAULT_DINO_CHECKPOINT = ROOT / "models/dinov2-base/dinov2_vitb14_pretrain.pth"
METRIC_FIELDS = (
    "prediction_dino_similarity",
    "source_baseline_dino_similarity",
    "dino_similarity_gain_vs_source",
    "dino_event_rank",
    "dino_event_reciprocal_rank",
    "dino_event_recall_at_1",
    "dino_event_recall_at_5",
    "dino_event_recall_at_10",
    "dino_stage_rank",
    "dino_stage_reciprocal_rank",
    "dino_stage_recall_at_1",
    "dino_stage_recall_at_5",
    "dino_stage_recall_at_10",
    "dino_candidate_count",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    return value


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(json_safe(value), ensure_ascii=False, indent=2) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(json_safe(row), ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_view(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"side0", "side", "front", "front_view"}:
        return "front"
    if text in {"top", "top_view"}:
        return "top"
    return text


def load_evaluations(paths: Sequence[Path], limit_rows: int | None) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Path]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    model_dirs: dict[str, Path] = {}
    for directory in paths:
        path = directory.resolve() / "per_pair_results.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = read_jsonl(path)
        if limit_rows is not None:
            rows = sorted(rows, key=lambda row: int(row.get("row_index", 10**9)))[:limit_rows]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            model = str(row.get("model_name") or "")
            if not model:
                raise ValueError(f"row without model_name in {path}")
            grouped[model].append(row)
        for model, model_rows in grouped.items():
            if model in by_model:
                raise ValueError(f"model {model!r} occurs in more than one evaluation directory")
            by_model[model] = sorted(model_rows, key=lambda row: int(row["row_index"]))
            model_dirs[model] = directory.resolve()
    if not by_model:
        raise ValueError("no evaluation rows were found")
    return by_model, model_dirs


def build_gallery(rows: Sequence[dict[str, Any]], min_images: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Build one metadata record per canonical source/target image path."""
    image_meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        for role in ("source", "target"):
            path_key = f"{role}_path"
            path = str(row.get(path_key) or row.get(f"{role}_image_path") or "")
            if not path:
                raise ValueError(f"row {row.get('row_index')} has no {role} image path")
            date = str(row.get(f"{role}_capture_date") or "")
            stage = str(row.get(f"{role}_stage") or "")
            image_id = str(row.get(f"{role}_image_id") or path)
            metadata = {
                "path": path,
                "relative_path": path,
                "plant_id": str(row.get("plant_id") or ""),
                "view": normalize_view(row.get("view")),
                "capture_date": date,
                "stage_id": stage,
                "event_id": image_id,
            }
            previous = image_meta.get(path)
            if previous is not None:
                for key in ("plant_id", "view", "capture_date", "stage_id", "event_id"):
                    if previous[key] != metadata[key]:
                        raise ValueError(f"conflicting {key} metadata for image {path}")
            else:
                image_meta[path] = metadata

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for metadata in image_meta.values():
        counts[(metadata["plant_id"], metadata["view"])] += 1
    excluded_sequences = sorted(key for key, count in counts.items() if count < min_images)
    excluded_set = set(excluded_sequences)
    gallery = [
        metadata
        for metadata in sorted(image_meta.values(), key=lambda item: item["path"])
        if (metadata["plant_id"], metadata["view"]) not in excluded_set
    ]
    protocol = {
        "min_gallery_images_per_plant_view": min_images,
        "gallery_sequence_key": "(plant_id, normalized_view)",
        "excluded_gallery_sequence_count": len(excluded_sequences),
        "excluded_gallery_image_count": sum(counts[key] for key in excluded_sequences),
        "remaining_gallery_image_count": len(gallery),
        "remaining_gallery_sequence_count": len(counts) - len(excluded_sequences),
        "excluded_sequences": [
            {"plant_id": key[0], "view": key[1], "image_count": counts[key]}
            for key in excluded_sequences
        ],
    }
    return gallery, image_meta, protocol


def load_image(path: str, normalize: bool) -> Image.Image:
    with Image.open(path) as image:
        output = image.convert("RGB")
    if normalize:
        array = np.asarray(output, dtype=np.uint8).copy()
        chroma = array.max(axis=2).astype(np.int16) - array.min(axis=2).astype(np.int16)
        near_white = (array.min(axis=2) >= 220) & (chroma <= 20)
        array[near_white] = 255
        output = Image.fromarray(array, mode="RGB")
    return output


def load_dino_model(model_name: str, checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor], dict[str, Any]]:
    import timm
    from timm.data import create_transform, resolve_model_data_config

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing DINO checkpoint: {checkpoint_path}")
    model = timm.create_model(model_name, pretrained=False, num_classes=0)
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(str(checkpoint_path), device="cpu")
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    checkpoint_format = "native_dinov2_or_timm"
    if "embeddings.position_embeddings" in state_dict:
        state_dict = hf_dinov2_to_timm(state_dict)
        checkpoint_format = "huggingface_dinov2_converted_to_timm_v1"
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or set(incompatible.unexpected_keys) - {"mask_token"}:
        raise RuntimeError(f"unexpected DINO checkpoint mismatch: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}")
    model = model.to(device).eval()
    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)
    metadata = {
        "model": model_name,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_format": checkpoint_format,
        "input_size": list(data_config["input_size"]),
        "interpolation": data_config["interpolation"],
        "crop_mode": data_config["crop_mode"],
        "crop_pct": data_config["crop_pct"],
        "mean": list(data_config["mean"]),
        "std": list(data_config["std"]),
        "embedding": "L2-normalized x_norm_clstoken",
    }
    return model, transform, metadata


def dino_embeddings(images: Sequence[Image.Image], model: torch.nn.Module, transform: Callable[[Image.Image], torch.Tensor], device: torch.device) -> torch.Tensor:
    inputs = torch.stack([transform(image) for image in images]).to(device)
    with torch.inference_mode():
        features = model.forward_features(inputs)
        if isinstance(features, dict):
            embeddings = features.get("x_norm_clstoken")
            if embeddings is None:
                raise KeyError("DINO forward_features did not return x_norm_clstoken")
        else:
            embeddings = model.forward_head(features, pre_logits=True)
    return F.normalize(embeddings.float(), dim=-1)


def encode_paths(paths: Sequence[str], model_name: str, checkpoint: Path, device_index: int, batch_size: int, normalize: bool, label: str, metadata: dict[str, Any] | None = None) -> np.ndarray:
    device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device_index)
    model, transform, model_metadata = load_dino_model(model_name, checkpoint, device)
    if metadata is not None:
        metadata.update(model_metadata)
        metadata.update({"device_type": device.type, "dtype": str(next(model.parameters()).dtype)})
    output: list[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        images = [load_image(path, normalize) for path in paths[start : start + batch_size]]
        output.append(dino_embeddings(images, model, transform, device).cpu().numpy().astype(np.float32, copy=False))
        print(f"[{label}] encoded {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
    return np.concatenate(output, axis=0) if output else np.empty((0, 0), dtype=np.float32)


def _encode_worker(rank: int, device_index: int, device_count: int, paths: list[str], model_name: str, checkpoint: str, batch_size: int, normalize: bool, output_dir: str) -> None:
    indices = list(range(rank, len(paths), device_count))
    local = [paths[index] for index in indices]
    metadata: dict[str, Any] = {}
    embedding = encode_paths(local, model_name, Path(checkpoint), device_index, batch_size, normalize, f"gpu{device_index}", metadata)
    np.savez_compressed(Path(output_dir) / f"shard-{rank:02d}.npz", indices=np.asarray(indices, dtype=np.int64), embedding=embedding)
    atomic_json(Path(output_dir) / f"shard-{rank:02d}.json", metadata)


def encode_distributed(paths: Sequence[str], model_name: str, checkpoint: Path, devices: Sequence[int], batch_size: int, normalize: bool, work_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    if not paths:
        return np.empty((0, 0), dtype=np.float32), {"embedding_dimension": 0, "devices": list(devices)}
    if len(devices) == 1:
        started = time.time()
        metadata: dict[str, Any] = {}
        result = encode_paths(paths, model_name, checkpoint, devices[0], batch_size, normalize, "DINO", metadata)
        return result, {**metadata, "embedding_dimension": int(result.shape[1]), "devices": list(devices), "elapsed_seconds": time.time() - started}
    if not torch.cuda.is_available():
        raise RuntimeError("multiple DINO devices require CUDA")
    work_dir.mkdir(parents=True, exist_ok=True)
    for child in work_dir.glob("shard-*.npz"):
        child.unlink()
    context = mp.get_context("spawn")
    processes = [context.Process(target=_encode_worker, args=(rank, device, len(devices), list(paths), model_name, str(checkpoint), batch_size, normalize, str(work_dir))) for rank, device in enumerate(devices)]
    started = time.time()
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failures = [process.exitcode for process in processes if process.exitcode != 0]
    if failures:
        raise RuntimeError(f"DINO worker failure exit codes: {failures}")
    shards = [np.load(work_dir / f"shard-{rank:02d}.npz") for rank in range(len(devices))]
    indices = np.concatenate([np.asarray(shard["indices"], dtype=np.int64) for shard in shards])
    values = np.concatenate([np.asarray(shard["embedding"], dtype=np.float32) for shard in shards], axis=0)
    expected = np.arange(len(paths), dtype=np.int64)
    if not np.array_equal(np.sort(indices), expected):
        raise RuntimeError("DINO shards do not cover every path exactly once")
    ordered = np.empty_like(values)
    ordered[indices] = values
    for shard in shards:
        shard.close()
    model_metadata = [json.loads((work_dir / f"shard-{rank:02d}.json").read_text(encoding="utf-8")) for rank in range(len(devices))]
    if any(item != model_metadata[0] for item in model_metadata[1:]):
        raise RuntimeError("DINO workers used different model/preprocessing settings")
    metadata = {**model_metadata[0], "embedding_dimension": int(ordered.shape[1]), "devices": list(devices), "elapsed_seconds": time.time() - started}
    return ordered, metadata


def parse_devices(value: str) -> list[int]:
    if value.strip().lower() == "auto":
        if not torch.cuda.is_available():
            return [0]
        return list(range(torch.cuda.device_count()))
    devices = [int(item.strip().removeprefix("cuda:")) for item in value.split(",") if item.strip()]
    if not devices:
        raise ValueError("--devices must contain at least one CUDA index")
    return devices


def valid_prediction_paths(rows_by_model: dict[str, list[dict[str, Any]]]) -> tuple[list[str], dict[str, str]]:
    paths: set[str] = set()
    status: dict[str, str] = {}
    for rows in rows_by_model.values():
        for row in rows:
            path = str(row.get("prediction_path") or "")
            key = f"{row.get('model_name')}::{row.get('row_index')}"
            if not path:
                status[key] = "unavailable"
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
            except (OSError, ValueError, SyntaxError):
                status[key] = "unavailable"
                continue
            status[key] = "available"
            paths.add(path)
    return sorted(paths), status


def hit(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def metric_stats(rows: Sequence[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [float(row[field]) for row in rows if row.get(field) is not None and row.get("dino_metric_status") == "available"]
    if not values:
        return {"mean": None, "median": None, "n": 0}
    return {"mean": statistics.fmean(values), "median": statistics.median(values), "n": len(values)}


def augment_model_rows(rows: list[dict[str, Any]], gallery: list[dict[str, Any]], image_meta: dict[str, dict[str, Any]], embeddings: dict[str, np.ndarray], prediction_status: dict[str, str]) -> list[dict[str, Any]]:
    gallery_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in gallery:
        gallery_by_key[(item["plant_id"], item["view"])].append(item)
    gallery_index = {item["path"]: index for index, item in enumerate(gallery)}
    output: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        model = str(row["model_name"])
        row_status = prediction_status.get(f"{model}::{row.get('row_index')}", "unavailable")
        target_path = str(row.get("target_path") or row.get("target_image_path") or "")
        source_path = str(row.get("source_path") or row.get("source_image_path") or "")
        key = (str(row.get("plant_id") or ""), normalize_view(row.get("view")))
        candidates = gallery_by_key.get(key, [])
        rank_fields: dict[str, Any] = {field: None for field in METRIC_FIELDS}
        rank_fields["dino_candidate_count"] = len(candidates)
        rank_fields["dino_metric_status"] = "unavailable"
        if row_status == "available" and target_path in gallery_index and source_path in gallery_index and row.get("prediction_path") in embeddings:
            prediction_vector = embeddings[str(row["prediction_path"])]
            target_vector = embeddings[target_path]
            source_vector = embeddings[source_path]
            prediction_similarity = float(np.dot(target_vector, prediction_vector))
            source_similarity = float(np.dot(target_vector, source_vector))
            score_by_path = {item["path"]: float(np.dot(prediction_vector, embeddings[item["path"]])) for item in candidates}
            ranked = sorted(candidates, key=lambda item: (-score_by_path[item["path"]], item["path"]))
            target_rank = next((position for position, item in enumerate(ranked, 1) if item["path"] == target_path), None)
            stage = str(row.get("target_stage") or image_meta[target_path]["stage_id"])
            stage_rank = next((position for position, item in enumerate(ranked, 1) if item["stage_id"] == stage), None)
            rank_fields.update(
                {
                    "prediction_dino_similarity": prediction_similarity,
                    "source_baseline_dino_similarity": source_similarity,
                    "dino_similarity_gain_vs_source": prediction_similarity - source_similarity,
                    "dino_event_rank": target_rank,
                    "dino_event_reciprocal_rank": None if target_rank is None else 1.0 / target_rank,
                    "dino_event_recall_at_1": hit(target_rank, 1),
                    "dino_event_recall_at_5": hit(target_rank, 5),
                    "dino_event_recall_at_10": hit(target_rank, 10),
                    "dino_stage_rank": stage_rank,
                    "dino_stage_reciprocal_rank": None if stage_rank is None else 1.0 / stage_rank,
                    "dino_stage_recall_at_1": hit(stage_rank, 1),
                    "dino_stage_recall_at_5": hit(stage_rank, 5),
                    "dino_stage_recall_at_10": hit(stage_rank, 10),
                    "dino_metric_status": "available",
                }
            )
        row.update(rank_fields)
        row["dino_algorithm_version"] = ALGORITHM_VERSION
        row["dino_protocol_version"] = PROTOCOL_VERSION
        row["dino_error"] = None if row["dino_metric_status"] == "available" else "prediction or canonical gallery image unavailable"
        row["dino_retry_count"] = 0
        output.append(row)
    return output


def grouped_rows(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(field) or "") for field in fields)].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        record = {field: value for field, value in zip(fields, key)}
        record["dataset_rows"] = len(group)
        record["dino_available"] = sum(row.get("dino_metric_status") == "available" for row in group)
        for metric in METRIC_FIELDS:
            values = [float(row[metric]) for row in group if row.get(metric) is not None and row.get("dino_metric_status") == "available"]
            record[f"{metric}_mean"] = statistics.fmean(values) if values else None
            record[f"{metric}_median"] = statistics.median(values) if values else None
            record[f"{metric}_n"] = len(values)
        output.append(record)
    return output


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    if not rows:
        atomic_text(path, "")
        return
    columns = list(fields)
    for key in rows[0]:
        if key not in columns:
            columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def merge_evaluation_directory(directory: Path, model: str, rows: list[dict[str, Any]], protocol: dict[str, Any], dino_metadata: dict[str, Any], output_dir: Path) -> None:
    per_pair_path = directory / "per_pair_results.jsonl"
    original = read_jsonl(per_pair_path)
    by_key = {(str(row["model_name"]), int(row["row_index"])): row for row in rows}
    merged: list[dict[str, Any]] = []
    for row in original:
        replacement = by_key.get((str(row.get("model_name") or model), int(row["row_index"])))
        merged.append(replacement if replacement is not None else row)
    write_jsonl(per_pair_path, merged)
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model_summary = summary.setdefault("models", {}).setdefault(model, {})
    model_summary["dino_metric_status"] = {
        "available": sum(row.get("dino_metric_status") == "available" for row in rows),
        "unavailable": sum(row.get("dino_metric_status") != "available" for row in rows),
    }
    metrics = model_summary.setdefault("metrics", {})
    for metric in METRIC_FIELDS:
        metrics[metric] = metric_stats(rows, metric)
    summary["dino_recall_protocol"] = protocol
    summary["dino_metadata"] = dino_metadata
    summary.setdefault("outputs", {})["dino_protocol"] = str((directory / "dino_protocol.json").resolve())
    summary.setdefault("outputs", {})["dino_metrics"] = str((directory / "dino_metrics.csv").resolve())
    atomic_json(summary_path, summary)
    atomic_json(directory / "dino_protocol.json", {**dino_metadata, **protocol, "algorithm_version": ALGORITHM_VERSION, "protocol_version": PROTOCOL_VERSION, "created_at_utc": utc_now()})
    grouped = grouped_rows(merged, ("model_name", "view", "split", "horizon_bucket", "source_stage", "target_stage"))
    write_csv(directory / "dino_metrics.csv", grouped, ("model_name", "view", "split", "horizon_bucket", "source_stage", "target_stage", "dataset_rows", "dino_available"))
    model_dir = output_dir / "models" / model
    model_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(model_dir / "per_pair_results.jsonl", rows)
    atomic_json(model_dir / "summary.json", {"model_name": model, "metrics": {metric: metric_stats(rows, metric) for metric in METRIC_FIELDS}})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, action="append", required=True, help="Evaluation directory; may be repeated.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dino-model", default=DEFAULT_DINO_MODEL)
    parser.add_argument("--dino-checkpoint", type=Path, default=DEFAULT_DINO_CHECKPOINT)
    parser.add_argument("--devices", default="auto", help="CUDA indices, for example 0,1,2,3; auto uses all visible GPUs.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-gallery-images-per-plant-view", type=int, default=5)
    parser.add_argument("--no-normalize-near-white", action="store_true")
    parser.add_argument("--limit-rows", type=int, default=None, help="Deterministic smoke-test limit per evaluation model.")
    parser.add_argument("--merge-evaluation", action="store_true", help="Atomically add the metrics to each input evaluation directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.min_gallery_images_per_plant_view <= 0:
        raise ValueError("--batch-size and --min-gallery-images-per-plant-view must be positive")
    if args.limit_rows is not None and args.limit_rows <= 0:
        raise ValueError("--limit-rows must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_model, model_dirs = load_evaluations([path.resolve() for path in args.evaluation_dir], args.limit_rows)
    all_rows = [row for rows in rows_by_model.values() for row in rows]
    gallery, image_meta, gallery_protocol = build_gallery(all_rows, args.min_gallery_images_per_plant_view)
    excluded_keys = {(item["plant_id"], item["view"]) for item in gallery_protocol["excluded_sequences"]}
    query_keys = {
        (int(row.get("row_index", -1)), str(row.get("plant_id") or ""), normalize_view(row.get("view")))
        for row in all_rows
    }
    query_excluded = sum((plant_id, view) in excluded_keys for _, plant_id, view in query_keys)
    gallery_protocol["excluded_query_count"] = query_excluded
    gallery_protocol["remaining_query_count"] = len(query_keys) - query_excluded
    gallery_protocol["canonical_query_count"] = len(query_keys)
    gallery_protocol["evaluation_row_count"] = len(all_rows)
    gallery_protocol["gallery_image_count_before_filter"] = len(image_meta)
    gallery_protocol["source"] = "canonical source/target paths in dataset manifest"
    atomic_text(output_dir / "gallery_records.jsonl", "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in gallery))

    prediction_paths, prediction_status = valid_prediction_paths(rows_by_model)
    unique_paths = [item["path"] for item in gallery] + prediction_paths
    unique_paths = list(dict.fromkeys(unique_paths))
    devices = parse_devices(args.devices)
    if len(devices) > 1 and not torch.cuda.is_available():
        raise RuntimeError("--devices specifies multiple GPUs but CUDA is unavailable")
    embeddings, encoding_metadata = encode_distributed(unique_paths, args.dino_model, args.dino_checkpoint.resolve(), devices, args.batch_size, not args.no_normalize_near_white, output_dir / ".encoding_work")
    path_embeddings = {path: embeddings[index] for index, path in enumerate(unique_paths)}
    protocol = {**gallery_protocol, "algorithm_version": ALGORITHM_VERSION, "protocol_version": PROTOCOL_VERSION, "dataset_manifest": str(args.dataset_manifest.resolve()), "dataset_manifest_sha256": sha256(args.dataset_manifest.resolve()), "dino_model": args.dino_model, "dino_checkpoint_sha256": sha256(args.dino_checkpoint.resolve()), "normalize_near_white": not args.no_normalize_near_white}
    dino_metadata = {**encoding_metadata, "model": args.dino_model, "checkpoint": str(args.dino_checkpoint.resolve()), "embedding_metric": "cosine similarity of L2-normalized DINOv2 CLS embeddings"}
    atomic_json(output_dir / "dino_protocol.json", {**protocol, **dino_metadata, "created_at_utc": utc_now()})
    for model, model_rows in rows_by_model.items():
        augmented = augment_model_rows(model_rows, gallery, image_meta, path_embeddings, prediction_status)
        if args.merge_evaluation:
            merge_evaluation_directory(model_dirs[model], model, augmented, protocol, dino_metadata, output_dir)
        else:
            model_dir = output_dir / "models" / model
            model_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(model_dir / "per_pair_results.jsonl", augmented)
            atomic_json(model_dir / "summary.json", {"model_name": model, "metrics": {metric: metric_stats(augmented, metric) for metric in METRIC_FIELDS}})
    atomic_json(output_dir / "summary.json", {"algorithm_version": ALGORITHM_VERSION, "protocol_version": PROTOCOL_VERSION, "created_at_utc": utc_now(), "models": sorted(rows_by_model), "model_count": len(rows_by_model), "gallery": protocol, "encoding": dino_metadata, "prediction_count": len(prediction_paths), "query_count": len(all_rows)})
    print(json.dumps(json_safe({"status": "completed", "output_dir": str(output_dir), "models": sorted(rows_by_model), "gallery": gallery_protocol, "encoding": dino_metadata}), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
