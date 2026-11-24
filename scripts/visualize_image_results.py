#!/usr/bin/env python3
"""Create aligned source/target/multi-model comparison sheets from existing results."""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]

from wgsr_v9.evaluation_io import (  # noqa: E402
    ALGORITHM_VERSION,
    PROTOCOL_VERSION,
    add_common_arguments,
    ensure_new_output_dir,
    load_dataset_manifest,
    load_predictions,
    pair_key_for_dataset_row,
    parse_result_specs,
    write_json_atomic,
    write_jsonl_atomic,
)


TILE_WIDTH = 320
TILE_HEIGHT = 380
TILE_GAP = 12
CANVAS_COLOR = (245, 246, 248)


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        image = opened.convert("RGBA")
        white = Image.new("RGBA", image.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, image).convert("RGB")


def _fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.copy()
    image.thumbnail(size, Image.Resampling.LANCZOS)
    return image


def _safe_name(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "sample"))
    return cleaned[:120] or "sample"


def _draw_wrapped(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], font: ImageFont.ImageFont, width: int) -> int:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and draw.textbbox((0, 0), candidate, font=font)[2] > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    y = xy[1]
    for line in lines:
        draw.text((xy[0], y), line, fill=(35, 39, 46), font=font)
        y += font.size + 3
    return y


def _tile(label: str, image_path: Path | None, detail: str | None = None) -> Image.Image:
    tile = Image.new("RGB", (TILE_WIDTH, TILE_HEIGHT), "white")
    draw = ImageDraw.Draw(tile)
    title_font = _font(18)
    detail_font = _font(13)
    draw.rounded_rectangle((0, 0, TILE_WIDTH - 1, TILE_HEIGHT - 1), radius=8, outline=(205, 210, 217), width=2)
    draw.text((12, 10), label, fill=(20, 24, 30), font=title_font)
    if detail:
        _draw_wrapped(draw, detail, (12, 38), detail_font, TILE_WIDTH - 24)
    box_top = 82
    box_size = (TILE_WIDTH - 20, TILE_HEIGHT - box_top - 12)
    if image_path is None or not image_path.is_file():
        draw.rectangle((10, box_top, TILE_WIDTH - 10, TILE_HEIGHT - 12), fill=(238, 240, 243))
        message = "MISSING RESULT" if image_path is None else f"FILE NOT FOUND\n{image_path}"
        draw.multiline_text((20, box_top + 20), message, fill=(145, 50, 50), font=detail_font, spacing=5)
        return tile
    image = _fit_image(_load_rgb(image_path), box_size)
    x = (TILE_WIDTH - image.width) // 2
    y = box_top + (box_size[1] - image.height) // 2
    tile.paste(image, (x, y))
    return tile


def _compose_row(
    dataset_row: dict[str, Any],
    model_names: list[str],
    predictions: dict[str, Any],
) -> Image.Image:
    pair_id = str(dataset_row.get("pair_id", dataset_row.get("row_index", "unknown")))
    metadata = (
        f"{pair_id} | {dataset_row.get('split', '')} | {dataset_row.get('view', '')} | "
        f"plant {dataset_row.get('plant_id', '')} | "
        f"{dataset_row.get('source_capture_date', '')} → {dataset_row.get('target_capture_date', '')} | "
        f"{dataset_row.get('source_stage', '')} → {dataset_row.get('target_stage', '')} | "
        f"{dataset_row.get('actual_interval_days', '')} days"
    )
    cells = [
        _tile("Real source", Path(str(dataset_row["source_image_path"]))),
        _tile("Real target", Path(str(dataset_row["target_image_path"]))),
    ]
    for model_name in model_names:
        prediction = predictions.get(model_name)
        cells.append(
            _tile(
                model_name,
                prediction.output_path if prediction else None,
                None if prediction else "No successful existing output matched this pair",
            )
        )
    header_height = 74
    width = len(cells) * TILE_WIDTH + (len(cells) + 1) * TILE_GAP
    height = header_height + TILE_HEIGHT + TILE_GAP
    sheet = Image.new("RGB", (width, height), CANVAS_COLOR)
    draw = ImageDraw.Draw(sheet)
    header_font = _font(15)
    _draw_wrapped(draw, metadata, (TILE_GAP, 8), header_font, width - 2 * TILE_GAP)
    for index, cell in enumerate(cells):
        x = TILE_GAP + index * (TILE_WIDTH + TILE_GAP)
        sheet.paste(cell, (x, header_height))
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--view", choices=("all", "front", "top"), default="all")
    parser.add_argument("--sample-pair-id", action="append", default=[])
    parser.add_argument("--sample-pair-group", action="append", default=[])
    parser.add_argument(
        "--selection",
        choices=("intersection", "available"),
        default="intersection",
        help="intersection requires a prediction for every model; available accepts any matched model output.",
    )
    args = parser.parse_args()
    if args.max_samples < 1:
        parser.error("--max-samples must be at least 1")
    output_dir = ensure_new_output_dir(args.output_dir)
    specs = parse_result_specs(args.result)
    dataset_path, dataset_rows, dataset_sha256 = load_dataset_manifest(args.dataset_manifest, args.path_root)
    predictions_by_model: dict[str, dict[tuple[str, str], Any]] = {}
    alignment_models: dict[str, Any] = {}
    for model_name, result_path in specs.items():
        predictions_by_model[model_name], alignment_models[model_name] = load_predictions(
            result_path,
            extra_roots=args.path_root,
            dataset_rows=dataset_rows,
        )

    requested_ids = set(args.sample_pair_id)
    requested_groups = set(args.sample_pair_group)
    filtered: list[dict[str, Any]] = []
    for row in dataset_rows:
        if args.view != "all":
            raw_view = str(row.get("view", "")).lower()
            is_front = raw_view in {"side0", "front", "front_view", "side"}
            if (args.view == "front") != is_front:
                continue
        if requested_ids and str(row.get("pair_id", "")) not in requested_ids:
            continue
        if requested_groups and str(row.get("pair_group_id", "")) not in requested_groups:
            continue
        key = pair_key_for_dataset_row(row)
        available = [predictions_by_model[name].get(key) is not None for name in specs]
        if args.selection == "intersection" and not all(available):
            continue
        if args.selection == "available" and not any(available):
            continue
        filtered.append(row)

    if requested_ids:
        found_ids = {str(row.get("pair_id", "")) for row in filtered}
        missing_ids = requested_ids - found_ids
        if missing_ids:
            raise ValueError(f"requested pair_id not found under current filters: {sorted(missing_ids)}")
    if requested_groups:
        found_groups = {str(row.get("pair_group_id", "")) for row in filtered}
        missing_groups = requested_groups - found_groups
        if missing_groups:
            raise ValueError(f"requested pair_group_id not found under current filters: {sorted(missing_groups)}")
    if not filtered:
        raise ValueError("no dataset rows remain after view/result alignment filters")

    rng = random.Random(args.seed)
    if len(filtered) > args.max_samples:
        selected = sorted(rng.sample(filtered, args.max_samples), key=lambda row: str(row.get("pair_id", "")))
    else:
        selected = sorted(filtered, key=lambda row: str(row.get("pair_id", "")))

    image_manifest: list[dict[str, Any]] = []
    row_images: list[Image.Image] = []
    for index, dataset_row in enumerate(selected, start=1):
        key = pair_key_for_dataset_row(dataset_row)
        matched = {name: predictions_by_model[name].get(key) for name in specs}
        composite = _compose_row(dataset_row, list(specs), matched)
        filename = f"sample-{index:03d}_{_safe_name(dataset_row.get('pair_id'))}.png"
        output_path = output_dir / filename
        composite.save(output_path, format="PNG", optimize=True)
        row_images.append(composite)
        image_manifest.append(
            {
                "algorithm_version": ALGORITHM_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "sample_index": index,
                "pair_id": dataset_row.get("pair_id"),
                "pair_group_id": dataset_row.get("pair_group_id"),
                "split": dataset_row.get("split"),
                "view": dataset_row.get("view"),
                "plant_id": dataset_row.get("plant_id"),
                "source_capture_date": dataset_row.get("source_capture_date"),
                "target_capture_date": dataset_row.get("target_capture_date"),
                "source_stage": dataset_row.get("source_stage"),
                "target_stage": dataset_row.get("target_stage"),
                "actual_interval_days": dataset_row.get("actual_interval_days"),
                "dataset_manifest": str(dataset_path),
                "dataset_manifest_sha256": dataset_sha256,
                "source_path": dataset_row["source_image_path"],
                "target_path": dataset_row["target_image_path"],
                "model_predictions": {
                    name: {
                        "prediction_path": str(matched[name].output_path) if matched[name] else None,
                        "result_ledger_path": str(matched[name].result_file) if matched[name] else None,
                        "status": "available" if matched[name] else "missing",
                    }
                    for name in specs
                },
                "composite_path": str(output_path),
            }
        )

    max_width = 1500
    resized_rows: list[Image.Image] = []
    for row_image in row_images:
        if row_image.width > max_width:
            height = int(row_image.height * max_width / row_image.width)
            resized_rows.append(row_image.resize((max_width, height), Image.Resampling.LANCZOS))
        else:
            resized_rows.append(row_image)
    contact_width = max(image.width for image in resized_rows)
    contact_height = sum(image.height for image in resized_rows) + (len(resized_rows) + 1) * TILE_GAP
    contact = Image.new("RGB", (contact_width, contact_height), CANVAS_COLOR)
    y = TILE_GAP
    for image in resized_rows:
        contact.paste(image, (0, y))
        y += image.height + TILE_GAP
    contact_path = output_dir / "contact_sheet.png"
    contact.save(contact_path, format="PNG", optimize=True)
    write_jsonl_atomic(output_dir / "visualization_manifest.jsonl", image_manifest)

    summary = {
        "algorithm_version": ALGORITHM_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "dataset_manifest": str(dataset_path),
        "dataset_manifest_sha256": dataset_sha256,
        "dataset_rows_after_filters": len(filtered),
        "selected_sample_count": len(selected),
        "model_names": list(specs),
        "selection": args.selection,
        "view_filter": args.view,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "models": alignment_models,
        "contact_sheet": str(contact_path),
        "sample_images": [row["composite_path"] for row in image_manifest],
    }
    write_json_atomic(output_dir / "summary.json", summary)
    print(f"Visualization complete: {output_dir}")
    print(f"Samples: {len(selected)} | Contact sheet: {contact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
