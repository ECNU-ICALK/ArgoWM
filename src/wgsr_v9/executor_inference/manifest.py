"""Input and seed manifest validation for executor inference."""

from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any

from ..io import canonical_sha256, read_jsonl, sha256_path


def read_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.is_file() else []


def parse_indices(value: str | None) -> set[int] | None:
    if value is None or not value.strip():
        return None
    result: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        index = int(token)
        if index < 0:
            raise ValueError("row indices must be non-negative")
        result.add(index)
    if not result:
        raise ValueError("--row-indices did not contain any indices")
    return result


def load_input_rows(path: Path, indices: set[int] | None) -> tuple[list[dict[str, Any]], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    seen_ids: set[str] = set()
    required = ("request_id", "image", "edit_image", "prompt", "view")
    for ordinal, row in enumerate(read_jsonl(path)):
        row_index = int(row.get("original_row_index", ordinal))
        if row_index in seen:
            raise ValueError(f"duplicate inference row index {row_index} in {path}")
        seen.add(row_index)
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"row {row_index} is missing required fields: {missing}")
        request_id = str(row["request_id"])
        if not request_id or request_id in seen_ids:
            raise ValueError(f"empty or duplicate inference request_id: {request_id}")
        seen_ids.add(request_id)
        if indices is None or row_index in indices:
            selected.append({**row, "inference_row_index": row_index})
    if indices is not None:
        missing_indices = sorted(indices - {int(row["inference_row_index"]) for row in selected})
        if missing_indices:
            raise ValueError(f"requested row indices are not in {path}: {missing_indices[:20]}")
    if not selected:
        raise ValueError(f"no rows selected from {path}")
    return selected, sha256_path(path)


def validate_prompt_rows(rows: list[dict[str, Any]], variant: str, *,
                         expected_protocol: str | None = None) -> None:
    """Reject protocol drift while keeping legacy manifests opt-in compatible."""
    from ..cot_metric_names import validate_metric_name
    from ..executor_training import NAMED_WEEKS_PROTOCOL_VERSION, executor_prompt, prediction_interval_weeks

    protocol = Path(__file__).resolve().parents[3] / "protocols/qwen_image21_executor_v9_v3_metric_named_weeks"
    template = suffix = bindings = None
    for row in rows:
        declared = row.get("executor_prompt_protocol_version", row.get("protocol_version"))
        if expected_protocol is not None and declared != expected_protocol:
            raise ValueError(f"{row['request_id']}: executor prompt protocol mismatch")
        if declared == "wgsr_maize_executor_days_v1":
            from ..maize.experiment import validate_executor_row
            validate_executor_row(row, variant)
            continue
        if declared != NAMED_WEEKS_PROTOCOL_VERSION:
            continue
        if template is None:
            template = (protocol / "EXECUTOR_PROMPT_TEMPLATE.txt").read_text()
            suffix = (protocol / "TEACHER_COT_SUFFIX.txt").read_text()
            bindings = json.loads((protocol / "cot/VIEW_VARIABLE_BINDINGS.json").read_text())
        if row.get("time_condition_unit") != "weeks" or "prediction_interval_weeks" not in row:
            raise ValueError("named executor input requires explicit rounded weeks metadata")
        prediction_interval_weeks(row)
        if row.get("executor_variant") != variant:
            raise ValueError("executor variant does not match input condition provenance")
        if variant == "predicted_cot" and row.get("cot_source") != "planner_prediction":
            raise ValueError("predicted_cot requires planner_prediction provenance")
        prompt = row["prompt"]
        if re.search(r"\b\d+\s+days?\b|</?think>", prompt, re.IGNORECASE):
            raise ValueError("named weeks prompt contains exact days or a think wrapper")
        body = None
        if variant != "simple":
            if prompt.count("<GROWTH_COT>") != 1 or prompt.count("</GROWTH_COT>") != 1:
                raise ValueError("expected exactly one GROWTH_COT wrapper")
            body = prompt.split("<GROWTH_COT>", 1)[1].split("</GROWTH_COT>", 1)[0].strip()
            if row.get("numeric_evidence_format") != "metric_named_v1":
                raise ValueError("named CoT input requires metric_named_v1 metadata")
            lines = [line.strip() for line in body.splitlines() if line.strip()]
            if [line for line in lines if line in {"Stage reasoning", "Growth plan", "Target phenotype"}] != [
                    "Stage reasoning", "Growth plan", "Target phenotype"]:
                raise ValueError("missing, duplicate or reordered CoT sections")
            spec = bindings["top" if row["view"] == "top" else "front"]
            growth = lines[lines.index("Growth plan") + 1:lines.index("Target phenotype")]
            expected_labels = [spec["section_labels"][key] for key in spec["ordered_keys"]]
            if [line.partition(":")[0] for line in growth] != expected_labels:
                raise ValueError("CoT Growth plan fields do not match the view protocol")
            for key, line in zip(spec["ordered_keys"], growth, strict=True):
                validate_metric_name(line, key, spec)
        request = {**row, "accession_id": row.get("accession_id") or str(row.get("plant_id", "")).split("-")[0]}
        expected = executor_prompt(request, prompt_template=template, cot_suffix=suffix,
                                   variant="simple" if variant == "simple" else "teacher_cot",
                                   teacher_cot=body, protocol_version=NAMED_WEEKS_PROTOCOL_VERSION)
        if prompt != expected:
            raise ValueError("executor prompt differs from the trained template, wrapper or rounded weeks")


def validate_input_contract(args: Any, rows: list[dict[str, Any]]) -> None:
    validate_prompt_rows(rows, args.variant,
                         expected_protocol=getattr(args, "executor_prompt_protocol_version", None))
    canonical_path = getattr(args, "evaluation_manifest", None)
    if canonical_path is None:
        return
    canonical_rows = read_jsonl(canonical_path)
    canonical = {str(row["pair_id"]): row for row in canonical_rows}
    if len(canonical) != len(canonical_rows):
        raise ValueError("canonical evaluation IDs must be unique")
    expected_count = getattr(args, "expected_rows", None)
    if expected_count is not None and len(canonical) != expected_count:
        raise ValueError("canonical evaluation row count differs from config")
    selected_ids = {str(row["request_id"]) for row in rows}
    if parse_indices(args.row_indices) is None and selected_ids != set(canonical):
        raise ValueError("inference must cover every canonical validation query")
    for row in rows:
        reference = canonical.get(str(row["request_id"]))
        if reference is None:
            raise ValueError("unknown validation query")
        for actual, expected in (("edit_image", "source_image_path"), ("image", "target_image_path")):
            if Path(row[actual]).resolve() != Path(reference[expected]).resolve():
                raise ValueError(f"canonical image mismatch: {row['request_id']}: {actual}")
        if row.get("actual_interval_days") != reference.get("actual_interval_days"):
            raise ValueError("canonical time interval mismatch")


def load_seed_map(path: Path | None) -> tuple[dict[str, int], str | None]:
    if path is None:
        return {}, None
    if not path.is_file():
        raise FileNotFoundError(path)
    mapping: dict[str, int] = {}
    for row in read_jsonl(path):
        request_id = str(row.get("request_id") or row.get("pair_id") or "")
        if not request_id or "seed" not in row:
            continue
        seed = int(row["seed"])
        old = mapping.setdefault(request_id, seed)
        if old != seed:
            raise ValueError(f"conflicting seeds for {request_id} in {path}")
    return mapping, sha256_path(path)


def input_hash(row: dict[str, Any]) -> str:
    return canonical_sha256(row)


def row_seed(row: dict[str, Any], seed_map: dict[str, int], seed_base: int) -> int:
    return int(seed_map.get(str(row["request_id"]), seed_base + int(row["inference_row_index"])))
