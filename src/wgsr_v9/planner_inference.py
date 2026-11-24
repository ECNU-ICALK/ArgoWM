"""Source-only planner requests, resumable outputs, and GPU measurements."""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from wgsr_v9.planner_adapter import to_growth_cot
from wgsr_v9.cot_metric_names import validate_metric_name
from wgsr_v9.planner_direct import validate_direct_body


ALGORITHM_VERSION = "wgsr_v9_planner_batched_inference_v1"
PROTOCOL_VERSION = "wgsr_v9_planner_cot_v2_source_only_inference_v1"


def model_device_map(model: Any) -> dict[str, str]:
    """Report Accelerate placement, or the verified device of an unsharded model."""
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        return {name: str(device) for name, device in device_map.items()}
    devices = {str(tensor.device) for tensor in model.parameters()}
    devices.update(str(tensor.device) for tensor in model.buffers())
    if len(devices) != 1:
        raise ValueError("Cannot report a single device for an unmapped empty or mixed-device model")
    return {"": devices.pop()}


def protocol_version_for_cot(
    cot_version: str, numeric_evidence_format: str | None = None, time_condition_unit: str | None = None,
    *, enable_thinking: bool = True, output_wrapper: str = "think",
) -> str:
    """Resolve the two mainline Planner contracts shipped in this repository."""
    if numeric_evidence_format == "maize_sv1_v1":
        if (cot_version, time_condition_unit, enable_thinking, output_wrapper) != ("v3", "days", False, "GROWTH_COT"):
            raise ValueError("Mismatched maize planner output contract")
        from .maize.experiment import INFERENCE_PROTOCOL
        return INFERENCE_PROTOCOL
    if (cot_version, numeric_evidence_format, time_condition_unit, enable_thinking, output_wrapper) == ("v3", "metric_named_v1", "weeks", False, "GROWTH_COT"):
        return "wgsr_v9_planner_cot_v3_direct_metric_named_weeks_source_only_inference_v1"
    raise ValueError(
        "Unsupported planner contract: use a mainline rice or maize recipe from configs/recipes "
        "with enable_thinking=false and output_wrapper=GROWTH_COT"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_request(row: dict[str, Any], package: Path) -> dict[str, Any]:
    messages = row["messages"]
    if [message["role"] for message in messages] != ["system", "user", "assistant"]:
        raise ValueError(f"Unexpected training message roles: {row['id']}")
    if len(row["images"]) != 1:
        raise ValueError(f"Expected one source image: {row['id']}")
    image = (package / row["images"][0]).resolve()
    if not image.is_file() or image != (package / row["input_path"]).resolve():
        raise ValueError(f"Source image mismatch: {row['id']}")
    # Construct a new allowlisted object; no teacher text or metadata reaches the model.
    return {"messages": [dict(message) for message in messages[:2]], "images": [str(image)]}


def validate_completion(
    text: str, finish_reason: str, *, numeric_evidence_format: str | None = None, view: str | None = None,
    output_wrapper: str = "think",
) -> tuple[str | None, str | None]:
    if finish_reason == "length":
        return None, "generation reached max_new_tokens"
    if numeric_evidence_format == "maize_sv1_v1":
        if output_wrapper != "GROWTH_COT" or view != "side0":
            return None, "mismatched maize output wrapper or view"
        from .maize.experiment import validate_growth_cot
        try:
            return validate_growth_cot(re.sub(r"^\s*<think>\s*</think>\s*", "", text, count=1)), None
        except (ValueError, TypeError) as exc:
            return None, str(exc)
    try:
        if output_wrapper == "GROWTH_COT":
            text = re.sub(r"^\s*<think>\s*</think>\s*", "", text, count=1).strip()
            if not text.startswith("<GROWTH_COT>") or not text.endswith("</GROWTH_COT>"):
                raise ValueError("direct output requires one GROWTH_COT answer, without nonempty thinking")
        elif output_wrapper != "think":
            raise ValueError("unknown planner output wrapper")
        cot = to_growth_cot(text)
        if output_wrapper == "GROWTH_COT":
            if numeric_evidence_format != "metric_named_v1":
                raise ValueError("direct output requires metric_named_v1")
            validate_direct_body(cot[len("<GROWTH_COT>"):-len("</GROWTH_COT>")].strip(), view)
    except (ValueError, TypeError) as exc:
        return None, str(exc)
    for heading in ("Stage reasoning", "Current stage:", "Future stage:", "Growth plan", "Target phenotype"):
        if heading not in cot:
            return None, f"missing required section: {heading}"
    if numeric_evidence_format is not None:
        if numeric_evidence_format != "metric_named_v1" or view not in {"front", "side0", "side90", "top"}:
            return None, "unsupported metric format or view"
        bindings_path = Path(__file__).resolve().parents[2] / "protocols/qwen_image21_executor_v9_v3_metric_named_weeks/cot/VIEW_VARIABLE_BINDINGS.json"
        spec = json.loads(bindings_path.read_text())["top" if view == "top" else "front"]
        labels = {label: key for key, label in spec["section_labels"].items()}
        seen = set()
        section = None
        try:
            for line in cot.splitlines():
                if line.strip() in {"Stage reasoning", "Growth plan", "Target phenotype"}:
                    section = line.strip()
                elif section == "Growth plan" and line.strip():
                    label = line.partition(":")[0].strip()
                    if label not in labels or label in seen:
                        raise ValueError("unknown or duplicate Growth plan field")
                    seen.add(label)
                    validate_metric_name(line, labels[label], spec)
            if seen != set(labels):
                raise ValueError("missing Growth plan fields")
            if re.search(r"\b\d+\s+days?\b", cot):
                raise ValueError("exact-day expression in weeks-only output")
        except ValueError as exc:
            return None, str(exc)
    return cot, None


class GpuMonitor:
    """Sample physical GPUs without initializing a CUDA context."""

    def __init__(self, devices: list[int], path: Path, protocol_version: str = PROTOCOL_VERSION):
        self.devices = devices
        self.path = path
        self.protocol_version = protocol_version
        self.stop = threading.Event()
        self.samples: list[dict[str, Any]] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.is_set():
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,power.draw,power.limit",
                 "--format=csv,noheader,nounits", "-i", ",".join(map(str, self.devices))],
                capture_output=True, text=True, timeout=10, check=False,
            )
            for line in result.stdout.splitlines():
                try:
                    index, memory, utilization, power, limit = map(float, line.split(","))
                except ValueError:
                    continue
                record = {"algorithm_version": ALGORITHM_VERSION, "protocol_version": self.protocol_version,
                          "input_path": f"gpu:{int(index)}", "output_path": str(self.path),
                          "status": "sampled", "error": None, "retry_count": 0,
                          "time": time.time(), "gpu": int(index), "memory_mib": memory,
                          "utilization_pct": utilization, "power_w": power, "power_limit_w": limit}
                self.samples.append(record)
                append_jsonl(self.path, record)
            self.stop.wait(1)

    def __enter__(self) -> GpuMonitor:
        self.thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop.set()
        self.thread.join(timeout=15)

    def summary(self, start: float, end: float) -> dict[str, Any]:
        summary = {}
        for device in self.devices:
            rows = [r for r in self.samples if r["gpu"] == device and start <= r["time"] <= end]
            if rows:
                summary[str(device)] = {
                    "samples": len(rows),
                    "memory_peak_mib": max(r["memory_mib"] for r in rows),
                    "utilization_mean_pct": statistics.mean(r["utilization_pct"] for r in rows),
                    "power_mean_w": statistics.mean(r["power_w"] for r in rows),
                    "power_peak_w": max(r["power_w"] for r in rows),
                }
        return summary
