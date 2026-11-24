"""Fixed Executor input ablations, shared by training and inference."""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Iterator

import torch

from .executor_dropout import COT_SEPARATOR, drop_cot
from .io import canonical_sha256, sha256_path

VERSION = "wgsr_v9_executor_input_ablation_v1"
MODES = ("no_metric_stage", "no_image")


def transform_prompt(prompt: str, mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"unsupported fixed ablation: {mode}")
    prefix = drop_cot(prompt)  # Validate the original wrapper even for no_image.
    if mode == "no_image":
        return prompt
    body = prompt[len(prefix + COT_SEPARATOR):]
    pattern = r"\nStage reasoning\nCurrent stage: [^\n]+\nFuture stage: [^\n]+\n\nGrowth plan\n"
    body, count = re.subn(pattern, "\nGrowth plan\n", body, count=1)
    if count != 1 or body.count("\nTarget phenotype\n") != 1:
        raise ValueError("expected reviewed stage/growth/phenotype sections")
    growth, phenotype = body.split("\nTarget phenotype\n")
    growth, count = re.subn(r" Metric = [^\n]+; Source = [^\n]+; Target = [^\n]+; Change = [^\n]+", "", growth)
    if count not in (4, 5):
        raise ValueError(f"expected four top or five front metric suffixes, got {count}")
    body = growth + "\nTarget phenotype\n" + phenotype
    # A few teacher descriptions repeat class IDs outside Stage reasoning.
    # Retain qualitative words (e.g. jointing), removing the explicit IDs.
    body = re.sub(r"\bS[1-8]_([A-Za-z_]+)\b", lambda m: m.group(1).replace("_", " "), body)
    body = re.sub(r" at S[1-8](?=[,.])", "", body)
    body = re.sub(r"\bS[1-8]\b ?", "", body)
    if re.search(r"Stage reasoning|Current stage:|Future stage:|Metric\s*=|(?:Source|Target|Change)\s*=|\bS[1-8]_", body):
        raise ValueError("explicit stage or structured metric information remains")
    if re.search(r"\d|%", body):
        raise ValueError("numeric information remains in qualitative CoT")
    return prefix + COT_SEPARATOR + body


def validate_policy(config: dict[str, Any]) -> None:
    if config.get("algorithm_version") != VERSION or config.get("mode") not in MODES:
        raise ValueError("unsupported fixed ablation policy")
    if not config.get("cache_root"):
        raise ValueError("fixed ablation requires a derived condition cache")


class AblationBatchSampler:
    """Keep the original batch order, adding provenance without consuming RNG."""
    def __init__(self, base: Any, mode: str):
        if mode not in MODES:
            raise ValueError(mode)
        self.base, self.mode = base, mode
        self.batch_size, self.drop_last = base.batch_size, False
        self.groups = base.groups

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)

    def __iter__(self) -> Iterator[list[tuple[int, str, int, int]]]:
        for microbatch, indices in enumerate(self.base):
            yield [(i, self.mode, self.base.epoch, microbatch // self.base.world_size) for i in indices]


class AblationCachedDataset:
    """Replace conditions only; retain the original target supervision exactly."""
    def __init__(self, dataset: Any, index: dict[str, Any], root: Path, mode: str):
        self.dataset, self.mode = dataset, mode
        self.cached_data = dataset.cached_data
        paths = [str(Path(p).resolve()) for p in self.cached_data]
        self.manifest = json.loads((root / "manifest.json").read_text())
        m = self.manifest
        if (m.get("algorithm_version"), m.get("mode"), m.get("status"), m.get("metadata_sha256"),
                m.get("cached_paths_sha256")) != (VERSION, mode, "complete", index["metadata_sha256"],
                                                 canonical_sha256(paths)):
            raise ValueError("fixed ablation cache identity mismatch")
        if m["plan_sha256"] != sha256_path(root / "plan.json"):
            raise ValueError("fixed ablation plan changed")
        self.items = {item["request_id"]: item for item in m["items"]}
        by_path = {str(Path(i["output_path"]).resolve()): i["request_id"] for i in index["items"]}
        self.ids = [by_path[p] for p in paths]
        if set(self.ids) != set(self.items) or len(self.items) != len(paths):
            raise ValueError("fixed ablation cache coverage mismatch")
        self.verified: set[str] = set()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, tag: tuple[int, str, int, int]) -> Any:
        index, mode, epoch, step = tag
        if mode != self.mode:
            raise ValueError("fixed ablation sampler/dataset mismatch")
        shared, positive, negative = (dict(part) for part in self.dataset[index])
        if any(k.startswith("numeric_") for k in positive):
            raise ValueError("fixed ablations require the text executor without Numeric")
        identifier = self.ids[index]
        spec = self.items[identifier]
        p = Path(spec["output_path"])
        if identifier not in self.verified:
            if sha256_path(p) != spec["sha256"]:
                raise ValueError(f"corrupt fixed ablation cache: {p}")
            self.verified.add(identifier)
        derived = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        if derived["prompt"] != transform_prompt(positive["prompt"], mode):
            raise ValueError("derived prompt does not match the original input")
        if mode == "no_image":
            if bool(derived["edit_image_pad_mask"].any()):
                raise ValueError("image tokens remain in no_image cache")
            shared["edit_latents"] = None
        else:
            original_positions = positive["edit_image_pad_mask"][0].nonzero().flatten()
            if not torch.equal(original_positions, derived["edit_image_pad_mask"][0].nonzero().flatten()):
                raise ValueError("source image token layout changed")
        for key in ("prompt", "prompt_embeds", "prompt_embeds_mask", "edit_image_pad_mask"):
            positive[key] = derived[key]
        shared["condition_ablation"] = {"action": mode, "epoch": epoch, "step": step,
                                        "request_ids": [identifier]}
        return shared, positive, negative
