"""Reviewed, mutually exclusive condition dropout for cached executor SFT.

All ranks use one action per global batch. The historical text-only action
tape is retained, including its drop_numeric label, which acts as keep for text
models. No numeric adapter is installed.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterator

import torch

from .io import canonical_sha256, sha256_path

VERSION = "wgsr_v9_executor_modality_dropout_v1"
ACTIONS = ("keep", "drop_text", "drop_image", "drop_numeric")
PROBABILITIES = (0.7, 0.1, 0.1, 0.1)
COT_SEPARATOR = "\nGrowth plan for the target image:\n<GROWTH_COT>"


def drop_cot(prompt: str) -> str:
    if prompt.count(COT_SEPARATOR) != 1 or prompt.count("</GROWTH_COT>") != 1:
        raise ValueError("drop_text requires exactly one reviewed v3 CoT suffix")
    prefix, body = prompt.split(COT_SEPARATOR)
    if not prefix or body.split("</GROWTH_COT>", 1)[1].strip():
        raise ValueError("unexpected text outside the reviewed CoT wrapper")
    return prefix


def stream_seed(seed: int, *parts: Any) -> int:
    value = json.dumps([VERSION, seed, *parts], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % (2**63 - 1)


def action_tape(steps: int, seed: int, epoch: int) -> list[str]:
    counts = [round(steps * p) for p in PROBABILITIES]
    if steps <= 0 or sum(counts) != steps or any(abs(steps * p - n) > 1e-8
                                               for p, n in zip(PROBABILITIES, counts)):
        raise ValueError("exact reviewed 70/10/10/10 schedule requires a multiple of 10 steps")
    tape = [action for action, count in zip(ACTIONS, counts) for _ in range(count)]
    random.Random(stream_seed(seed, "actions", epoch)).shuffle(tape)
    return tape


def validate_policy(config: dict[str, Any]) -> None:
    if config.get("algorithm_version") != VERSION:
        raise ValueError("unsupported condition dropout version")
    if config.get("probabilities") != dict(zip(ACTIONS, PROBABILITIES)):
        raise ValueError("unreviewed condition dropout probabilities")
    if config.get("text_scope") != "cot_body" or config.get("sampling_unit") != "global_step":
        raise ValueError("reviewed dropout preserves the task prefix and samples per global step")
    if config.get("numeric_whole_dropout") != 0 or config.get("numeric_slot_dropout") != 0:
        raise ValueError("internal numeric dropout must be disabled for exclusive modality dropout")


class DropoutBatchSampler:
    """Attach an immutable action to each index before Accelerate shards batches."""

    def __init__(self, base: Any, *, seed: int):
        if base.gradient_accumulation_steps != 1:
            raise ValueError("dropout v1 requires accumulation=1")
        self.base, self.seed = base, seed
        self.batch_size, self.drop_last = base.batch_size, False
        self.groups = base.groups

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)

    def __iter__(self) -> Iterator[list[tuple[int, str, int, int]]]:
        actions = action_tape(len(self) // self.base.world_size, self.seed, self.base.epoch)
        for microbatch, indices in enumerate(self.base):
            step = microbatch // self.base.world_size
            yield [(i, actions[step], self.base.epoch, step) for i in indices]


def load_dropout_manifest(root: Path, index: dict[str, Any], cached_paths: list[str],
                          seed: int, epochs: int) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text())
    expected = (VERSION, "complete", index["metadata_sha256"], seed, epochs,
                canonical_sha256(cached_paths))
    found = tuple(manifest.get(k) for k in ("algorithm_version", "status", "metadata_sha256",
                                          "seed", "epochs", "cached_paths_sha256"))
    if found != expected:
        raise ValueError("dropout cache identity/seed/epochs mismatch or cache incomplete")
    if sha256_path(root / "plan.json") != manifest["plan_sha256"]:
        raise ValueError("dropout action/cache plan changed")
    return manifest


class DropoutCachedDataset:
    def __init__(self, dataset: Any, index: dict[str, Any], root: Path, *, seed: int, epochs: int):
        self.dataset, self.root = dataset, root
        self.cached_data = dataset.cached_data
        paths = [str(Path(p).resolve()) for p in self.cached_data]
        self.manifest = load_dropout_manifest(root, index, paths, seed, epochs)
        self.items = {item["request_id"]: item for item in self.manifest["items"]}
        by_path = {str(Path(i["output_path"]).resolve()): i["request_id"] for i in index["items"]}
        self.ids = [by_path[p] for p in paths]
        self.verified: set[str] = set()

    def __len__(self) -> int:
        return len(self.dataset)

    def _load(self, spec: dict[str, Any]) -> Any:
        path = Path(spec["output_path"])
        if str(path) not in self.verified:
            if sha256_path(path) != spec["sha256"]:
                raise ValueError(f"dropout cache integrity mismatch: {path}")
            self.verified.add(str(path))
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)

    def __getitem__(self, tag: tuple[int, str, int, int]) -> Any:
        index, action, epoch, step = tag
        if action not in ACTIONS:
            raise ValueError("unknown dropout action")
        # Read the unmodified text cache before applying the scheduled action.
        shared, positive, negative = (dict(p) for p in self.dataset[index])
        request_id = self.ids[index]
        if action in {"drop_text", "drop_image"}:
            item = self.items[request_id]
            spec = item[action]
            cached = self._load(spec)
            if action == "drop_text":
                other_shared, other_positive, _ = cached
                if other_positive["prompt"] != drop_cot(positive["prompt"]):
                    raise ValueError("drop_text did not retain exactly the task prefix")
                if not torch.equal(shared["input_latents"], other_shared["input_latents"]):
                    raise ValueError("drop_text target differs from original supervision")
                if len(shared["edit_latents"]) != len(other_shared["edit_latents"]) or any(
                    not torch.equal(a, b) for a, b in zip(shared["edit_latents"], other_shared["edit_latents"])
                ):
                    raise ValueError("drop_text source latent differs")
            else:
                other_positive = cached
                if other_positive["prompt"] != positive["prompt"]:
                    raise ValueError("drop_image changed the text")
                if bool(other_positive["edit_image_pad_mask"].any()):
                    raise ValueError("drop_image still contains image tokens")
                shared["edit_latents"] = None
            for key in ("prompt", "prompt_embeds", "prompt_embeds_mask", "edit_image_pad_mask"):
                positive[key] = other_positive[key]
        shared["condition_dropout"] = {"action": action, "epoch": epoch, "step": step,
                                       "request_ids": [request_id]}
        return shared, positive, negative


def action_summary(steps: int, seed: int, epochs: int) -> list[dict[str, Any]]:
    return [{"epoch": epoch, "counts": dict(Counter(action_tape(steps, seed, epoch))),
             "sha256": canonical_sha256(action_tape(steps, seed, epoch))} for epoch in range(epochs)]
