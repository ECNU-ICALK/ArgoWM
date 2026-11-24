"""Homogeneous cached batching for Qwen Image 2.1 DiffSynth full-DiT SFT.

The Qwen Image 2.1 text encoder returns a different sequence length for every
prompt.  A cache therefore cannot use the full prompt length as a batching
key: doing so creates many tiny buckets and repeats samples at every bucket
boundary.  Image geometry and the positions of image tokens must remain
homogeneous, while text tokens can be padded at the end of a batch and
masked with ``prompt_embeds_mask``.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import random
from typing import Any, Hashable, Iterable, Iterator

import torch


def layout_key(inputs: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]) -> tuple[Hashable, ...]:
    shared, positive, _negative = inputs
    if positive.get("prompt_embeds_mask") is not None:
        raise ValueError("Only unpadded cached prompt layouts are supported")
    mask = positive["edit_image_pad_mask"].detach().cpu().contiguous()
    return (
        tuple(shared["input_latents"].shape[1:]),
        tuple(tuple(latent.shape[1:]) for latent in shared.get("edit_latents") or []),
        tuple(positive["prompt_embeds"].shape[1:]),
        hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
    )


def batch_layout_key(inputs: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]) -> tuple[Hashable, ...]:
    """Return the layout that must be shared by all rows in a microbatch.

    Prompt length is deliberately excluded.  Qwen's image token positions are
    still included, because ``QwenImage21DiT`` receives one shared image mask
    for a whole batch.  The full mask is not stored in the key; a digest of its
    true positions keeps the index compact while ``collate_cached`` performs a
    second exact check before concatenating tensors.
    """

    shared, positive, _negative = inputs
    mask = positive["edit_image_pad_mask"].detach().cpu().contiguous()
    if mask.ndim != 2 or mask.shape[0] != 1:
        raise ValueError(f"expected edit_image_pad_mask with shape (1, sequence), got {tuple(mask.shape)}")
    image_positions = mask[0].nonzero(as_tuple=True)[0].numpy()
    position_digest = hashlib.sha256(image_positions.tobytes()).hexdigest()
    return (
        tuple(shared["input_latents"].shape[1:]),
        tuple(tuple(latent.shape[1:]) for latent in shared.get("edit_latents") or []),
        int(image_positions.size),
        position_digest,
    )


def _pad_prompt_embeds(prompt_embeds: torch.Tensor, length: int) -> torch.Tensor:
    if prompt_embeds.ndim != 3 or prompt_embeds.shape[0] != 1:
        raise ValueError(
            "cached prompt_embeds must have shape (1, sequence, hidden), "
            f"got {tuple(prompt_embeds.shape)}"
        )
    padding = length - prompt_embeds.shape[1]
    if padding < 0:
        raise ValueError("cannot pad prompt embeddings to a shorter sequence")
    if padding == 0:
        return prompt_embeds
    return torch.cat(
        [prompt_embeds, prompt_embeds.new_zeros((1, padding, prompt_embeds.shape[2]))],
        dim=1,
    )


def _pad_mask(mask: torch.Tensor, length: int, *, name: str) -> torch.Tensor:
    if mask.ndim != 2 or mask.shape[0] != 1:
        raise ValueError(f"cached {name} must have shape (1, sequence), got {tuple(mask.shape)}")
    padding = length - mask.shape[1]
    if padding < 0:
        raise ValueError(f"cannot pad {name} to a shorter sequence")
    if padding == 0:
        return mask
    return torch.cat([mask, mask.new_zeros((1, padding))], dim=1)


def collate_cached(
    rows: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not rows:
        raise ValueError("Cannot collate an empty batch")
    key = batch_layout_key(rows[0])
    if any(batch_layout_key(row) != key for row in rows[1:]):
        raise ValueError("Cannot mix different geometry or image-token layout in one batch")
    prompt_lengths = [row[1]["prompt_embeds"].shape[1] for row in rows]
    max_prompt_length = max(prompt_lengths)
    reference_image_positions = rows[0][1]["edit_image_pad_mask"][0].nonzero(as_tuple=True)[0]
    padded_image_masks: list[torch.Tensor] = []
    padded_prompt_masks: list[torch.Tensor] = []
    for row, prompt_length in zip(rows, prompt_lengths):
        positive_row = row[1]
        image_mask = positive_row["edit_image_pad_mask"]
        image_positions = image_mask[0].nonzero(as_tuple=True)[0]
        if not torch.equal(image_positions, reference_image_positions):
            raise ValueError("Cannot mix rows with different image-token positions in one batch")
        if image_mask.shape[1] != prompt_length:
            raise ValueError(
                "prompt_embeds and edit_image_pad_mask must have the same sequence length, "
                f"got {prompt_length} and {image_mask.shape[1]}"
            )
        prompt_mask = positive_row.get("prompt_embeds_mask")
        if prompt_mask is None:
            prompt_mask = torch.ones(
                (1, prompt_length), dtype=torch.bool, device=positive_row["prompt_embeds"].device
            )
        else:
            prompt_mask = prompt_mask.bool()
        if prompt_mask.shape[1] != prompt_length:
            raise ValueError(
                "prompt_embeds_mask and prompt_embeds must have the same sequence length, "
                f"got {prompt_mask.shape[1]} and {prompt_length}"
            )
        padded_image_masks.append(_pad_mask(image_mask, max_prompt_length, name="edit_image_pad_mask"))
        padded_prompt_masks.append(_pad_mask(prompt_mask, max_prompt_length, name="prompt_embeds_mask"))
    shared, positive, negative = (dict(part) for part in rows[0])
    shared["input_latents"] = torch.cat([row[0]["input_latents"] for row in rows], dim=0)
    shared["edit_latents"] = [
        torch.cat([row[0]["edit_latents"][index] for row in rows], dim=0)
        for index in range(len(rows[0][0].get("edit_latents") or []))
    ] or None
    positive["prompt_embeds"] = torch.cat(
        [_pad_prompt_embeds(row[1]["prompt_embeds"], max_prompt_length) for row in rows], dim=0
    )
    # The upstream Qwen Image 2.1 DiT reads one shared image-layout mask.
    positive["edit_image_pad_mask"] = padded_image_masks[0]
    prompt_embeds_mask = torch.cat(padded_prompt_masks, dim=0)
    # Preserve the compact ``None`` representation when no padding is needed.
    positive["prompt_embeds_mask"] = None if bool(prompt_embeds_mask.all()) else prompt_embeds_mask
    numeric_keys = ("numeric_features", "numeric_mask", "numeric_view")
    if any(any(key in row[1] for key in numeric_keys) for row in rows):
        if not all(all(key in row[1] for key in numeric_keys) for row in rows):
            raise ValueError("Cannot mix numeric and legacy cached rows or incomplete numeric conditions")
        for key in numeric_keys:
            positive[key] = torch.cat([row[1][key] for row in rows], dim=0)
    for condition_key in ("condition_dropout", "condition_ablation"):
        if not any(condition_key in row[0] for row in rows):
            continue
        labels = [row[0].get(condition_key) for row in rows]
        if any(label is None for label in labels) or any(
            any(label[key] != labels[0][key] for key in ("action", "epoch", "step")) for label in labels
        ):
            raise ValueError("one batch must share one dropout action/epoch/step")
        shared[condition_key] = {**labels[0], "request_ids": [
            identifier for label in labels for identifier in label["request_ids"]]}
    return shared, positive, negative


class LayoutBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Yield local microbatches with shared image/text layouts across ranks."""

    def __init__(
        self,
        keys: Iterable[Hashable],
        batch_size: int,
        world_size: int,
        *,
        seed: int,
        gradient_accumulation_steps: int,
    ) -> None:
        if batch_size <= 0 or world_size <= 0 or gradient_accumulation_steps <= 0:
            raise ValueError("batch size, world size, and accumulation steps must be positive")
        self.batch_size = batch_size
        self.world_size = world_size
        self.seed = seed
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.epoch = 0
        self.groups: dict[Hashable, list[int]] = defaultdict(list)
        for index, key in enumerate(keys):
            self.groups[key].append(index)
        if not self.groups:
            raise ValueError("layout sampler requires at least one sample")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        global_batches = sum(
            math.ceil(len(group) / (self.batch_size * self.world_size)) for group in self.groups.values()
        )
        global_batches = math.ceil(global_batches / self.gradient_accumulation_steps)
        global_batches *= self.gradient_accumulation_steps
        return global_batches * self.world_size

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        global_batches: list[list[int]] = []
        global_batch_size = self.batch_size * self.world_size
        for original in self.groups.values():
            group = original.copy()
            generator.shuffle(group)
            padded_length = math.ceil(len(group) / global_batch_size) * global_batch_size
            group.extend(group[index % len(original)] for index in range(padded_length - len(group)))
            global_batches.extend(
                group[start : start + global_batch_size]
                for start in range(0, len(group), global_batch_size)
            )
        generator.shuffle(global_batches)
        while len(global_batches) % self.gradient_accumulation_steps:
            global_batches.append(global_batches[-1].copy())
        for indices in global_batches:
            for start in range(0, global_batch_size, self.batch_size):
                yield indices[start : start + self.batch_size]
