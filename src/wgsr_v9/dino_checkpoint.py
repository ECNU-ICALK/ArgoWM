"""Convert local Hugging Face DINOv2 weights for the existing timm evaluator."""

from __future__ import annotations

import re
from collections.abc import Mapping

import torch


def hf_dinov2_to_timm(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map standard (non-register, GELU MLP) DINOv2 weights without casting.

    Q/K/V are concatenated in that order. Every source tensor must be consumed;
    only the mask token is omitted because evaluation never masks patches.
    The caller must also validate all destination keys and tensor shapes.
    """
    remaining = dict(state)
    output: dict[str, torch.Tensor] = {}
    for source, target in {
        "embeddings.cls_token": "cls_token",
        "embeddings.position_embeddings": "pos_embed",
        "embeddings.patch_embeddings.projection.weight": "patch_embed.proj.weight",
        "embeddings.patch_embeddings.projection.bias": "patch_embed.proj.bias",
        "layernorm.weight": "norm.weight",
        "layernorm.bias": "norm.bias",
    }.items():
        output[target] = remaining.pop(source)
    remaining.pop("embeddings.mask_token")
    layers = sorted({int(match.group(1)) for key in remaining
                     if (match := re.match(r"encoder\.layer\.(\d+)\.", key))})
    if not layers or layers != list(range(len(layers))):
        raise ValueError(f"DINOv2 layers must be contiguous from zero: {layers}")
    for index in layers:
        source = f"encoder.layer.{index}."
        target = f"blocks.{index}."
        legacy_attention = source + "attention.attention.query.weight" in remaining
        qkv_parts = ("attention.attention.query", "attention.attention.key", "attention.attention.value") if legacy_attention else ("attention.q_proj", "attention.k_proj", "attention.v_proj")
        output_projection = "attention.output.dense" if legacy_attention else "attention.o_proj"
        for suffix in ("weight", "bias"):
            output[target + f"attn.qkv.{suffix}"] = torch.cat([
                remaining.pop(source + f"{part}.{suffix}")
                for part in qkv_parts
            ], dim=0)
            output[target + f"attn.proj.{suffix}"] = remaining.pop(source + f"{output_projection}.{suffix}")
            for part in ("norm1", "norm2", "mlp.fc1", "mlp.fc2"):
                output[target + f"{part}.{suffix}"] = remaining.pop(source + f"{part}.{suffix}")
        for number in (1, 2):
            output[target + f"ls{number}.gamma"] = remaining.pop(source + f"layer_scale{number}.lambda1")
    if remaining:
        raise ValueError(f"unsupported Hugging Face DINOv2 tensors: {sorted(remaining)}")
    return output
