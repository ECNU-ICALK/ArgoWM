"""ms-swift 4.6 plugin for the explicitly selected grouped planner experiment."""
from __future__ import annotations
from functools import lru_cache
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from swift.loss import BaseLoss, loss_map
from swift.trainers import per_token_loss_func
from wgsr_v9.planner_grouped_loss import grouped_loss, supervised_groups, token_groups, validate_weights


class PlannerGroupedLoss(BaseLoss):
    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        config = json.loads(os.environ["WGSR_PLANNER_LOSS_WEIGHTS"])
        self.weights = tuple(float(config[k]) for k in ("stage", "metric", "text"))
        validate_weights(self.weights)
        if args.gradient_accumulation_steps != 1 or args.average_tokens_across_devices:
            raise ValueError("grouped sample-mean loss requires accumulation=1 and token averaging disabled")
        if trainer.template.padding_free or trainer.template.packing or trainer.template.sequence_parallel_size != 1:
            raise ValueError("grouped planner loss requires ordinary padded examples without sequence parallelism")
        self.tokenizer = trainer.template.tokenizer
        if not self.tokenizer.is_fast:
            raise ValueError("exact token offsets require a fast tokenizer")
        self.classify = lru_cache(maxsize=4096)(lambda ids: token_groups(ids, self.tokenizer))

    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, **kwargs):
        if loss_scale is not None:
            raise ValueError("unexpected nonbinary loss_scale; would apply weighting twice")
        # num_items_in_batch counts tokens. This objective averages complete examples,
        # and fixed accumulation=1 plus average_tokens_across_devices=false prevents
        # Trainer from rescaling it by token count or world size.
        groups = supervised_groups(labels, self.tokenizer, self.classify)
        shifted = torch.roll(groups, shifts=-1, dims=-1)
        token_loss = per_token_loss_func(outputs, labels).reshape(labels.shape)
        loss, parts = grouped_loss(token_loss, shifted, self.weights)
        mode = "train" if self.trainer.model.training else "eval"
        for name, value in parts.items():
            self.trainer.custom_metrics[mode][f"planner_{name}_loss"].update(value.detach())
        return loss


loss_map["wgsr_planner_grouped_v1"] = PlannerGroupedLoss
