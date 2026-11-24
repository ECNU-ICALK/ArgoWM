"""Reuse the reviewed normalized loss with maize-specific output token spans."""
from functools import lru_cache
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from planner_grouped_loss_plugin import PlannerGroupedLoss, loss_map
from wgsr_v9.maize.grouped_loss import token_groups


class MaizePlannerGroupedLoss(PlannerGroupedLoss):
    def __init__(self, args: Any, trainer: Any) -> None:
        super().__init__(args, trainer)
        self.classify = lru_cache(maxsize=4096)(lambda ids: token_groups(ids, self.tokenizer))


loss_map["wgsr_maize_planner_grouped_v1"] = MaizePlannerGroupedLoss
