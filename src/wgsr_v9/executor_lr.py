"""Executor learning-rate schedules measured in optimizer updates, not ranks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


SCHEDULE_VERSION = "wgsr_executor_warmup_stable_cosine_v1"


@dataclass(frozen=True)
class LearningRateSchedule:
    # Missing configuration preserves the historical ConstantLR behavior.
    name: str = "legacy_constant"
    warmup_ratio: float = 0.1
    cosine_ratio: float = 0.3
    warmup_start_factor: float = 0.0
    min_lr_factor: float = 0.1

    def __post_init__(self) -> None:
        if self.name not in {"legacy_constant", "warmup_stable_cosine"}:
            raise ValueError(f"unsupported executor LR schedule: {self.name}")
        for field in ("warmup_ratio", "cosine_ratio", "warmup_start_factor", "min_lr_factor"):
            value = getattr(self, field)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be finite and in [0, 1]")
        if self.warmup_ratio <= 0 or self.cosine_ratio <= 0:
            raise ValueError("warmup_ratio and cosine_ratio must be positive")
        if self.warmup_ratio + self.cosine_ratio > 1.0:
            raise ValueError("warmup_ratio + cosine_ratio must not exceed 1")

    @classmethod
    def from_config(cls, value: dict[str, Any] | None) -> LearningRateSchedule:
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise ValueError("lr_scheduler must be a mapping")
        if "name" not in value:
            raise ValueError("lr_scheduler requires an explicit name")
        return cls(**value)

    @property
    def manual_step(self) -> bool:
        return self.name != "legacy_constant"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) if self.manual_step else {"name": self.name}

    def resolve(self, total_steps: int) -> dict[str, Any]:
        if total_steps <= 0:
            raise ValueError("LR schedule requires a positive optimizer-update budget")
        result = {"name": self.name, "total_optimizer_steps": total_steps}
        if not self.manual_step:
            return result
        warmup_steps = math.ceil(total_steps * self.warmup_ratio)
        cosine_steps = math.ceil(total_steps * self.cosine_ratio)
        if warmup_steps + cosine_steps > total_steps:
            raise ValueError("optimizer-update budget is too short for the rounded LR phases")
        return {
            **result,
            **self.to_dict(),
            "algorithm_version": SCHEDULE_VERSION,
            "warmup_steps": warmup_steps,
            "stable_steps": total_steps - warmup_steps - cosine_steps,
            "cosine_steps": cosine_steps,
            "cosine_start_step": total_steps - cosine_steps,
            "step_unit": "successful_optimizer_updates",
            "boundary_convention": "LR after N updates is used by update N+1; initial N=0",
        }

    def build(self, optimizer: Any, total_steps: int) -> Any:
        # Import lazily: the command-preview environment need not load torch.
        from torch.optim.lr_scheduler import ConstantLR, LambdaLR

        if not self.manual_step:
            return ConstantLR(optimizer)
        resolved = self.resolve(total_steps)
        curve = WarmupStableCosine(
            warmup_steps=resolved["warmup_steps"],
            cosine_start_step=resolved["cosine_start_step"],
            cosine_steps=resolved["cosine_steps"],
            start_factor=self.warmup_start_factor,
            min_factor=self.min_lr_factor,
        )
        return LambdaLR(optimizer, curve)


@dataclass
class WarmupStableCosine:
    warmup_steps: int
    cosine_start_step: int
    cosine_steps: int
    start_factor: float
    min_factor: float

    def __call__(self, completed_updates: int) -> float:
        step = max(0, completed_updates)
        if step < self.warmup_steps:
            return self.start_factor + (1.0 - self.start_factor) * step / self.warmup_steps
        if step <= self.cosine_start_step:
            return 1.0
        progress = min(1.0, (step - self.cosine_start_step) / self.cosine_steps)
        return self.min_factor + (1.0 - self.min_factor) * (1.0 + math.cos(math.pi * progress)) / 2.0


def step_learning_rate(
    scheduler: Any, *, manual_step: bool, sync_gradients: bool, optimizer_step_was_skipped: bool
) -> None:
    """New schedules advance once per successful update; legacy wrappers gate themselves."""
    if not manual_step or (sync_gradients and not optimizer_step_was_skipped):
        scheduler.step()
