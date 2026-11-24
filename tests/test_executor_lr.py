from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from accelerate.scheduler import AcceleratedScheduler
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wgsr_v9.executor_launcher import build_plan  # noqa: E402
from wgsr_v9.executor_lr import LearningRateSchedule, step_learning_rate  # noqa: E402
from wgsr_v9.io import sha256_path, write_json  # noqa: E402


class ExecutorLearningRateTests(unittest.TestCase):
    def make_optimizer(self) -> torch.optim.AdamW:
        return torch.optim.AdamW([
            {"params": [torch.nn.Parameter(torch.tensor(1.0))], "lr": 1e-5},
            {"params": [torch.nn.Parameter(torch.tensor(2.0))], "lr": 1e-4},
        ])

    def update(self, optimizer: torch.optim.Optimizer) -> None:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad()

    def test_eight_rank_wrapper_phases_and_skipped_updates(self) -> None:
        optimizer = self.make_optimizer()
        schedule = LearningRateSchedule(name="warmup_stable_cosine")
        scheduler = schedule.build(optimizer, 2500)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [0.0, 0.0])
        resolved = schedule.resolve(2500)
        self.assertEqual((resolved["warmup_steps"], resolved["stable_steps"], resolved["cosine_steps"]),
                         (250, 1500, 750))
        with patch("accelerate.scheduler.GradientState", return_value=SimpleNamespace(
            sync_gradients=True, adjust_scheduler=False
        )), patch("accelerate.scheduler.AcceleratorState", return_value=SimpleNamespace(num_processes=8)):
            wrapped = AcceleratedScheduler(scheduler, optimizer, step_with_optimizer=False)
            observed = []
            for update in range(1, 2501):
                # Accumulating a microbatch must not advance the schedule.
                step_learning_rate(wrapped, manual_step=True, sync_gradients=False,
                                   optimizer_step_was_skipped=False)
                self.assertEqual(scheduler.last_epoch, update - 1)
                if update in (20, 250, 1750, 2200):
                    step_learning_rate(wrapped, manual_step=True, sync_gradients=True,
                                       optimizer_step_was_skipped=True)
                    self.assertEqual(scheduler.last_epoch, update - 1)
                self.update(optimizer)
                step_learning_rate(wrapped, manual_step=True, sync_gradients=True,
                                   optimizer_step_was_skipped=False)
                self.assertEqual(scheduler.last_epoch, update)
                dit_lr, second_group_lr = wrapped.get_last_lr()
                self.assertAlmostEqual(second_group_lr / dit_lr, 10.0)
                observed.append(dit_lr / 1e-5)
        for update, factor in ((125, 0.5), (250, 1.0), (1750, 1.0), (2125, 0.55), (2500, 0.1)):
            self.assertAlmostEqual(observed[update - 1], factor)
        self.assertTrue(all(a <= b for a, b in zip(observed[:249], observed[1:250])))
        self.assertTrue(all(factor == 1.0 for factor in observed[249:1750]))
        self.assertTrue(all(a >= b for a, b in zip(observed[1749:-1], observed[1750:])))
        self.assertLess(float(optimizer.param_groups[0]["params"][0].detach()), 1.0)
        self.assertLess(float(optimizer.param_groups[1]["params"][0].detach()), 2.0)
        self.update(optimizer)
        wrapped.step()
        self.assertAlmostEqual(wrapped.get_last_lr()[0], 1e-6)

    def test_legacy_constant_path_is_preserved(self) -> None:
        schedule = LearningRateSchedule.from_config(None)
        self.assertFalse(schedule.manual_step)
        optimizer = self.make_optimizer()
        scheduler = schedule.build(optimizer, 2500)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ConstantLR)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-5 / 3)
        optimizer.step_was_skipped = False
        with patch("accelerate.scheduler.GradientState", return_value=SimpleNamespace(
            sync_gradients=True, adjust_scheduler=False
        )), patch("accelerate.scheduler.AcceleratorState", return_value=SimpleNamespace(num_processes=8)):
            wrapped = AcceleratedScheduler(scheduler, optimizer)
            self.update(optimizer)
            step_learning_rate(wrapped, manual_step=False, sync_gradients=True,
                               optimizer_step_was_skipped=False)
        self.assertEqual(scheduler.last_epoch, 8)
        self.assertAlmostEqual(wrapped.get_last_lr()[0], 1e-5)

    def test_scheduler_state_restores_both_parameter_groups(self) -> None:
        config = LearningRateSchedule(name="warmup_stable_cosine", warmup_start_factor=0.01)
        optimizer = self.make_optimizer()
        scheduler = config.build(optimizer, 100)
        for _ in range(80):
            self.update(optimizer)
            scheduler.step()
        saved_optimizer = copy.deepcopy(optimizer.state_dict())
        saved_scheduler = copy.deepcopy(scheduler.state_dict())
        resumed_optimizer = self.make_optimizer()
        resumed_scheduler = config.build(resumed_optimizer, 100)
        resumed_optimizer.load_state_dict(saved_optimizer)
        resumed_scheduler.load_state_dict(saved_scheduler)
        for _ in range(20):
            self.update(optimizer)
            scheduler.step()
            self.update(resumed_optimizer)
            resumed_scheduler.step()
            self.assertEqual(resumed_scheduler.get_last_lr(), scheduler.get_last_lr())
        self.assertAlmostEqual(resumed_scheduler.get_last_lr()[1], 1e-5)

    def test_invalid_ratios_and_insufficient_budget_fail_early(self) -> None:
        for value in ({"warmup_ratio": 0.8, "cosine_ratio": 0.3},
                      {"warmup_start_factor": float("nan")}, {"min_lr_factor": -0.1},
                      {"cosine_ratio": 0.0}, {"name": "typo"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                LearningRateSchedule.from_config({"name": "warmup_stable_cosine", **value})
        with self.assertRaises(ValueError):
            LearningRateSchedule.from_config({"warmup_ratio": 0.1})
        for budget in (0, 1):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                LearningRateSchedule(name="warmup_stable_cosine").resolve(budget)

    def test_three_launch_configs_forward_the_same_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            upstream = root / "framework/examples/qwen_image_21/model_training/train.py"
            upstream.parent.mkdir(parents=True)
            upstream.touch()
            variants = {}
            for variant in ("simple", "teacher_cot"):
                metadata = root / "data" / variant / "train.jsonl"
                metadata.parent.mkdir(parents=True)
                metadata.write_text('{}\n' * 20000)
                variants[variant] = {"status": "complete", "rows_by_split": {"train": 20000},
                                    "generated_file_sha256": {
                    "train.jsonl": sha256_path(metadata),
                }}
            schedules = []
            for filename, variant in (("recipes/rice/executor_simple.json", "simple"),
                                      ("recipes/rice/executor_teacher_cot.json", "teacher_cot"),
                                      ("recipes/maize/executor_teacher_cot.json", "teacher_cot")):
                config = yaml.safe_load((REPO_ROOT / "configs" / filename).read_text())
                if config.get("species") == "maize":
                    for value in variants.values(): value["rows_by_split"]["train"] = 6400
                config.update(framework_repo=str(root / "framework"), data_root=str(root / "data"))
                write_json(root / "data/package_manifest.json", {
                    "status": "complete", "species": config.get("species", "rice"), "algorithm_version": config['version'],
                    "protocol_version": config['protocol_version'], "variants": variants,
                })
                with patch("wgsr_v9.executor_launcher._model_paths", return_value=([], [])):
                    plan = build_plan(config, variant, "train")
                    command = plan["commands"][0]
                    self.assertEqual(command[command.index('--lr-scheduler') + 1], 'warmup_stable_cosine')
                    self.assertEqual(command[command.index('--lr-warmup-ratio') + 1], '0.1')
                    self.assertEqual(command[command.index('--lr-cosine-ratio') + 1], '0.3')
                    self.assertEqual(command[command.index('--lr-warmup-start-factor') + 1], '0.0')
                    self.assertEqual(command[command.index('--lr-min-factor') + 1], '0.1')
                    schedules.append(plan['lr_scheduler'])
                    del config['lr_scheduler']
                    legacy = build_plan(config, variant, "train")
                    self.assertEqual(legacy['lr_scheduler'], {'name': 'legacy_constant'})
                    self.assertNotIn('--lr-scheduler', legacy['commands'][0])
            self.assertEqual(schedules[0], schedules[1])
            self.assertEqual(schedules[0], schedules[2])


if __name__ == "__main__":
    unittest.main()
