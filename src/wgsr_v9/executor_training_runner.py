#!/usr/bin/env python3
"""Train Qwen Image 2.1 full DiT on validated DiffSynth latent caches."""

from __future__ import annotations

import argparse
import ast
from contextlib import nullcontext
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import accelerate
from accelerate.utils import set_seed
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]

from wgsr_v9.diffsynth_batching import LayoutBatchSampler, collate_cached  # noqa: E402
from wgsr_v9.executor_lr import LearningRateSchedule, step_learning_rate  # noqa: E402
from wgsr_v9.executor_training import ALGORITHM_VERSION  # noqa: E402
from wgsr_v9.io import append_jsonl, sha256_path, write_json  # noqa: E402


def _batch_key_from_index_item(item: dict[str, Any]) -> str:
    """Read the batching key, with compatibility for pre-padding indexes.

    Older ``layout_index.json`` files only contain ``layout_key``.  Their
    first two tuple entries are the target and source latent geometries, so a
    geometry-only fallback still lets the new collator pad prompt sequences.
    The collator performs the exact image-mask alignment check after loading
    each batch.
    """

    batch_key = item.get("batch_layout_key")
    if isinstance(batch_key, str) and batch_key:
        return batch_key
    layout = item.get("layout_key")
    if not isinstance(layout, str):
        raise ValueError("cache index item is missing layout_key/batch_layout_key")
    try:
        parsed = ast.literal_eval(layout)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"invalid layout_key in cache index: {layout!r}") from exc
    if not isinstance(parsed, tuple) or len(parsed) < 2:
        raise ValueError(f"invalid layout_key in cache index: {layout!r}")
    return repr((parsed[0], parsed[1]))


def _load_upstream_train(repository: Path) -> Any:
    train_path = repository / "examples/qwen_image_21/model_training/train.py"
    if not train_path.is_file():
        raise FileNotFoundError(train_path)
    sys.path.insert(0, str(repository))
    spec = importlib.util.spec_from_file_location("wgsr_v9_upstream_qwen_image21_train", train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load DiffSynth training module from {train_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--diffsynth-repo", type=Path, required=True)
    bootstrap_args, remaining = bootstrap.parse_known_args()
    upstream = _load_upstream_train(bootstrap_args.diffsynth_repo.resolve())
    from diffsynth.diffusion.runner import (
        initialize_deepspeed_gradient_checkpointing,
        save_training_args,
    )
    parser = upstream.qwen_image_21_parser()
    parser.add_argument("--diffsynth-repo", type=Path, default=bootstrap_args.diffsynth_repo)
    parser.add_argument("--layout-index", type=Path, required=True)
    parser.add_argument("--variant", choices=("simple", "teacher_cot"), required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--expected-global-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--max-optimizer-steps", type=int, default=None)
    parser.add_argument("--lr-scheduler", choices=("legacy_constant", "warmup_stable_cosine"),
                        default="legacy_constant")
    parser.add_argument("--lr-warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lr-cosine-ratio", type=float, default=0.3)
    parser.add_argument("--lr-warmup-start-factor", type=float, default=0.0)
    parser.add_argument("--lr-min-factor", type=float, default=0.1)
    parser.add_argument("--condition-dropout-cache", type=Path, default=None)
    parser.add_argument("--condition-dropout-seed", type=int, default=20261107)
    parser.add_argument("--condition-ablation-cache", type=Path, default=None)
    parser.add_argument("--condition-ablation", choices=("no_metric_stage", "no_image"))
    args = parser.parse_args(remaining)
    if (args.condition_ablation_cache is None) != (args.condition_ablation is None):
        raise ValueError("fixed ablation mode/cache must be specified together")
    if args.condition_ablation is not None and (
        args.condition_dropout_cache is not None
        or args.variant != "teacher_cot" or args.dataset_num_workers != 0
    ):
        raise ValueError("fixed ablations require text Executor, no random dropout, and zero dataset workers")
    lr_schedule = LearningRateSchedule(
        name=args.lr_scheduler, warmup_ratio=args.lr_warmup_ratio,
        cosine_ratio=args.lr_cosine_ratio, warmup_start_factor=args.lr_warmup_start_factor,
        min_lr_factor=args.lr_min_factor,
    )

    if args.dataset_metadata_path is not None or args.dataset_repeat != 1:
        raise ValueError("cached SFT requires no metadata file and dataset_repeat=1")
    if args.batch_size != 2 or args.gradient_accumulation_steps != 1:
        raise ValueError("reviewed recipe requires per-GPU batch 2 and gradient accumulation 1")
    if args.max_optimizer_steps is not None and args.max_optimizer_steps <= 0:
        raise ValueError("--max-optimizer-steps must be positive")
    if args.num_epochs <= 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("num_epochs and learning_rate must be positive and finite")
    if args.condition_dropout_cache is not None:
        if args.variant != "teacher_cot" or args.dataset_num_workers != 0:
            raise ValueError("condition dropout v1 requires teacher_cot and dataset_num_workers=0")

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # Default AcceleratedScheduler advances num_processes times per call.
        # The new schedule is already expressed in global optimizer updates.
        step_scheduler_with_optimizer=not lr_schedule.manual_step,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)
        ],
    )
    expected_global_batch = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    if expected_global_batch != args.expected_global_batch_size or accelerator.num_processes != args.expected_world_size:
        raise ValueError(
            f"reviewed global batch is {args.expected_global_batch_size} on {args.expected_world_size} ranks; "
            f"got batch={args.batch_size}, ranks={accelerator.num_processes}, "
            f"accumulation={args.gradient_accumulation_steps}"
        )
    if accelerator.state.deepspeed_plugin is None:
        raise RuntimeError("training must use the reviewed DeepSpeed ZeRO-3 Accelerate config")
    ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
    if lr_schedule.manual_step and "scheduler" in ds_config:
        raise ValueError("executor LR schedule cannot be combined with a DeepSpeed-config scheduler")
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_batch_size"] = expected_global_batch
    if ds_config.get("zero_optimization", {}).get("stage") != 3:
        raise RuntimeError("training config must use DeepSpeed ZeRO-3")
    if ds_config.get("zero_optimization", {}).get("offload_optimizer", {}).get("device", "none") != "none":
        raise RuntimeError("CPU optimizer offload is outside the reviewed recipe")
    if ds_config.get("zero_optimization", {}).get("offload_param", {}).get("device", "none") != "none":
        raise RuntimeError("CPU parameter offload is outside the reviewed recipe")

    set_seed(args.seed, device_specific=True)
    cache_root = Path(args.dataset_base_path).resolve()
    layout_path = Path(args.layout_index).resolve()
    index = json.loads(layout_path.read_text(encoding="utf-8"))
    if index.get("status") != "complete" or not index.get("all_condition_and_target_tensors_finite"):
        raise ValueError("cache index is incomplete or reports nonfinite training tensors")
    dataset = upstream.UnifiedDataset(base_path=str(cache_root), repeat=1)
    if len(dataset) != index["count"]:
        raise ValueError(f"cache Dataset count {len(dataset)} does not match index {index['count']}")
    cached_paths = [str(Path(path).resolve()) for path in dataset.cached_data]
    indexed_paths = [item["output_path"] for item in index["items"]]
    if set(cached_paths) != set(indexed_paths):
        raise ValueError("DiffSynth cache files do not match the validated cache index")
    key_by_path = {item["output_path"]: _batch_key_from_index_item(item) for item in index["items"]}
    batch_sampler = LayoutBatchSampler(
        [key_by_path[path] for path in cached_paths],
        args.batch_size,
        accelerator.num_processes,
        seed=args.seed,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    microbatches_per_rank = len(batch_sampler) // accelerator.num_processes
    updates_per_epoch = microbatches_per_rank // args.gradient_accumulation_steps
    total_optimizer_steps = updates_per_epoch * args.num_epochs
    if args.max_optimizer_steps is not None:
        total_optimizer_steps = min(total_optimizer_steps, args.max_optimizer_steps)
    lr_schedule_plan = lr_schedule.resolve(total_optimizer_steps)
    dropout_manifest = None
    if args.condition_dropout_cache is not None:
        from wgsr_v9.executor_dropout import DropoutBatchSampler, DropoutCachedDataset, action_summary, stream_seed
        dataset = DropoutCachedDataset(dataset, index, args.condition_dropout_cache,
                                      seed=args.condition_dropout_seed, epochs=args.num_epochs)
        dropout_manifest = dataset.manifest
        if dropout_manifest["training_seed"] != args.seed or dropout_manifest["world_size"] != accelerator.num_processes:
            raise ValueError("dropout cache was planned with a different training seed/world size")
        if dropout_manifest["input_sha256"].get(str(layout_path)) != sha256_path(layout_path):
            raise ValueError("dropout layout index changed")
        batch_sampler = DropoutBatchSampler(batch_sampler, seed=args.condition_dropout_seed)
    ablation_manifest = None
    if args.condition_ablation is not None:
        from wgsr_v9.executor_ablation import AblationBatchSampler, AblationCachedDataset
        dataset = AblationCachedDataset(dataset, index, args.condition_ablation_cache, args.condition_ablation)
        ablation_manifest = dataset.manifest
        if ablation_manifest["input_sha256"].get(str(layout_path)) != sha256_path(layout_path):
            raise ValueError("fixed ablation layout index changed")
        batch_sampler = AblationBatchSampler(batch_sampler, args.condition_ablation)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_cached,
        num_workers=args.dataset_num_workers,
    )
    model = upstream.QwenImage21TrainingModule(
        model_paths=args.model_paths,
        processor_path=args.processor_path,
        trainable_models="dit",
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        task="sft:train",
        device="cpu",
    )
    optimizer = torch.optim.AdamW(
        model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    peak_learning_rates = [group["lr"] for group in optimizer.param_groups]
    scheduler = lr_schedule.build(optimizer, total_optimizer_steps)
    lr_schedule_plan["peak_learning_rates"] = peak_learning_rates
    lr_schedule_plan["initial_learning_rates"] = scheduler.get_last_lr()
    output = Path(args.output_path).resolve()
    if accelerator.is_main_process:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"checkpoint output directory is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    logger = upstream.ModelLogger(output, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt, enable_csv_log=True)
    if accelerator.is_main_process:
        save_training_args(args)
        global_microbatches = microbatches_per_rank * accelerator.num_processes
        sample_visits_per_epoch = global_microbatches * args.batch_size
        batch_plan = {
            "algorithm_version": ALGORITHM_VERSION,
            "executor_variant": args.variant,
            "input_path": str(layout_path),
            "output_path": str(output),
            "original_unique_samples": len(dataset),
            "sample_visits_per_epoch": sample_visits_per_epoch,
            "sample_visits_per_rank_per_epoch": microbatches_per_rank * args.batch_size,
            "padding_repeated_visits_per_epoch": sample_visits_per_epoch - len(dataset),
            "microbatches_per_rank_per_epoch": microbatches_per_rank,
            "optimizer_updates_per_epoch": (
                microbatches_per_rank // args.gradient_accumulation_steps
            ),
            "per_gpu_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "global_batch_size": expected_global_batch,
            "epochs": args.num_epochs,
            "lr_scheduler": lr_schedule_plan,
            "seed": args.seed,
            "image_dropout": 0.1 if dropout_manifest is not None else 0.0,
            "condition_dropout": None if dropout_manifest is None else {
                "plan_sha256": dropout_manifest["plan_sha256"], "policy": dropout_manifest["policy"],
                "action_schedule": action_summary(updates_per_epoch, args.condition_dropout_seed, args.num_epochs)},
            "validation_during_training": False,
            "condition_ablation": None if ablation_manifest is None else {
                "algorithm_version": ablation_manifest["algorithm_version"],
                "mode": args.condition_ablation,
                "manifest_sha256": sha256_path(args.condition_ablation_cache / "manifest.json"),
                "plan_sha256": ablation_manifest["plan_sha256"],
                "flow_rng": "legacy formal text recipe; no change to RNG consumption in dataset/sampler"},
            "training_implementation": (
                f"upstream Qwen Image 2.1 module, DiT, FlowMatchSFTLoss, AdamW; LR={lr_schedule.name}; "
                "homogeneous image-layout batches with tail-padded prompt embeddings and attention masks"
            ),
            "batch_layout_groups": len(batch_sampler.groups),
            "prompt_sequence_padding": "tail_padding_with_prompt_embeds_mask",
            "status": "ready",
            "error": None,
            "retry_count": 0,
        }
        write_json(output / "batch_plan.json", batch_plan)
        print("BATCH_PLAN", json.dumps(batch_plan), flush=True)

    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    if lr_schedule.manual_step and (
        getattr(scheduler, "step_with_optimizer", True)
        or getattr(model, "lr_scheduler", None) is not None
    ):
        raise RuntimeError("executor LR scheduler must be stepped manually once per successful update")
    initialize_deepspeed_gradient_checkpointing(accelerator)
    torch.cuda.reset_peak_memory_stats(accelerator.device)
    global_step = 0
    successful_optimizer_steps = 0
    accumulated_losses: list[float] = []
    accumulated_shapes: list[list[int]] = []
    metrics_file = (output / "optimizer_metrics.jsonl").open("a", encoding="utf-8") if accelerator.is_main_process else None
    started = time.monotonic()
    try:
        for epoch in range(args.num_epochs):
            batch_sampler.set_epoch(epoch)
            if hasattr(dataloader, "set_epoch"):
                dataloader.set_epoch(epoch)
            for microstep, inputs in enumerate(tqdm(dataloader, disable=not accelerator.is_main_process), 1):
                with accelerator.accumulate(model):
                    dropout = inputs[0].pop("condition_dropout", None)
                    ablation = inputs[0].pop("condition_ablation", None)
                    if ablation_manifest is not None:
                        if ablation is None or (ablation["epoch"], ablation["step"], ablation["action"]) != (
                            epoch, microstep - 1, args.condition_ablation
                        ):
                            raise ValueError("fixed ablation batch lost alignment")
                        append_jsonl(output / f"ablation-rank-{accelerator.process_index}.jsonl", {
                            "algorithm_version": ablation_manifest["algorithm_version"],
                            "input_path": str(args.condition_ablation_cache), "output_path": str(output),
                            "rank": accelerator.process_index, **ablation,
                            "status": "ready", "error": None, "retry_count": 0})
                    if dropout_manifest is not None:
                        if dropout is None or dropout["epoch"] != epoch or dropout["step"] != microstep - 1:
                            raise ValueError("dropout action tape lost alignment after distributed sharding")
                        append_jsonl(output / f"dropout-rank-{accelerator.process_index}.jsonl", {
                            "algorithm_version": dropout_manifest["algorithm_version"],
                            "input_path": str(args.condition_dropout_cache), "output_path": str(output),
                            "rank": accelerator.process_index, **dropout,
                            "status": "ready", "error": None, "retry_count": 0})
                    actual_batch = int(inputs[0]["input_latents"].shape[0])
                    if actual_batch != args.batch_size:
                        raise ValueError(f"expected tensor batch {args.batch_size}, got {actual_batch}")
                    accumulated_shapes.append(list(inputs[0]["input_latents"].shape))
                    deepspeed_steps_before = model.global_steps
                    learning_rates_used = [group["lr"] for group in optimizer.param_groups]
                    # Preserve the dropout experiment's isolated flow-noise/time random stream.
                    rng_context = (torch.random.fork_rng(devices=[accelerator.device.index])
                                   if dropout_manifest is not None else nullcontext())
                    with rng_context:
                        if dropout_manifest is not None:
                            torch.manual_seed(stream_seed(args.seed, "flow", epoch, microstep, accelerator.process_index))
                        loss = model({}, inputs=inputs)
                    global_loss = accelerator.reduce(loss.detach().float(), reduction="mean")
                    if not bool(torch.isfinite(global_loss)):
                        raise FloatingPointError(f"nonfinite loss at epoch={epoch + 1}, microstep={microstep}")
                    accumulated_losses.append(float(global_loss))
                    accelerator.backward(loss)
                    optimizer.step()
                    step_learning_rate(
                        scheduler, manual_step=lr_schedule.manual_step,
                        sync_gradients=accelerator.sync_gradients,
                        optimizer_step_was_skipped=optimizer.step_was_skipped,
                    )
                    optimizer.zero_grad()
                    if model.global_steps != deepspeed_steps_before + int(accelerator.sync_gradients):
                        raise RuntimeError("DeepSpeed optimizer step did not match the accumulation boundary")
                    logger.on_step_end(accelerator, model, loss=loss)
                    if accelerator.sync_gradients:
                        global_step += 1
                        successful_optimizer_steps += int(not optimizer.step_was_skipped)
                        grad_norm = accelerator.deepspeed_engine_wrapped.get_global_grad_norm()
                        if grad_norm is not None and not math.isfinite(float(grad_norm)):
                            raise FloatingPointError(f"nonfinite gradient norm at optimizer step {global_step}")
                        if accelerator.is_main_process:
                            record = {
                                "algorithm_version": ALGORITHM_VERSION,
                                "executor_variant": args.variant,
                                "input_path": str(layout_path),
                                "output_path": str(output / "optimizer_metrics.jsonl"),
                                "optimizer_step": global_step,
                                "successful_optimizer_steps": successful_optimizer_steps,
                                "optimizer_step_was_skipped": optimizer.step_was_skipped,
                                "epoch": epoch + 1,
                                "microstep_in_epoch": microstep,
                                "loss_global_mean": sum(accumulated_losses) / len(accumulated_losses),
                                "gradient_norm": None if grad_norm is None else float(grad_norm),
                                "learning_rate": optimizer.param_groups[0]["lr"],
                                "learning_rate_used": learning_rates_used[0],
                                "learning_rates_used": learning_rates_used,
                                "learning_rates_next": [group["lr"] for group in optimizer.param_groups],
                                "condition_dropout": dropout,
                                "condition_ablation": ablation,
                                "numeric_diagnostics": {},  # Preserve the text-run log schema.
                                "per_gpu_batch_size": args.batch_size,
                                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                                "global_batch_size": expected_global_batch,
                                "elapsed_seconds": time.monotonic() - started,
                                "deepspeed_global_steps": model.global_steps,
                                "input_latent_shapes": accumulated_shapes,
                                "peak_allocated_bytes_rank0": torch.cuda.max_memory_allocated(accelerator.device),
                                "peak_reserved_bytes_rank0": torch.cuda.max_memory_reserved(accelerator.device),
                                "status": "ready",
                                "error": None,
                                "retry_count": 0,
                            }
                            metrics_file.write(json.dumps(record) + "\n")
                            metrics_file.flush()
                            print("OPTIMIZER_STEP", json.dumps(record), flush=True)
                        accumulated_losses = []
                        accumulated_shapes = []
                if args.max_optimizer_steps is not None and global_step >= args.max_optimizer_steps:
                    break
            logger.on_epoch_end(accelerator, model, epoch)
            if args.max_optimizer_steps is not None and global_step >= args.max_optimizer_steps:
                break
        logger.on_training_end(accelerator, model)
        if accelerator.is_main_process:
            write_json(
                output / "completed.json",
                {
                    "algorithm_version": ALGORITHM_VERSION,
                    "executor_variant": args.variant,
                    "input_path": str(layout_path),
                    "output_path": str(output),
                    "optimizer_steps": global_step,
                    "successful_optimizer_steps": successful_optimizer_steps,
                    "lr_scheduler": lr_schedule_plan,
                    "final_learning_rates": [group["lr"] for group in optimizer.param_groups],
                    "seconds": time.monotonic() - started,
                    "status": "complete",
                    "error": None,
                    "retry_count": 0,
                },
            )
    finally:
        if metrics_file is not None:
            metrics_file.close()
    accelerator.wait_for_everyone()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
