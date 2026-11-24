"""Command-line parsing for resumable executor inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
    default_checkpoint,
    default_input_manifest,
    default_output_root,
    load_config,
    resolve_path,
)
from .runner import supervisor_main, worker_main


def parse_args() -> argparse.Namespace:
    # Parse the config path first so versioned experiments can keep their
    # inference roots and checkpoints isolated without replacing the default
    # v1 recipe.
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=None)
    bootstrap_args, _ = bootstrap.parse_known_args()
    config_path = resolve_path(bootstrap_args.config) if bootstrap_args.config else None
    config = load_config(config_path) if config_path is not None else load_config()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=config_path)
    parser.add_argument(
        "--worker", action="store_true", help="run one worker; normally the command is a supervisor"
    )
    parser.add_argument("--variant", choices=("simple", "teacher_cot", "predicted_cot"), required=True)
    parser.add_argument("--input-manifest", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--pretrained", action="store_true", default=config.get("pretrained", False),
                        help="use original model transformer weights without any fine-tuned checkpoint")
    parser.add_argument(
        "--model-path", type=Path, default=resolve_path(config["model_path"])
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--seed-manifest",
        type=Path,
        default=resolve_path(config["baseline_seed_manifest"]),
    )
    parser.add_argument("--seed-base", type=int, default=20261030)
    parser.add_argument("--steps", type=int, default=int(config["num_inference_steps"]))
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default=str(config["dtype"]))
    kv_cache = parser.add_mutually_exclusive_group()
    kv_cache.add_argument("--use-kv-cache", dest="use_kv_cache", action="store_true")
    kv_cache.add_argument("--no-use-kv-cache", dest="use_kv_cache", action="store_false")
    parser.set_defaults(use_kv_cache=bool(config["use_kv_cache"]))
    parser.add_argument("--world-size", type=int, default=int(config["world_size"]))
    parser.add_argument("--gpu-ids", default=str(config["physical_gpu_ids"]))
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--row-indices",
        default=None,
        help="comma-separated input row indices, useful for case-study smoke runs",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--condition-dropout", choices=("keep", "drop_text", "drop_image"),
                        default=config.get("condition_dropout", "keep"),
                        help="fixed diagnostic action; formal inference uses keep")
    parser.add_argument("--condition-ablation", choices=("none", "no_metric_stage", "no_image"),
                        default=config.get("condition_ablation", "none"))
    args = parser.parse_args()
    args.executor_prompt_protocol_version = config.get("executor_prompt_protocol_version")
    args.evaluation_manifest = resolve_path(config["evaluation_manifest"]) if config.get("evaluation_manifest") else None
    args.expected_rows = config.get("expected_rows")
    if args.pretrained and args.checkpoint is not None:
        raise ValueError("--pretrained cannot be combined with a checkpoint")
    if args.pretrained and (config.get("checkpoints") or config.get("executor_training_root")):
        raise ValueError("pretrained config must not contain fine-tuned checkpoints or a training root")
    if args.condition_ablation != "none" and (
        args.condition_dropout != "keep" or args.variant == "simple"
    ):
        raise ValueError("fixed ablation requires CoT text Executor without diagnostic dropout")
    args.input_manifest = resolve_path(
        args.input_manifest or default_input_manifest(config, args.variant)
    )
    args.checkpoint = None if args.pretrained else resolve_path(
        args.checkpoint or default_checkpoint(config, args.variant)
    )
    args.model_path = resolve_path(args.model_path)
    args.output_root = resolve_path(args.output_root or default_output_root(config, args.variant))
    if args.seed_manifest is not None:
        args.seed_manifest = resolve_path(args.seed_manifest)
    if args.world_size < 1:
        raise ValueError("--world-size must be positive")
    if args.worker and not (0 <= args.rank < args.world_size):
        raise ValueError("worker rank is outside world size")
    return args


def main() -> int:
    args = parse_args()
    return worker_main(args) if args.worker else supervisor_main(args)
