"""Versioned constants for Qwen Image 2.1 executor inference."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs/local/rice/executor_infer.json"
CLI_SCRIPT = REPO_ROOT / "scripts/run_executor_inference.py"
ALGORITHM_VERSION = "wgsr_v9_qwen_image21_executor_inference_v1"
PROTOCOL_VERSION = "wgsr_v9_qwen_image21_executor_inference_v1"
EXPECTED_SIZES = {(672, 896), (896, 672)}
