#!/usr/bin/env python3
"""CLI entry point for resumable Qwen Image 2.1 executor inference."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wgsr_v9.executor_inference import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
