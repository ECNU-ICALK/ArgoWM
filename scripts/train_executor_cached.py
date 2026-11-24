#!/usr/bin/env python3
"""Thin CLI entry point for :mod:`wgsr_v9.executor_training_runner`."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wgsr_v9.executor_training_runner import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
