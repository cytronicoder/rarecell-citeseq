#!/usr/bin/env python
"""Compatibility wrapper for make_baseline_representations.py."""

from __future__ import annotations

import runpy
from pathlib import Path

SCRIPT = Path(__file__).resolve().with_name("make_baseline_representations.py")

if __name__ == "__main__":
    runpy.run_path(str(SCRIPT), run_name="__main__")
