#!/usr/bin/env python
"""Generate all benchmark summary tables from the raw results CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from rarecell.benchmark import make_benchmark_summary, make_best_method_summary, make_abundant_control_summary

METRICS_DIR = ROOT / "results" / "metrics"
TABLES_DIR = ROOT / "results" / "tables"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw-csv",
        default=str(METRICS_DIR / "rare_cell_benchmark_raw.csv"),
        help="Path to the raw benchmark results CSV.",
    )
    p.add_argument(
        "--control-csv",
        default=str(METRICS_DIR / "abundant_cell_control_raw.csv"),
        help="Path to the abundant-cell control raw CSV (falls back to control_abundant_cell_downsampling.csv).",
    )
    p.add_argument(
        "--out-dir",
        default=str(TABLES_DIR),
        help="Output directory for summary tables.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_csv = Path(args.raw_csv)
    if not raw_csv.exists():
        print(f"ERROR: Raw benchmark CSV not found: {raw_csv}", file=sys.stderr)
        print("Run scripts/run_benchmark.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading raw benchmark results from {raw_csv}")
    raw_df = pd.read_csv(raw_csv)

    print("Generating benchmark_summary.csv …")
    summary = make_benchmark_summary(raw_df)
    summary_path = out / "benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    print("Generating best_method_by_fraction.csv …")
    best = make_best_method_summary(summary)
    best_path = out / "best_method_by_fraction.csv"
    best.to_csv(best_path, index=False)
    print(f"  Saved: {best_path}")

    control_csv = Path(args.control_csv)
    if not control_csv.exists():
        fallback = METRICS_DIR / "control_abundant_cell_downsampling.csv"
        if fallback.exists():
            print(f"  Abundant-cell control CSV not found at {control_csv}; using fallback {fallback}")
            control_csv = fallback
        else:
            print(
                f"  Abundant-cell control CSV not found at {control_csv}; "
                "skipping abundant_cell_control_summary.csv."
            )
            control_csv = None

    if control_csv is not None:
        print(f"Generating abundant_cell_control_summary.csv from {control_csv} …")
        control_df = pd.read_csv(control_csv)
        control_summary = make_abundant_control_summary(control_df)
        control_summary_path = out / "abundant_cell_control_summary.csv"
        control_summary.to_csv(control_summary_path, index=False)
        print(f"  Saved: {control_summary_path}")

    print("Summary table generation complete.")


if __name__ == "__main__":
    main()
