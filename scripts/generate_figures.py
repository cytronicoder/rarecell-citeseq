#!/usr/bin/env python
"""Generate essential figures from benchmark results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from rarecell.config import FIGURES_DIR, TABLES_DIR
from rarecell.plotting import save_all_standard_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=None, help="Benchmark results CSV. Defaults to newest canonical table.")
    parser.add_argument("--output-prefix", default=None, help="Prefix for generated figure names.")
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR), help="Directory for figures.")
    parser.add_argument("--tables-dir", default=str(TABLES_DIR), help="Directory for figure index CSV.")
    return parser.parse_args()


def _latest_results_table() -> Path:
    matches = sorted(
        TABLES_DIR.glob("*__benchmark_results.csv"),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError("No benchmark results table found. Run `python scripts/run_benchmark.py` first.")
    return matches[0]


def _prefix_from_results(path: Path) -> str:
    suffix = "__benchmark_results.csv"
    if not path.name.endswith(suffix):
        raise ValueError(f"Cannot infer output prefix from {path}; pass --output-prefix.")
    return path.name[: -len(suffix)]


def main() -> None:
    args = parse_args()
    results_path = Path(args.results) if args.results else _latest_results_table()
    results_df = pd.read_csv(results_path)
    output_prefix = args.output_prefix or _prefix_from_results(results_path)
    paths = save_all_standard_plots(
        results_df,
        output_prefix=output_prefix,
        figures_dir=args.figures_dir,
        tables_dir=args.tables_dir,
    )
    if not paths:
        raise RuntimeError("No figures were generated.")
    for path in paths:
        print(path)
    print(f"Figure index: {Path(args.tables_dir) / f'{output_prefix}__figure_index.csv'}")


if __name__ == "__main__":
    main()
