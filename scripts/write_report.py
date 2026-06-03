#!/usr/bin/env python
"""Write a concise Markdown report from benchmark outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from rarecell.config import LOGS_DIR, REPORTS_DIR, TABLES_DIR
from rarecell.reporting import write_benchmark_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-summary", default=None, help="Run summary JSON. Defaults to newest canonical summary.")
    parser.add_argument("--metric-summary", default=None,
                        help="Metric summary CSV. Inferred from output prefix if omitted.")
    parser.add_argument("--figure-index", default=None, help="Optional figure index CSV.")
    parser.add_argument("--output", default=None, help="Output Markdown path.")
    return parser.parse_args()


def _latest(pattern: str, directory: Path) -> Path:
    matches = sorted(directory.glob(pattern), key=lambda path: (path.stat().st_mtime, path.name), reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}.")
    return matches[0]


def main() -> None:
    args = parse_args()
    run_summary_path = Path(args.run_summary) if args.run_summary else _latest("*__run_summary.json", LOGS_DIR)
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    output_prefix = run_summary["output_prefix"]

    metric_summary_path = (
        Path(args.metric_summary)
        if args.metric_summary
        else TABLES_DIR / f"{output_prefix}__metric_summary.csv"
    )
    metric_summary = pd.read_csv(metric_summary_path)

    figure_index_path = (
        Path(args.figure_index)
        if args.figure_index
        else TABLES_DIR / f"{output_prefix}__figure_index.csv"
    )
    figure_index = pd.read_csv(figure_index_path) if figure_index_path.exists() else None

    output_path = (
        Path(args.output)
        if args.output
        else REPORTS_DIR / f"{output_prefix}__benchmark_report.md"
    )
    report_path = write_benchmark_report(output_path, run_summary, metric_summary, figure_index)
    print(report_path)


if __name__ == "__main__":
    main()
