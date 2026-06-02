"""Benchmark report writing utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    formatted = df.copy()
    for col in formatted.columns:
        if is_numeric_dtype(formatted[col]):
            formatted[col] = formatted[col].map(
                lambda v: "" if pd.isna(v) else f"{float(v):.4g}"
            )
        else:
            formatted[col] = formatted[col].astype(str)
    header = "| " + " | ".join(map(str, formatted.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    body = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in formatted.to_numpy()
    ]
    return "\n".join([header, sep, *body])


def write_benchmark_report(
        output_path: str | Path,
        run_summary: dict,
        metric_summary_df: pd.DataFrame,
        figure_index_df: pd.DataFrame | None = None,
) -> Path:
    """Write a Markdown benchmark report.

    Include: input file, label key, target label, representations,
    retained fractions, seeds, n_neighbors, dataset summary, output files,
    main metric summary table, and figure index if available.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    preferred_columns = [
        "representation",
        "retain_fraction",
        "f1_mean",
        "recall_mean",
        "neighborhood_purity_mean",
    ]
    available = [c for c in preferred_columns if c in metric_summary_df.columns]
    summary_table = _markdown_table(metric_summary_df[available]) if available else ""

    output_files = run_summary.get("output_files", [])
    figures = [p for p in output_files if "/figures/" in str(p)]
    logs = [p for p in output_files if "/logs/" in str(p)]

    output_prefix = run_summary.get("output_prefix", "")
    results_csv = f"results/tables/{output_prefix}__benchmark_results.csv"
    results_parquet = f"results/metrics/{output_prefix}__benchmark_results.parquet"

    sections = [
        "# Rare-cell downsampling benchmark report",
        "",
        "## Input",
        f"- Input file: {run_summary.get('input_file', '')}",
        f"- Label key: {run_summary.get('label_key', '')}",
        f"- Target label: {run_summary.get('target_label', '')}",
        f"- Representations: {', '.join(map(str, run_summary.get('representations', [])))}",
        f"- Retain fractions: {', '.join(map(str, run_summary.get('retain_fractions', [])))}",
        f"- Seeds: {', '.join(map(str, run_summary.get('seeds', [])))}",
        f"- n_neighbors: {run_summary.get('n_neighbors', '')}",
        "",
        "## Dataset summary",
        f"- Total cells: {run_summary.get('n_cells', '')}",
        f"- Original target cells: {run_summary.get('target_cell_count_original', '')}",
        f"- Number of cell types: {run_summary.get('n_cell_types', '')}",
        "",
        "## Output files",
        f"- Results CSV: {results_csv}",
        f"- Results parquet: {results_parquet}",
        f"- Figures: {', '.join(figures) if figures else 'None'}",
        f"- Logs: {', '.join(logs) if logs else 'None'}",
        "",
        "## Main metric summary",
        summary_table,
    ]

    if figure_index_df is not None and not figure_index_df.empty:
        sections += ["", "## Figure index", _markdown_table(figure_index_df)]

    sections.append("")
    out.write_text("\n".join(sections), encoding="utf-8")
    return out
