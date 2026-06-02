"""Benchmark runner, result validation, summaries, and reports."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from rarecell.config import REPRESENTATION_KEY_MAP
from rarecell.downsampling import downsample_target_cells, validate_target_label
from rarecell.metrics import compute_all_metrics
from rarecell.preprocessing import preprocess_protein, preprocess_rna
from rarecell.representations import build_joint_representation
from rarecell.script_utils import resolve_representations, standard_representation_name

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
LOGS_DIR = RESULTS_DIR / "logs"
INTERMEDIATE_DIR = RESULTS_DIR / "intermediate"
REPORTS_DIR = RESULTS_DIR / "reports"

RESULT_COLUMNS = [
    "dataset",
    "target_cell_type",
    "target_label",
    "label_key",
    "fraction",
    "retain_fraction",
    "seed",
    "representation",
    "n_neighbors",
    "n_cells_total",
    "n_target_original",
    "n_cells",
    "n_target_remaining",
    "n_target",
    "n_other",
    "target_fraction",
    "precision",
    "recall",
    "f1",
    "neighborhood_purity",
    "silhouette_target",
    "target_silhouette",
]

METRIC_COLUMNS = [
    "precision",
    "recall",
    "f1",
    "neighborhood_purity",
    "target_silhouette",
    "n_target",
]


def ensure_project_directories() -> None:
    """Create the project directories used by the benchmark pipeline."""
    for path in [
        PROJECT_ROOT / "src",
        RESULTS_DIR,
        TABLES_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        LOGS_DIR,
        INTERMEDIATE_DIR,
        REPORTS_DIR,
        PROJECT_ROOT / "notebooks",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def validate_representations(adata: Any, representations: list[str]) -> None:
    """Confirm every requested public representation resolves to an AnnData obsm key."""
    try:
        resolve_representations(adata, representations)
    except KeyError as exc:
        available = list(getattr(adata, "obsm", {}).keys())
        raise ValueError(
            "Missing required adata.obsm representations: "
            f"{list(representations)}. Available representations: {available or '<none>'}."
        ) from exc


def rebuild_baseline_representations(
        adata: Any,
        rna_pcs: int = 50,
        protein_pcs: int = 20,
        n_top_genes: int = 3000,
) -> Any:
    """Recompute RNA, protein, and simple joint representations for one AnnData."""
    if "protein_counts" not in getattr(adata, "obsm", {}):
        raise KeyError(
            "Expected adata.obsm['protein_counts'] to rebuild benchmark representations. "
            "Import CITE-seq data into the internal AnnData layout first."
        )
    rebuilt = preprocess_rna(adata, n_top_genes=n_top_genes, n_pcs=rna_pcs)
    rebuilt = preprocess_protein(rebuilt, n_components=protein_pcs)
    rebuilt = build_joint_representation(rebuilt)
    return rebuilt


def run_downsampling_benchmark(
        adata,
        label_key: str,
        target_label: str,
        representation_keys: list[str] | None,
        retain_fractions: list[float],
        seeds: list[int],
        n_neighbors: int = 15,
        dataset: str = "unknown",
        output_path: str | Path | None = None,
        rna_pcs: int = 50,
        protein_pcs: int = 20,
        n_top_genes: int = 3000,
) -> pd.DataFrame:
    """Run the full benchmark, rebuilding representations after downsampling."""
    validate_target_label(adata, label_key, target_label)
    requested_representations = representation_keys or list(REPRESENTATION_KEY_MAP)
    original_labels = adata.obs[label_key].astype(str)
    n_target_original = int((original_labels == str(target_label)).sum())

    rows = []
    for retain_fraction in retain_fractions:
        for seed in seeds:
            adata_sub = downsample_target_cells(
                adata,
                label_key=label_key,
                target_label=target_label,
                retain_fraction=float(retain_fraction),
                seed=int(seed),
            )
            rebuilt = rebuild_baseline_representations(
                adata_sub,
                rna_pcs=int(rna_pcs),
                protein_pcs=int(protein_pcs),
                n_top_genes=int(n_top_genes),
            )
            representation_pairs = resolve_representations(rebuilt, requested_representations)
            labels = rebuilt.obs[label_key].astype(str).to_numpy()
            for representation_name, representation_key in representation_pairs:
                embedding = rebuilt.obsm[representation_key]
                metrics = compute_all_metrics(
                    embedding,
                    labels,
                    target_label,
                    seed=int(seed),
                    n_neighbors=int(n_neighbors),
                )
                row = {
                    "dataset": str(dataset),
                    "target_cell_type": str(target_label),
                    "target_label": str(target_label),
                    "label_key": str(label_key),
                    "fraction": float(retain_fraction),
                    "retain_fraction": float(retain_fraction),
                    "seed": int(seed),
                    "representation": str(representation_name),
                    "n_neighbors": int(n_neighbors),
                    "n_cells_total": int(rebuilt.n_obs),
                    "n_target_original": int(n_target_original),
                }
                row.update(metrics)
                row["n_cells_total"] = row.get("n_cells", row["n_cells_total"])
                row["n_target_remaining"] = row.get("n_target")
                row["silhouette_target"] = row.get("silhouette_target", row.get("target_silhouette"))
                rows.append(row)

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out, index=False)
    return results


def save_benchmark_results(results_df: pd.DataFrame, output_prefix: str) -> None:
    """Save benchmark results to CSV and optionally parquet."""
    ensure_project_directories()
    csv_path = TABLES_DIR / f"{output_prefix}__benchmark_results.csv"
    parquet_path = METRICS_DIR / f"{output_prefix}__benchmark_results.parquet"

    results_df.to_csv(csv_path, index=False)
    try:
        import pyarrow  # noqa: F401

        results_df.to_parquet(parquet_path, index=False)
    except Exception as exc:
        logging.warning(
            "Parquet output was skipped because optional parquet support is unavailable or failed. "
            "CSV output was saved successfully at: %s "
            "To enable parquet output, install pyarrow. Original error: %s",
            csv_path,
            exc,
        )


def validate_results_table(
        results_df: pd.DataFrame,
        representation_keys: list[str],
        retain_fractions: list[float],
        seeds: list[int],
) -> None:
    """Validate that benchmark results contain expected rows and fields."""
    assert not results_df.empty, "Benchmark results table is empty."

    missing_columns = [column for column in RESULT_COLUMNS if column not in results_df.columns]
    assert not missing_columns, f"Results table is missing columns: {missing_columns}."

    observed_representations = set(results_df["representation"].astype(str))
    expected_representations = set(standard_representation_name(key) for key in representation_keys)
    assert expected_representations.issubset(observed_representations), (
        f"Missing representation rows. Expected {sorted(expected_representations)}, "
        f"observed {sorted(observed_representations)}."
    )

    observed_fractions = set(results_df["retain_fraction"].astype(float))
    expected_fractions = set(map(float, retain_fractions))
    assert expected_fractions.issubset(observed_fractions), (
        f"Missing retain_fraction rows. Expected {sorted(expected_fractions)}, "
        f"observed {sorted(observed_fractions)}."
    )

    observed_seeds = set(results_df["seed"].astype(int))
    expected_seeds = set(map(int, seeds))
    assert expected_seeds.issubset(observed_seeds), (
        f"Missing seed rows. Expected {sorted(expected_seeds)}, "
        f"observed {sorted(observed_seeds)}."
    )

    numeric_columns = [
        "fraction", "retain_fraction", "seed", "n_neighbors", "n_cells_total", "n_target_original",
        "n_cells", "n_target_remaining",
        "n_target", "n_other", "target_fraction", "precision", "recall", "f1",
        "neighborhood_purity", "silhouette_target", "target_silhouette",
    ]
    non_numeric = [
        column
        for column in numeric_columns
        if column in results_df and not is_numeric_dtype(results_df[column])
    ]
    assert not non_numeric, (
        f"Expected numeric metric/count columns, found non-numeric columns: {non_numeric}."
    )

    ordered_counts = (
        results_df.groupby(["representation", "seed", "retain_fraction"], dropna=False)["n_target"]
        .max()
        .reset_index()
    )
    for (representation, seed), group in ordered_counts.groupby(["representation", "seed"], dropna=False):
        group = group.sort_values("retain_fraction")
        counts = group["n_target"].to_numpy(dtype=float)
        if counts.size > 1:
            assert bool(np.all(np.diff(counts) >= 0)), (
                "n_target should not increase as retain_fraction decreases "
                f"for representation={representation}, seed={seed}."
            )


def cell_counts_by_fraction(
        adata,
        label_key: str,
        target_label: str,
        retain_fractions: list[float],
        seeds: list[int],
) -> pd.DataFrame:
    """Return cell-type counts for every downsampling condition."""
    validate_target_label(adata, label_key, target_label)
    rows = []
    for retain_fraction in retain_fractions:
        for seed in seeds:
            adata_sub = downsample_target_cells(
                adata,
                label_key=label_key,
                target_label=target_label,
                retain_fraction=float(retain_fraction),
                seed=int(seed),
            )
            counts = adata_sub.obs[label_key].astype(str).value_counts(dropna=False)
            for cell_type, n_cells in counts.items():
                rows.append(
                    {
                        "retain_fraction": float(retain_fraction),
                        "seed": int(seed),
                        "cell_type": str(cell_type),
                        "n_cells": int(n_cells),
                        "target_label": str(target_label),
                    }
                )
    return pd.DataFrame(rows)


def make_metric_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize benchmark metrics by representation and retained fraction."""
    summary = (
        results_df.groupby(["representation", "retain_fraction"], dropna=False)[METRIC_COLUMNS]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(map(str, column)).rstrip("_") if isinstance(column, tuple) else str(column)
        for column in summary.columns
    ]
    return summary


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    formatted = df.copy()
    for column in formatted.columns:
        if is_numeric_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4g}"
            )
        else:
            formatted[column] = formatted[column].astype(str)
    header = "| " + " | ".join(map(str, formatted.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in formatted.to_numpy()
    ]
    return "\n".join([header, separator, *body])


def write_markdown_report(
        output_prefix: str,
        input_file: str,
        label_key: str,
        target_label: str,
        representation_keys: list[str],
        retain_fractions: list[float],
        seeds: list[int],
        n_neighbors: int,
        n_cells: int,
        original_target_cells: int,
        n_cell_types: int,
        output_files: list[str],
        metric_summary: pd.DataFrame,
) -> Path:
    """Write the automatic Markdown benchmark report to results/reports/."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{output_prefix}__benchmark_report.md"

    preferred_columns = [
        "representation",
        "retain_fraction",
        "f1_mean",
        "recall_mean",
        "neighborhood_purity_mean",
    ]
    available_columns = [column for column in preferred_columns if column in metric_summary.columns]
    table = _markdown_table(metric_summary[available_columns]) if available_columns else ""

    figures = [path for path in output_files if "/figures/" in path or path.startswith("results/figures/")]
    logs = [path for path in output_files if "/logs/" in path or path.startswith("results/logs/")]
    results_csv = f"results/tables/{output_prefix}__benchmark_results.csv"
    results_parquet = f"results/metrics/{output_prefix}__benchmark_results.parquet"
    parquet_exists = (METRICS_DIR / f"{output_prefix}__benchmark_results.parquet").exists()

    content = [
        "# Rare-cell downsampling benchmark report",
        "",
        "## Input",
        f"- Input file: {input_file}",
        f"- Label key: {label_key}",
        f"- Target label: {target_label}",
        f"- Representations: {', '.join(map(str, representation_keys))}",
        f"- Retain fractions: {', '.join(map(str, retain_fractions))}",
        f"- Seeds: {', '.join(map(str, seeds))}",
        f"- n_neighbors: {n_neighbors}",
        "",
        "## Dataset summary",
        f"- Total cells: {n_cells}",
        f"- Original target cells: {original_target_cells}",
        f"- Number of cell types: {n_cell_types}",
        "",
        "## Output files",
        f"- Results CSV: {results_csv}",
        f"- Results parquet: {results_parquet if parquet_exists else 'Skipped or unavailable'}",
        f"- Figures: {', '.join(figures) if figures else 'None'}",
        f"- Logs: {', '.join(logs) if logs else 'None'}",
        "",
        "## Main metric summary",
        table,
        "",
        "## Notes",
        "- This is an automatically generated report.",
        "",
    ]
    report_path.write_text("\n".join(content), encoding="utf-8")
    return report_path
