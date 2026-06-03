"""Benchmark runner, result validation, summaries, and reports."""

from __future__ import annotations

import logging
import time
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
from rarecell.utils import resolve_representations, standard_representation_name

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
        logger: logging.Logger | None = None,
        show_progress: bool = False,
) -> pd.DataFrame:
    """Run the full benchmark, rebuilding representations after downsampling."""
    log = logger or logging.getLogger(__name__)
    validate_target_label(adata, label_key, target_label)
    requested_representations = representation_keys or list(REPRESENTATION_KEY_MAP)
    original_labels = adata.obs[label_key].astype(str)
    n_target_original = int((original_labels == str(target_label)).sum())

    rows = []
    conditions = [(float(retain_fraction), int(seed)) for retain_fraction in retain_fractions for seed in seeds]
    iterator = conditions
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(conditions, desc=f"benchmark target={target_label}")
        except Exception:
            log.info("tqdm is unavailable; benchmark progress will be reported through log messages.")

    for condition_index, (retain_fraction, seed) in enumerate(iterator, start=1):
        condition_started = time.perf_counter()
        log.info(
            "Running benchmark condition %d/%d: target=%s fraction=%s seed=%s",
            condition_index,
            len(conditions),
            target_label,
            retain_fraction,
            seed,
        )
        adata_sub = downsample_target_cells(
            adata,
            label_key=label_key,
            target_label=target_label,
            retain_fraction=retain_fraction,
            seed=seed,
        )
        rebuild_started = time.perf_counter()
        rebuilt = rebuild_baseline_representations(
            adata_sub,
            rna_pcs=int(rna_pcs),
            protein_pcs=int(protein_pcs),
            n_top_genes=int(n_top_genes),
        )
        log.info(
            "Rebuilt representations for target=%s fraction=%s seed=%s in %.2fs",
            target_label,
            retain_fraction,
            seed,
            time.perf_counter() - rebuild_started,
        )
        representation_pairs = resolve_representations(rebuilt, requested_representations)
        labels = rebuilt.obs[label_key].astype(str).to_numpy()
        for representation_name, representation_key in representation_pairs:
            metric_started = time.perf_counter()
            embedding = rebuilt.obsm[representation_key]
            metrics = compute_all_metrics(
                embedding,
                labels,
                target_label,
                seed=seed,
                n_neighbors=int(n_neighbors),
            )
            row = {
                "dataset": str(dataset),
                "target_cell_type": str(target_label),
                "target_label": str(target_label),
                "label_key": str(label_key),
                "fraction": retain_fraction,
                "retain_fraction": retain_fraction,
                "seed": seed,
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
            log.info(
                "Computed metrics for representation=%s target=%s fraction=%s seed=%s in %.2fs",
                representation_name,
                target_label,
                retain_fraction,
                seed,
                time.perf_counter() - metric_started,
            )
        log.info(
            "Finished benchmark condition target=%s fraction=%s seed=%s in %.2fs",
            target_label,
            retain_fraction,
            seed,
            time.perf_counter() - condition_started,
        )

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


# ---------------------------------------------------------------------------
# Summary table generators (from summaries.py)
# ---------------------------------------------------------------------------

_METRIC_COLS = ["precision", "recall", "f1", "neighborhood_purity", "silhouette_target"]
_GROUP_COLS = ["target_cell_type", "representation", "fraction"]


def make_benchmark_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw benchmark results across seeds."""
    if raw_df.empty:
        return pd.DataFrame(
            columns=["target_cell_type", "representation", "fraction", "metric", "mean", "std", "n_seeds", "n_valid"]
        )

    present_metrics = [m for m in _METRIC_COLS if m in raw_df.columns]
    fraction_col = "fraction" if "fraction" in raw_df.columns else "retain_fraction"

    rows: list[dict] = []
    for (target, rep, frac), group in raw_df.groupby(
            ["target_cell_type", "representation", fraction_col], dropna=False
    ):
        n_seeds = int(len(group))
        for metric in present_metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            valid = values.dropna()
            rows.append(
                {
                    "target_cell_type": str(target),
                    "representation": str(rep),
                    "fraction": float(frac),
                    "metric": metric,
                    "mean": float(valid.mean()) if len(valid) > 0 else float("nan"),
                    "std": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
                    "n_seeds": n_seeds,
                    "n_valid": int(len(valid)),
                }
            )
    return pd.DataFrame(rows)


def make_best_method_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """For each (target_cell_type, fraction, metric), find the best representation."""
    if summary_df.empty:
        return pd.DataFrame(
            columns=[
                "target_cell_type", "fraction", "metric",
                "best_representation", "best_mean",
                "second_best_representation", "second_best_mean", "delta",
            ]
        )

    rows: list[dict] = []
    for (target, frac, metric), group in summary_df.groupby(
            ["target_cell_type", "fraction", "metric"], dropna=False
    ):
        ranked = group.sort_values("mean", ascending=False).reset_index(drop=True)
        if ranked.empty:
            continue
        best_rep = str(ranked.loc[0, "representation"])
        best_mean = float(ranked.loc[0, "mean"]) if pd.notna(ranked.loc[0, "mean"]) else float("nan")
        if len(ranked) >= 2:
            second_rep = str(ranked.loc[1, "representation"])
            second_mean = float(ranked.loc[1, "mean"]) if pd.notna(ranked.loc[1, "mean"]) else float("nan")
        else:
            second_rep = ""
            second_mean = float("nan")
        delta = float(best_mean - second_mean) if (pd.notna(best_mean) and pd.notna(second_mean)) else float("nan")
        rows.append(
            {
                "target_cell_type": str(target),
                "fraction": float(frac),
                "metric": metric,
                "best_representation": best_rep,
                "best_mean": best_mean,
                "second_best_representation": second_rep,
                "second_best_mean": second_mean,
                "delta": delta,
            }
        )
    return pd.DataFrame(rows)


def make_abundant_control_summary(control_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize abundant-cell control results across seeds."""
    if control_df.empty:
        return pd.DataFrame(
            columns=["target_cell_type", "representation", "fraction", "metric", "mean", "std", "n_seeds", "n_valid"]
        )

    fraction_col = "fraction" if "fraction" in control_df.columns else "retain_fraction"
    target_col = "target_cell_type" if "target_cell_type" in control_df.columns else "target_label"
    present_metrics = [m for m in _METRIC_COLS if m in control_df.columns]

    rows: list[dict] = []
    for (target, rep, frac), group in control_df.groupby(
            [target_col, "representation", fraction_col], dropna=False
    ):
        n_seeds = int(len(group))
        for metric in present_metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            valid = values.dropna()
            rows.append(
                {
                    "target_cell_type": str(target),
                    "representation": str(rep),
                    "fraction": float(frac),
                    "metric": metric,
                    "mean": float(valid.mean()) if len(valid) > 0 else float("nan"),
                    "std": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
                    "n_seeds": n_seeds,
                    "n_valid": int(len(valid)),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Control analyses (from controls.py)
# ---------------------------------------------------------------------------


def select_control_cell_type(
        labels: pd.Series,
        target: str,
        requested: str | None = None,
) -> str | None:
    """Return the most abundant non-target cell type as the abundant-cell control."""
    clean = labels.dropna().astype(str)
    if requested:
        if str(requested) not in set(clean):
            raise ValueError(
                f"Requested control_cell_type '{requested}' is absent from labels. "
                f"Available: {sorted(set(clean))}."
            )
        return str(requested)
    non_target = clean[clean != str(target)]
    counts = non_target.value_counts()
    if counts.empty:
        return None
    return str(counts.index[0])


def run_random_label_control(
        adata: Any,
        label_key: str,
        target_label: str,
        representations: list[str],
        retain_fraction: float,
        seed: int,
        n_neighbors: int = 15,
) -> pd.DataFrame:
    """Run a single random-label control condition.

    Permutes cell-type labels (preserving marginal distribution) then evaluates
    all representations on the permuted labels.
    """
    adata_sub = downsample_target_cells(
        adata,
        label_key=label_key,
        target_label=target_label,
        retain_fraction=float(retain_fraction),
        seed=int(seed),
    )
    rebuilt = rebuild_baseline_representations(adata_sub)
    labels = rebuilt.obs[label_key].astype(str).to_numpy()
    rng = np.random.default_rng(int(seed))
    permuted = labels.copy()
    rng.shuffle(permuted)

    rep_pairs = resolve_representations(rebuilt, representations)
    rows: list[dict] = []
    for rep_name, rep_key in rep_pairs:
        if rep_key not in rebuilt.obsm:
            continue
        metrics = compute_all_metrics(
            rebuilt.obsm[rep_key], permuted, target_label,
            seed=int(seed), n_neighbors=int(n_neighbors),
        )
        rows.append(
            {
                "control": "random_label",
                "target_cell_type": str(target_label),
                "representation": rep_name,
                "fraction": float(retain_fraction),
                "seed": int(seed),
                "precision": metrics.get("precision", float("nan")),
                "recall": metrics.get("recall", float("nan")),
                "f1": metrics.get("f1", float("nan")),
                "neighborhood_purity": metrics.get("neighborhood_purity", float("nan")),
                "silhouette_target": metrics.get("silhouette_target", float("nan")),
            }
        )
    return pd.DataFrame(rows)
