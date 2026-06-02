"""Reusable plots for rare-cell downsampling benchmark results."""

from __future__ import annotations

import os
from pathlib import Path

_MPLCONFIGDIR = Path(__file__).resolve().parents[1] / "results" / "intermediate" / "matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:
    sns = None

from rarecell.figure_utils import FIGURE_FORMATS, save_figure, set_plot_style

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

_METRIC_DESCRIPTORS: dict[str, tuple[str, str]] = {
    "f1": ("rare_cell_f1_curve", "Rare-cell F1 score vs. retained fraction"),
    "recall": ("rare_cell_recall_curve", "Rare-cell recall vs. retained fraction"),
    "precision": ("rare_cell_precision_curve", "Rare-cell precision vs. retained fraction"),
    "neighborhood_purity": ("neighborhood_purity_curve", "Neighborhood purity vs. retained fraction"),
    "target_silhouette": ("target_silhouette_curve", "Target silhouette score vs. retained fraction"),
}

_HEATMAP_DESCRIPTORS: dict[str, tuple[str, str]] = {
    "f1": ("rare_cell_f1_heatmap", "Mean rare-cell F1 score by representation and fraction"),
    "recall": ("rare_cell_recall_heatmap", "Mean rare-cell recall by representation and fraction"),
    "neighborhood_purity": ("neighborhood_purity_heatmap", "Mean neighborhood purity by representation and fraction"),
    "target_silhouette": ("target_silhouette_heatmap", "Mean target silhouette by representation and fraction"),
}


def _metric_label(metric: str) -> str:
    return str(metric).replace("_", " ").title()


def _format_fraction(value: float) -> str:
    return f"{float(value):g}"


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25)


def plot_metric_curve(
        results_df: pd.DataFrame,
        metric: str,
        output_stem: str | Path,
        title: str | None = None,
        formats: tuple[str, ...] = FIGURE_FORMATS,
) -> list[Path]:
    """Plot metric versus retain_fraction grouped by representation and save to output_stem.

    Saves both PNG and PDF by default. Returns list of saved paths.
    """
    if metric not in results_df.columns:
        raise KeyError(f"Metric '{metric}' is not present in results_df.")

    set_plot_style()
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    table = results_df.copy()
    table[metric] = pd.to_numeric(table[metric], errors="coerce")
    table["retain_fraction"] = pd.to_numeric(table["retain_fraction"], errors="coerce")
    summary = (
        table.groupby(["representation", "retain_fraction"], dropna=False)[metric]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("retain_fraction", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for representation, group in summary.groupby("representation", sort=False):
        group = group.sort_values("retain_fraction", ascending=False)
        ax.errorbar(
            group["retain_fraction"].astype(float),
            group["mean"],
            yerr=group["std"].fillna(0),
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=str(representation),
        )

    positive_fractions = summary["retain_fraction"].dropna().astype(float)
    if (
            not positive_fractions.empty
            and bool((positive_fractions > 0).all())
            and float(positive_fractions.max() / positive_fractions.min()) >= 10
    ):
        ax.set_xscale("log")
    if not positive_fractions.empty:
        ax.invert_xaxis()
    ax.set_xlabel("Target-cell retained fraction")
    ax.set_ylabel(_metric_label(metric))
    ax.set_title(title or f"{_metric_label(metric)} across target-cell downsampling")
    ax.legend(title="Representation", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    _style_axis(ax)
    fig.tight_layout()
    paths = save_figure(fig, stem, formats=formats)
    plt.close(fig)
    return paths


def plot_target_counts(
        results_df: pd.DataFrame,
        output_stem: str | Path,
        formats: tuple[str, ...] = FIGURE_FORMATS,
) -> list[Path]:
    """Plot retained target-cell count versus retain_fraction and save to output_stem."""
    if "n_target" not in results_df.columns:
        raise KeyError("Column 'n_target' is not present in results_df.")

    set_plot_style()
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    table = results_df.copy()
    table["n_target"] = pd.to_numeric(table["n_target"], errors="coerce")
    table["retain_fraction"] = pd.to_numeric(table["retain_fraction"], errors="coerce")
    summary = (
        table.drop_duplicates(["retain_fraction", "seed", "n_target"])
        .groupby("retain_fraction", dropna=False)["n_target"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("retain_fraction", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.errorbar(
        summary["retain_fraction"].astype(float),
        summary["mean"],
        yerr=summary["std"].fillna(0),
        marker="o",
        linewidth=1.8,
        capsize=3,
        color="#4C78A8",
    )
    positive_fractions = summary["retain_fraction"].dropna().astype(float)
    if (
            not positive_fractions.empty
            and bool((positive_fractions > 0).all())
            and float(positive_fractions.max() / positive_fractions.min()) >= 10
    ):
        ax.set_xscale("log")
    if not positive_fractions.empty:
        ax.invert_xaxis()
    ax.set_xlabel("Target-cell retained fraction")
    ax.set_ylabel("Retained target cells")
    ax.set_title("Target-cell counts after downsampling")
    _style_axis(ax)
    fig.tight_layout()
    paths = save_figure(fig, stem, formats=formats)
    plt.close(fig)
    return paths


def plot_metric_heatmap(
        results_df: pd.DataFrame,
        metric: str,
        output_stem: str | Path,
        formats: tuple[str, ...] = FIGURE_FORMATS,
) -> list[Path]:
    """Create a representation-by-fraction heatmap of mean metric values and save to output_stem."""
    if metric not in results_df.columns:
        raise KeyError(f"Metric '{metric}' is not present in results_df.")

    set_plot_style()
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    table = results_df.copy()
    table[metric] = pd.to_numeric(table[metric], errors="coerce")
    table["retain_fraction"] = pd.to_numeric(table["retain_fraction"], errors="coerce")
    pivot = table.pivot_table(
        index="representation",
        columns="retain_fraction",
        values=metric,
        aggfunc="mean",
    )
    pivot = pivot.reindex(sorted(pivot.columns, reverse=True), axis=1)

    fig, ax = plt.subplots(
        figsize=(max(6.5, 0.8 * len(pivot.columns)), max(3, 0.5 * len(pivot.index)))
    )
    if sns is not None:
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            cbar_kws={"label": f"Mean {_metric_label(metric)}"},
            ax=ax,
        )
    else:
        image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([_format_fraction(v) for v in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index.astype(str))
        for row in range(pivot.shape[0]):
            for col in range(pivot.shape[1]):
                value = pivot.iloc[row, col]
                if pd.notna(value):
                    ax.text(col, row, f"{value:.2f}", ha="center", va="center", color="white")
        fig.colorbar(image, ax=ax, label=f"Mean {_metric_label(metric)}")
    ax.set_xlabel("Retained target-cell fraction")
    ax.set_ylabel("Representation")
    ax.set_title(f"Mean {_metric_label(metric)} across downsampling levels")
    fig.tight_layout()
    paths = save_figure(fig, stem, formats=formats)
    plt.close(fig)
    return paths


def save_all_standard_plots(
        results_df: pd.DataFrame,
        output_prefix: str,
        figures_dir: str | Path | None = None,
        tables_dir: str | Path | None = None,
) -> list[Path]:
    """Save all standard benchmark figures as PNG and PDF.

    Uses canonical double-underscore file naming:
    {output_prefix}__{figure_descriptor}.{extension}

    Also writes a figure index CSV to results/tables/{output_prefix}__figure_index.csv.

    Returns a flat list of all saved file paths.
    """
    figs_dir = Path(figures_dir) if figures_dir is not None else FIGURES_DIR
    tabs_dir = Path(tables_dir) if tables_dir is not None else TABLES_DIR
    figs_dir.mkdir(parents=True, exist_ok=True)
    tabs_dir.mkdir(parents=True, exist_ok=True)

    all_paths: list[Path] = []
    index_rows: list[dict] = []

    for metric, (descriptor, description) in _METRIC_DESCRIPTORS.items():
        if metric not in results_df.columns:
            continue
        stem = figs_dir / f"{output_prefix}__{descriptor}"
        paths = plot_metric_curve(results_df, metric, stem)
        all_paths.extend(paths)
        png = next((p for p in paths if p.suffix == ".png"), None)
        pdf = next((p for p in paths if p.suffix == ".pdf"), None)
        index_rows.append(
            {
                "figure_name": descriptor,
                "description": description,
                "png_path": str(png) if png else "",
                "pdf_path": str(pdf) if pdf else "",
                "svg_path": "",
                "created_by": "save_all_standard_plots",
            }
        )

    counts_stem = figs_dir / f"{output_prefix}__target_cell_counts"
    counts_paths = plot_target_counts(results_df, counts_stem)
    all_paths.extend(counts_paths)
    png = next((p for p in counts_paths if p.suffix == ".png"), None)
    pdf = next((p for p in counts_paths if p.suffix == ".pdf"), None)
    index_rows.append(
        {
            "figure_name": "target_cell_counts",
            "description": "Target-cell counts after downsampling by fraction",
            "png_path": str(png) if png else "",
            "pdf_path": str(pdf) if pdf else "",
            "svg_path": "",
            "created_by": "save_all_standard_plots",
        }
    )

    for metric, (descriptor, description) in _HEATMAP_DESCRIPTORS.items():
        if metric not in results_df.columns:
            continue
        stem = figs_dir / f"{output_prefix}__{descriptor}"
        paths = plot_metric_heatmap(results_df, metric, stem)
        all_paths.extend(paths)
        png = next((p for p in paths if p.suffix == ".png"), None)
        pdf = next((p for p in paths if p.suffix == ".pdf"), None)
        index_rows.append(
            {
                "figure_name": descriptor,
                "description": description,
                "png_path": str(png) if png else "",
                "pdf_path": str(pdf) if pdf else "",
                "svg_path": "",
                "created_by": "save_all_standard_plots",
            }
        )

    figure_index = pd.DataFrame(index_rows)
    index_path = tabs_dir / f"{output_prefix}__figure_index.csv"
    figure_index.to_csv(index_path, index=False)

    return all_paths
