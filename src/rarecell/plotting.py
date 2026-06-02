"""Matplotlib plotting helpers for exploratory CITE-seq figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from rarecell.figure_utils import FIGURE_FORMATS, save_figure, set_plot_style as _set_plot_style

    _HAS_FIGURE_UTILS = True
except ImportError:
    _HAS_FIGURE_UTILS = False

from .utils import ensure_directory, to_dense_array


def _apply_style() -> None:
    if _HAS_FIGURE_UTILS:
        _set_plot_style()


REPRESENTATION_LABELS = {
    "rna_pca": "RNA-only PCA",
    "protein_pca": "Protein-only PCA",
    "X_rna_pca": "RNA-only PCA",
    "X_protein_pca": "Protein-only PCA",
    "X_joint_simple": "RNA + protein PCA",
    "joint_pca": "RNA + protein PCA",
}

METRIC_LABELS = {
    "precision": "Rare-cell precision",
    "recall": "Rare-cell recall",
    "f1": "Rare-cell F1 score",
    "macro_f1": "Macro-F1 across cell types",
    "neighborhood_purity": "Target-cell neighborhood purity",
    "target_silhouette": "Target-vs-other silhouette score",
    "marker_auc": "Marker ROC-AUC",
}

PANEL_METRIC_TITLES = {
    "f1": "F1 score",
    "recall": "Recall",
    "neighborhood_purity": "Neighborhood purity",
    "target_silhouette": "Silhouette score",
}


def format_representation_label(value: str) -> str:
    """Return a human-readable representation label."""
    return REPRESENTATION_LABELS.get(str(value), str(value).replace("_", " "))


def format_metric_label(value: str) -> str:
    """Return a human-readable metric label."""
    return METRIC_LABELS.get(str(value), str(value).replace("_", " "))


def format_fraction_label(value) -> str:
    """Return a readable retained-fraction label, e.g. 1.0 -> '100%'."""
    try:
        percent = float(value) * 100
    except (TypeError, ValueError):
        return str(value).replace("_", " ")
    return f"{percent:g}%"


def format_category_label(value: str) -> str:
    """Return a human-readable label for plotted cell groups."""
    text = str(value)
    if text.startswith("cluster_"):
        return f"Cluster {text.removeprefix('cluster_')}"
    if text.startswith("cluster "):
        return f"Cluster {text.removeprefix('cluster ')}"
    return text.replace("_", " ")


def _infer_legend_title(values: pd.Index | list[str], default: str) -> str:
    labels = [str(value) for value in values]
    if labels and all(label.startswith(("cluster_", "cluster ")) for label in labels):
        return "Cluster"
    return default


def _fraction_axis_positions(values: pd.Series) -> dict[float, int]:
    fractions = sorted(pd.Series(values).dropna().astype(float).unique(), reverse=True)
    return {fraction: index for index, fraction in enumerate(fractions)}


def _set_fraction_axis(ax, position_map: dict[float, int]) -> None:
    fractions = list(position_map.keys())
    positions = list(position_map.values())
    ax.set_xticks(positions)
    ax.set_xticklabels([format_fraction_label(value) for value in fractions])
    ax.set_xlim(min(positions) - 0.25, max(positions) + 0.25)


def _style_axes(ax) -> None:
    ax.title.set_fontsize(13)
    ax.xaxis.label.set_fontsize(11)
    ax.yaxis.label.set_fontsize(11)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_histogram(
        values: pd.Series | np.ndarray | list[float],
        out_path: str | Path,
        title: str,
        xlabel: str,
        ylabel: str = "Number of cells",
        bins: int = 50,
) -> Path:
    """Save a simple histogram with matplotlib only."""
    _apply_style()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    clean = pd.Series(values).dropna().astype(float)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(clean, bins=bins, color="#4C78A8", edgecolor="white", linewidth=0.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _style_axes(ax)
    fig.tight_layout()
    if _HAS_FIGURE_UTILS:
        save_figure(fig, out.with_suffix(""), formats=FIGURE_FORMATS)
    else:
        fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_bar_counts(
        series_or_counts: pd.Series | dict[str, int],
        out_path: str | Path,
        title: str,
        xlabel: str,
        ylabel: str,
) -> Path:
    """Save a count bar plot from raw labels or precomputed counts."""
    _apply_style()
    if isinstance(series_or_counts, pd.Series):
        looks_like_counts = (
                pd.api.types.is_numeric_dtype(series_or_counts)
                and not isinstance(series_or_counts.index, pd.RangeIndex)
                and len(series_or_counts) <= 50
        )
        if looks_like_counts:
            counts = series_or_counts.copy()
        else:
            counts = series_or_counts.astype(str).value_counts()
    else:
        counts = pd.Series(series_or_counts)
    counts = counts.sort_values(ascending=False)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(counts)), 4))
    counts.plot(kind="bar", ax=ax, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=45)
    _style_axes(ax)
    fig.tight_layout()
    if _HAS_FIGURE_UTILS:
        save_figure(fig, out.with_suffix(""), formats=FIGURE_FORMATS)
    else:
        fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_dataset_dimensions(validation_dict: dict[str, Any], out_path: str | Path) -> Path:
    """Save a dataset-dimension bar plot from validation metadata."""
    values = {
        "Cells": validation_dict.get("n_cells", 0),
        "Genes": validation_dict.get("n_genes", 0),
        "Protein features": validation_dict.get("n_protein_features", 0),
    }
    return plot_bar_counts(
        pd.Series(values),
        out_path,
        "CITE-seq dataset dimensions",
        "Feature group",
        "Number of features or cells",
    )


def plot_cell_type_counts(labels: pd.Series, out_path: str | Path) -> Path:
    """Save a bar plot of cell-type counts."""
    return plot_bar_counts(
        labels,
        out_path,
        "Cell counts by annotated cell type",
        "Cell type",
        "Number of cells",
    )


def plot_embedding(
        embedding: pd.DataFrame | np.ndarray,
        labels: pd.Series | None = None,
        title: str = "",
        out_path: str | Path | None = None,
        legend_title: str = "Cell type",
) -> Path | None:
    """Plot the first two embedding dimensions, optionally colored by labels."""
    emb = embedding if isinstance(embedding, pd.DataFrame) else pd.DataFrame(to_dense_array(embedding))
    if emb.shape[1] < 2:
        values = np.column_stack([emb.iloc[:, 0].to_numpy(), np.zeros(emb.shape[0])])
    else:
        values = emb.iloc[:, :2].to_numpy()

    fig, ax = plt.subplots(figsize=(7.5, 5))
    has_legend = False
    if labels is None:
        ax.scatter(values[:, 0], values[:, 1], s=8, alpha=0.8, color="#4C78A8", linewidths=0)
    else:
        aligned = labels.reindex(emb.index) if isinstance(labels, pd.Series) else pd.Series(labels, index=emb.index)
        codes, uniques = pd.factorize(aligned.astype(str), sort=True)
        scatter = ax.scatter(values[:, 0], values[:, 1], c=codes, s=8, alpha=0.85, cmap="tab20", linewidths=0)
        if len(uniques) <= 20:
            handles, _ = scatter.legend_elements(num=len(uniques))
            display_labels = [format_category_label(value) for value in uniques]
            display_title = _infer_legend_title(uniques, legend_title)
            ax.legend(
                handles,
                display_labels,
                title=display_title,
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                borderaxespad=0,
                fontsize=8,
                title_fontsize=9,
                frameon=False,
            )
            has_legend = True
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    _style_axes(ax)
    fig.tight_layout(rect=(0, 0, 0.82, 1) if has_legend else None)

    if out_path is None:
        plt.close(fig)
        return None
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_qc_summary(summary: dict[str, Any], out_path: str | Path) -> Path:
    """Save a compact QC overview plot from summary tables."""
    validation = summary.get("validation", {})
    values = {
        "Cells": validation.get("n_cells", 0),
        "Genes": validation.get("n_genes", 0),
        "Protein features": validation.get("n_protein_features", 0),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(values.keys(), values.values(), color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylabel("Number of features or cells")
    ax.set_title("Quality-control summary for the CITE-seq dataset")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def save_all_baseline_figures(
        out_dir: str | Path,
        rna_umap: pd.DataFrame,
        protein_umap: pd.DataFrame,
        joint_umap: pd.DataFrame,
        labels: pd.Series | None = None,
) -> dict[str, Path]:
    """Save the first exploratory embedding figures."""
    out = ensure_directory(out_dir)
    paths: dict[str, Path] = {}
    if labels is not None:
        paths["cell_type_counts"] = plot_cell_type_counts(labels, out / "cell_type_counts.png")
    paths["rna_umap"] = plot_embedding(rna_umap, labels, "RNA-only PCA representation", out / "rna_umap.png")
    paths["protein_umap"] = plot_embedding(
        protein_umap, labels, "Protein-only PCA representation", out / "protein_umap.png"
    )
    paths["joint_umap"] = plot_embedding(
        joint_umap, labels, "Joint RNA–protein representation", out / "joint_umap.png"
    )
    return {key: path for key, path in paths.items() if path is not None}


def _metric_summary(results_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    table = results_df.copy()
    table[metric] = pd.to_numeric(table[metric], errors="coerce")
    grouped = table.groupby(["representation", "retain_fraction"], dropna=False)[metric]
    return grouped.agg(["mean", "sem"]).reset_index()


def plot_recovery_curve(
        results_df,
        metric: str,
        output_path,
        title: str | None = None,
):
    """Plot mean metric recovery across downsampling fractions."""
    if metric not in results_df:
        raise KeyError(f"Metric '{metric}' is not present in results_df.")
    summary = _metric_summary(results_df, metric)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 4))
    position_map = _fraction_axis_positions(summary["retain_fraction"])
    for representation, group in summary.groupby("representation", sort=False):
        group = group.sort_values("retain_fraction", ascending=False)
        x_positions = group["retain_fraction"].astype(float).map(position_map)
        ax.errorbar(
            x_positions,
            group["mean"],
            yerr=group["sem"].fillna(0),
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=format_representation_label(str(representation)),
        )
    _set_fraction_axis(ax, position_map)
    metric_label = format_metric_label(metric)
    ax.set_xlabel("Retained target-cell fraction")
    ax.set_ylabel(metric_label)
    ax.set_title(title or f"{metric_label} under controlled downsampling")
    ax.grid(alpha=0.25)
    ax.legend(
        title="Representation",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )
    _style_axes(ax)
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_metric_heatmap(
        results_df,
        metric: str,
        output_path,
):
    """Plot a representation-by-fraction heatmap of mean metric values."""
    if metric not in results_df:
        raise KeyError(f"Metric '{metric}' is not present in results_df.")
    table = results_df.copy()
    table[metric] = pd.to_numeric(table[metric], errors="coerce")
    pivot = table.pivot_table(
        index="representation",
        columns="retain_fraction",
        values=metric,
        aggfunc="mean",
    )
    pivot = pivot.reindex(sorted(pivot.columns, reverse=True), axis=1)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7.5, 0.8 * len(pivot.columns)), max(3, 0.45 * len(pivot.index))))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([format_fraction_label(value) for value in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([format_representation_label(value) for value in pivot.index.astype(str)])
    ax.set_xlabel("Retained target-cell fraction")
    ax.set_ylabel("Representation")
    metric_label = format_metric_label(metric)
    mean_metric_label = f"Mean {metric_label[0].lower()}{metric_label[1:]}" if metric_label else "Mean metric"
    ax.set_title(f"{mean_metric_label} across downsampling levels")
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.iloc[row, col]
            if pd.notna(value):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, label=mean_metric_label)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_target_cell_counts(
        results_df,
        output_path,
):
    """Plot retained target-cell counts across downsampling fractions."""
    if "n_target_after" not in results_df:
        raise KeyError("Column 'n_target_after' is not present in results_df.")
    table = results_df.copy()
    grouped = table.groupby(["representation", "retain_fraction"], dropna=False)["n_target_after"].mean().reset_index()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    position_map = _fraction_axis_positions(grouped["retain_fraction"])
    for representation, group in grouped.groupby("representation", sort=False):
        group = group.sort_values("retain_fraction", ascending=False)
        x_positions = group["retain_fraction"].astype(float).map(position_map)
        ax.plot(
            x_positions,
            group["n_target_after"],
            marker="o",
            label=format_representation_label(str(representation)),
        )
    _set_fraction_axis(ax, position_map)
    ax.set_xlabel("Retained target-cell fraction")
    ax.set_ylabel("Number of target cells")
    ax.set_title("Target-cell counts after downsampling")
    ax.grid(alpha=0.25)
    if grouped["representation"].nunique() > 1:
        ax.legend(title="Representation", frameon=False, fontsize=9, title_fontsize=10)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_benchmark_summary_panel(
        results_df,
        output_path,
):
    """Create a compact multi-panel benchmark metric summary."""
    metrics = [metric for metric in ["f1", "recall", "neighborhood_purity", "target_silhouette"] if
               metric in results_df]
    if not metrics:
        raise ValueError("No supported benchmark metrics are present in results_df.")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_panels = len(metrics)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4 * n_rows), squeeze=False)
    axes_flat = axes.reshape(-1)

    for ax, metric in zip(axes_flat, metrics):
        summary = _metric_summary(results_df, metric)
        position_map = _fraction_axis_positions(summary["retain_fraction"])
        for representation, group in summary.groupby("representation", sort=False):
            group = group.sort_values("retain_fraction", ascending=False)
            x_positions = group["retain_fraction"].astype(float).map(position_map)
            ax.errorbar(
                x_positions,
                group["mean"],
                yerr=group["sem"].fillna(0),
                marker="o",
                linewidth=1.5,
                capsize=2,
                label=format_representation_label(str(representation)),
            )
        _set_fraction_axis(ax, position_map)
        ax.set_title(PANEL_METRIC_TITLES.get(metric, format_metric_label(metric)))
        ax.set_xlabel("Retained target-cell fraction")
        ax.set_ylabel(format_metric_label(metric))
        ax.grid(alpha=0.25)
        _style_axes(ax)

    for ax in axes_flat[n_panels:]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="Representation",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=min(3, len(labels)),
            frameon=False,
            fontsize=9,
            title_fontsize=10,
        )
    fig.suptitle("Rare-cell recovery metrics across downsampling levels", fontsize=14, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_marker_auc_results(
        table: pd.DataFrame,
        output_path: str | Path,
        target_label: str | None = None,
) -> Path:
    """Plot marker ROC-AUC values or a clear placeholder when no markers are available."""
    _apply_style()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    title_target = str(target_label) if target_label else "the selected target population"

    if table.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "No marker features were available for the selected target population.",
            ha="center",
            va="center",
            wrap=True,
        )
        ax.set_title(f"Marker discrimination of {title_target}")
        ax.set_axis_off()
        fig.tight_layout()
        if _HAS_FIGURE_UTILS:
            save_figure(fig, out.with_suffix(""), formats=FIGURE_FORMATS)
        else:
            fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out

    plot_table = table.sort_values("auc", ascending=False).copy()
    plot_table["auc"] = pd.to_numeric(plot_table["auc"], errors="coerce")
    marker_sources = set(plot_table.get("marker_source", pd.Series(dtype=str)).astype(str))
    is_data_driven = marker_sources == {"data_driven_cluster_marker"}
    hue_column = "marker_source" if len(marker_sources) > 1 else "modality"
    legend_title = "Marker source" if hue_column == "marker_source" else "Modality"
    title = (
        f"Data-driven RNA markers for {title_target}"
        if is_data_driven
        else f"Marker discrimination of {title_target}"
    )

    markers = list(dict.fromkeys(plot_table["marker"].astype(str)))
    groups = list(dict.fromkeys(plot_table[hue_column].astype(str)))
    colors = {
        "rna": "#4C78A8",
        "protein": "#F58518",
        "data_driven_cluster_marker": "#4C78A8",
        "predefined_biological_marker": "#F58518",
    }
    display_group = {
        "rna": "RNA",
        "protein": "Protein",
        "data_driven_cluster_marker": "Data-driven cluster marker",
        "predefined_biological_marker": "Predefined biological marker",
    }

    fig, ax = plt.subplots(figsize=(max(7, 0.45 * max(1, len(markers))), 4.5))
    bar_width = 0.8 / max(1, len(groups))
    x_positions = np.arange(len(markers))
    for offset, group in enumerate(groups):
        values = (
            plot_table.loc[plot_table[hue_column].astype(str) == group]
            .set_index("marker")["auc"]
            .reindex(markers)
        )
        positions = x_positions - 0.4 + bar_width / 2 + offset * bar_width
        ax.bar(
            positions,
            values,
            width=bar_width,
            label=display_group.get(group, group.replace("_", " ")),
            color=colors.get(group, "#54A24B"),
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(markers, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Marker ROC-AUC")
    ax.set_xlabel("Marker")
    ax.set_title(title)
    if groups:
        ax.legend(title=legend_title, frameon=False, fontsize=9, title_fontsize=10)
    _style_axes(ax)
    fig.tight_layout()
    if _HAS_FIGURE_UTILS:
        save_figure(fig, out.with_suffix(""), formats=FIGURE_FORMATS)
    else:
        fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out
