"""Matplotlib plotting helpers for CITE-seq benchmark figures."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib as mpl

_MPLCONFIGDIR = Path(__file__).resolve().parents[2] / "results" / "intermediate" / "matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:
    sns = None

from .utils import to_dense_array

# ---------------------------------------------------------------------------
# Figure style and save utilities (from figure_utils.py)
# ---------------------------------------------------------------------------

FIGURE_DPI = 300
FIGURE_FORMATS = ("png", "pdf")
DEFAULT_FIGSIZE = (7, 5)
WIDE_FIGSIZE = (9, 5)
HEATMAP_FIGSIZE = (7, 4.8)
SMALL_FIGSIZE = (5, 4)


def set_plot_style() -> None:
    """Apply consistent matplotlib/seaborn style for all benchmark figures."""
    if sns is not None:
        sns.set_theme(
            context="paper",
            style="whitegrid",
            palette="colorblind",
            font_scale=1.1,
        )
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "legend.title_fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(
        fig,
        output_stem: str | Path,
        formats: tuple[str, ...] = FIGURE_FORMATS,
        dpi: int = FIGURE_DPI,
) -> list[Path]:
    """Save a matplotlib figure to multiple formats (PNG and PDF by default)."""
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fmt in formats:
        out = stem.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        logging.info("Saved figure: %s", out)
        saved.append(out)
    return saved


def _save_fig(fig, output_path: str | Path, dpi: int = 300) -> Path:
    """Save a matplotlib figure to a single path (extension preserved)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    return out


def _apply_style() -> None:
    set_plot_style()


METHOD_ORDER = ("RNA-only PCA", "Protein-only PCA", "RNA + protein PCA")

METHOD_COLORS = {
    "RNA-only PCA": "#4C78A8",
    "Protein-only PCA": "#F58518",
    "RNA + protein PCA": "#54A24B",
}

REPRESENTATION_LABELS = {
    "rna_pca": "RNA-only PCA",
    "protein_pca": "Protein-only PCA",
    "X_rna_pca": "RNA-only PCA",
    "X_protein_pca": "Protein-only PCA",
    "X_joint_simple": "RNA + protein PCA",
    "joint_pca": "RNA + protein PCA",
    "RNA-only PCA": "RNA-only PCA",
    "Protein-only PCA": "Protein-only PCA",
    "RNA + protein PCA": "RNA + protein PCA",
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
    return get_method_display_name(value)


def get_method_display_name(value: str) -> str:
    """Return the manuscript-facing method name for an internal representation key."""
    return REPRESENTATION_LABELS.get(str(value), str(value).replace("_", " "))


def order_methods(values) -> list[str]:
    """Return methods in the shared plotting order, preserving unknown methods afterward."""
    unique = list(dict.fromkeys(str(value) for value in values if pd.notna(value)))

    def sort_key(value: str) -> tuple[int, int | str]:
        display = get_method_display_name(value)
        if display in METHOD_ORDER:
            return (0, METHOD_ORDER.index(display))
        return (1, display)

    return sorted(unique, key=sort_key)


def get_method_colors(values=None) -> dict[str, str]:
    """Return stable colors keyed by both internal and display method labels."""
    colors = dict(METHOD_COLORS)
    if values is not None:
        fallback = plt.get_cmap("tab10")
        for index, value in enumerate(order_methods(values)):
            display = get_method_display_name(value)
            colors.setdefault(display, fallback(index % fallback.N))
            colors[str(value)] = colors[display]
    return colors


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


def format_fraction_labels(values) -> list[str]:
    """Return readable retained-fraction labels for a sequence of values."""
    return [format_fraction_label(value) for value in values]


def place_legend_outside(
        ax,
        *,
        handles=None,
        labels=None,
        title: str = "Representation",
        location: str = "right",
        ncol: int | None = None,
):
    """Place a legend outside the data area without colliding with titles."""
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    if location == "bottom":
        return ax.legend(
            handles,
            labels,
            title=title,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=ncol or min(3, len(labels)),
            frameon=False,
            fontsize=9,
            title_fontsize=10,
            borderaxespad=0,
        )
    return ax.legend(
        handles,
        labels,
        title=title,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
        title_fontsize=10,
        borderaxespad=0,
    )


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
    ax.set_xticklabels(format_fraction_labels(fractions))
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
    save_figure(fig, out.with_suffix(""), formats=FIGURE_FORMATS)
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
    save_figure(fig, out.with_suffix(""), formats=FIGURE_FORMATS)
    plt.close(fig)
    return out


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


def _metric_fraction_column(table: pd.DataFrame) -> str:
    if "fraction" in table.columns:
        return "fraction"
    if "retain_fraction" in table.columns:
        return "retain_fraction"
    raise KeyError("Expected a 'fraction' or 'retain_fraction' column.")


def _target_title(target: str) -> str:
    return f"Target: {format_category_label(target)}"


def _set_unit_interval_ylim(ax, values: pd.Series, errors: pd.Series | None = None) -> None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        ax.set_ylim(0, 1)
        return
    lower = clean.min()
    upper = clean.max()
    if errors is not None:
        err = pd.to_numeric(errors, errors="coerce").fillna(0)
        lower = min(lower, (clean - err).min())
        upper = max(upper, (clean + err).max())
    if lower >= 0 and upper <= 1.05:
        ax.set_ylim(0, 1)


def plot_multi_target_metric_curve(
        raw_df: pd.DataFrame,
        metric: str,
        output_path: str | Path,
        title: str | None = None,
) -> Path:
    """Plot a multi-target benchmark metric curve with variability across seeds."""
    if raw_df.empty or metric not in raw_df.columns:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        ax.text(0.5, 0.5, f"{metric} data unavailable.", ha="center", va="center")
        ax.set_title(title or "Multi-target benchmark")
        ax.set_axis_off()
        _save_fig(fig, out)
        plt.close(fig)
        return out
    table = raw_df.copy()
    fraction_col = _metric_fraction_column(table)
    if "representation" not in table.columns:
        raise KeyError("Column 'representation' is required for multi-target metric plotting.")
    if "target_cell_type" not in table.columns:
        table["target_cell_type"] = "target population"
    table[metric] = pd.to_numeric(table[metric], errors="coerce")
    summary = (
        table.groupby(["target_cell_type", "representation", fraction_col], dropna=False)[metric]
        .agg(["mean", "sem"])
        .reset_index()
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    targets = list(dict.fromkeys(summary["target_cell_type"].astype(str)))
    n_targets = max(1, len(targets))
    fig_width = min(13, max(6.4, 4.2 * n_targets + 1.6))
    fig, axes = plt.subplots(1, n_targets, figsize=(fig_width, 4.2), squeeze=False)
    colors = get_method_colors(summary["representation"])
    for ax, target in zip(axes[0], targets):
        target_table = summary.loc[summary["target_cell_type"].astype(str) == target]
        position_map = _fraction_axis_positions(target_table[fraction_col])
        for representation in order_methods(target_table["representation"]):
            group = target_table.loc[target_table["representation"].astype(str) == representation]
            group = group.sort_values(fraction_col, ascending=False)
            x_positions = group[fraction_col].astype(float).map(position_map)
            ax.errorbar(
                x_positions,
                group["mean"],
                yerr=group["sem"].fillna(0),
                marker="o",
                linewidth=1.6,
                capsize=2,
                markersize=4.5,
                color=colors.get(str(representation), colors.get(get_method_display_name(str(representation)))),
                label=get_method_display_name(str(representation)),
            )
        _set_fraction_axis(ax, position_map)
        if n_targets > 1:
            ax.set_title(_target_title(target), fontsize=12, pad=6)
        else:
            ax.set_title("", pad=0)
        ax.set_xlabel("Retained target-cell fraction")
        ax.set_ylabel(format_metric_label(metric))
        if metric in {"f1", "precision", "recall", "neighborhood_purity", "marker_auc"}:
            _set_unit_interval_ylim(ax, target_table["mean"], target_table["sem"])
        ax.grid(alpha=0.25)
        _style_axes(ax)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="Representation",
            loc="center left",
            bbox_to_anchor=(1.005, 0.5),
            frameon=False,
            fontsize=9,
            title_fontsize=10,
        )
    default_title = f"{format_metric_label(metric)} across target cell types"
    if n_targets == 1:
        default_title = f"{format_metric_label(metric)} under controlled downsampling"
    fig.suptitle(title or default_title, fontsize=15, y=0.97)
    fig.subplots_adjust(left=0.11, right=0.78 if handles else 0.96, top=0.84, bottom=0.17, wspace=0.35)
    _save_fig(fig, out)
    plt.close(fig)
    return out


def plot_cross_target_method_ranking(ranking_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot mean cross-target rank for each representation and metric."""
    if ranking_df.empty:
        return _save_placeholder(
            output_path,
            "Cross-target method ranking",
            "Cross-target method ranking table is unavailable.",
        )
    table = ranking_df.copy()
    table["mean_rank"] = pd.to_numeric(table["mean_rank"], errors="coerce")
    metrics = list(dict.fromkeys(table["metric"].astype(str)))
    reps = order_methods(table["representation"])
    x = np.arange(len(reps))
    width = 0.8 / max(1, len(metrics))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7, 1.25 * len(reps)), 4.2))
    for index, metric in enumerate(metrics):
        values = (
            table.loc[table["metric"].astype(str) == metric]
            .set_index("representation")["mean_rank"]
            .reindex(reps)
        )
        ax.bar(x - 0.4 + width / 2 + index * width, values, width=width, label=format_metric_label(metric))
    ax.set_xticks(x)
    ax.set_xticklabels([get_method_display_name(rep) for rep in reps], rotation=20, ha="right")
    ax.set_ylabel("Mean rank across benchmark conditions")
    ax.set_title("Cross-target method ranking", fontsize=15, pad=8)
    ax.invert_yaxis()
    ax.text(
        0.01,
        0.98,
        "Lower rank is better",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#555555",
    )
    if len(metrics) > 1:
        place_legend_outside(ax, title="Metric")
    ax.grid(axis="y", alpha=0.25)
    _style_axes(ax)
    fig.subplots_adjust(right=0.78 if len(metrics) > 1 else 0.96, top=0.88, bottom=0.25)
    _save_fig(fig, out)
    plt.close(fig)
    return out


def plot_study_design(output_path: str | Path, config: dict | None = None) -> Path:
    """Draw a schematic study design figure for the benchmark workflow."""
    cfg = config or {}
    targets = cfg.get("target_cell_types") or [cfg.get("target_cell_type", "target cell types")]
    fractions = cfg.get("fractions", [1.0, 0.5, 0.25, 0.1, 0.05])
    reps = cfg.get("representations", ["rna_pca", "protein_pca", "joint_pca"])
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 3.0))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    target_text = "\n".join(map(str, targets[:3]))
    if len(targets) > 3:
        target_text += f"\n+{len(targets) - 3} more"
    boxes = [
        ("CITE-seq input", "RNA matrix\nprotein counts\ncell labels"),
        ("Target populations", target_text),
        ("Controlled\ndownsampling", ", ".join(format_fraction_label(v) for v in fractions)),
        ("Representations", "\n".join(get_method_display_name(v) for v in reps)),
        ("Metrics", "F1 score\nneighborhood purity\nmarker ROC-AUC"),
    ]
    x_positions = np.linspace(0.095, 0.905, len(boxes))
    for i, ((heading, body), x_pos) in enumerate(zip(boxes, x_positions)):
        rect = plt.Rectangle(
            (x_pos - 0.078, 0.24),
            0.156,
            0.5,
            facecolor="#F3F4F6",
            edgecolor="#4B5563",
            lw=1.1,
        )
        ax.add_patch(rect)
        ax.text(x_pos, 0.62, heading, ha="center", va="center", fontsize=10, weight="bold", linespacing=1.05)
        ax.text(x_pos, 0.42, body, ha="center", va="center", fontsize=8.5, linespacing=1.1)
        if i < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(x_positions[i + 1] - 0.085, 0.49),
                xytext=(x_pos + 0.085, 0.49),
                arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#4B5563"},
            )
    fig.suptitle("Rare-cell preservation benchmark study design", fontsize=14, y=0.95)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.06)
    _save_fig(fig, out)
    plt.close(fig)
    return out


def plot_final_figure_2_main_benchmark(raw_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save the main multi-target F1 benchmark figure."""
    return plot_multi_target_metric_curve(
        raw_df,
        metric="f1",
        output_path=output_path,
        title="Rare-cell recovery under controlled downsampling",
    )


# ---------------------------------------------------------------------------
# Standard benchmark figure set (from benchmark_plots.py)
# ---------------------------------------------------------------------------

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

_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
_FIGURES_DIR = _RESULTS_DIR / "figures"
_TABLES_DIR = _RESULTS_DIR / "tables"


def _bp_metric_label(metric: str) -> str:
    return str(metric).replace("_", " ").title()


def _bp_format_fraction(value: float) -> str:
    return f"{float(value):g}"


def _bp_style_axis(ax) -> None:
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
    """Plot metric versus retain_fraction grouped by representation.

    Saves both PNG and PDF. Returns list of saved paths.
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
    ax.set_ylabel(_bp_metric_label(metric))
    ax.set_title(title or f"{_bp_metric_label(metric)} across target-cell downsampling")
    ax.legend(title="Representation", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    _bp_style_axis(ax)
    fig.tight_layout()
    paths = save_figure(fig, stem, formats=formats)
    plt.close(fig)
    return paths


def plot_target_counts(
        results_df: pd.DataFrame,
        output_stem: str | Path,
        formats: tuple[str, ...] = FIGURE_FORMATS,
) -> list[Path]:
    """Plot retained target-cell count versus retain_fraction."""
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
    _bp_style_axis(ax)
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
    """Create a representation-by-fraction heatmap of mean metric values."""
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
            cbar_kws={"label": f"Mean {_bp_metric_label(metric)}"},
            ax=ax,
        )
    else:
        image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([_bp_format_fraction(v) for v in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index.astype(str))
        for row in range(pivot.shape[0]):
            for col in range(pivot.shape[1]):
                value = pivot.iloc[row, col]
                if pd.notna(value):
                    ax.text(col, row, f"{value:.2f}", ha="center", va="center", color="white")
        fig.colorbar(image, ax=ax, label=f"Mean {_bp_metric_label(metric)}")
    ax.set_xlabel("Retained target-cell fraction")
    ax.set_ylabel("Representation")
    ax.set_title(f"Mean {_bp_metric_label(metric)} across downsampling levels")
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

    Uses canonical double-underscore naming: {output_prefix}__{descriptor}.{ext}.
    Also writes a figure index CSV.

    Returns a flat list of all saved file paths.
    """
    figs_dir = Path(figures_dir) if figures_dir is not None else _FIGURES_DIR
    tabs_dir = Path(tables_dir) if tables_dir is not None else _TABLES_DIR
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
                "created_by": "save_all_standard_plots",
            }
        )

    figure_index = pd.DataFrame(index_rows)
    index_path = tabs_dir / f"{output_prefix}__figure_index.csv"
    figure_index.to_csv(index_path, index=False)

    return all_paths
