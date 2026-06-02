"""Centralized figure style, save, and axis utilities for benchmark figures."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl

try:
    import seaborn as sns
except ImportError:
    sns = None

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
    """Save a matplotlib figure to multiple formats.

    Parameters
    ----------
    fig:
        matplotlib Figure object.
    output_stem:
        Path without extension, e.g. results/figures/pbmc5k_b_cell__rare_cell_f1_curve
    formats:
        Output formats tuple, e.g. ("png", "pdf").
    dpi:
        Export DPI.

    Returns
    -------
    list[Path]
        Saved file paths.
    """
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fmt in formats:
        out = stem.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        logging.info("Saved figure: %s", out)
        saved.append(out)
    return saved


def clean_axis(
        ax,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
) -> None:
    """Apply consistent title, axis labels, grid, and spine formatting."""
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25)


def format_fraction_axis(ax, invert: bool = True) -> None:
    """Format the downsampling fraction axis consistently.

    x-axis = retained target-cell fraction.
    Label: 'Target-cell retained fraction'.
    If invert=True, lower retained fraction appears to the right.
    """
    ax.set_xlabel("Target-cell retained fraction")
    if invert:
        ax.invert_xaxis()
