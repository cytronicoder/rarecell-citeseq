#!/usr/bin/env python
"""Generate all canonical benchmark figures from raw results CSVs."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"
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
        help="Path to abundant-cell control raw CSV.",
    )
    p.add_argument(
        "--config",
        default="config/benchmark_config.yaml",
        help="Benchmark YAML config.",
    )
    p.add_argument(
        "--out-dir",
        default=str(FIGURES_DIR),
        help="Output directory for figures.",
    )
    p.add_argument(
        "--skip-umap",
        action="store_true",
        help="Skip the UMAP comparison figure (slow; requires AnnData).",
    )
    return p.parse_args()


def _read_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _plot_recovery_curve(results_df: pd.DataFrame, metric: str, out_path: Path, title: str | None = None) -> None:
    from rarecell.plotting import plot_recovery_curve
    plot_recovery_curve(results_df, metric, out_path, title=title)
    print(f"  Saved: {out_path}")


def _plot_cell_counts(results_df: pd.DataFrame, control_df: pd.DataFrame | None, out_path: Path) -> None:
    from rarecell.plotting import format_fraction_label

    fig, ax = plt.subplots(figsize=(8, 4))
    fraction_col = "fraction" if "fraction" in results_df.columns else "retain_fraction"

    target_summary = (
        results_df.groupby(fraction_col, dropna=False)["n_target"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(fraction_col, ascending=False)
    )
    fractions_ordered = list(target_summary[fraction_col].astype(float))
    positions = list(range(len(fractions_ordered)))
    pos_map = {f: i for i, f in enumerate(fractions_ordered)}

    x_pos = target_summary[fraction_col].astype(float).map(pos_map)
    ax.errorbar(
        x_pos, target_summary["mean"], yerr=target_summary["std"].fillna(0),
        marker="o", linewidth=1.8, capsize=3, label="Target cell type", color="#D62728",
    )

    if control_df is not None and "n_target" in control_df.columns:
        ctrl_frac_col = "fraction" if "fraction" in control_df.columns else "retain_fraction"
        control_summary = (
            control_df.groupby(ctrl_frac_col, dropna=False)["n_target"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values(ctrl_frac_col, ascending=False)
        )
        ctrl_x = control_summary[ctrl_frac_col].astype(float).map(pos_map)
        ax.errorbar(
            ctrl_x, control_summary["mean"], yerr=control_summary["std"].fillna(0),
            marker="s", linewidth=1.8, capsize=3, label="Abundant-cell control", color="#1F77B4",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([format_fraction_label(f) for f in fractions_ordered])
    ax.set_xlabel("Retained fraction")
    ax.set_ylabel("Retained cells")
    ax.set_title("Cell counts after downsampling")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _plot_control_comparison(
        results_df: pd.DataFrame, control_df: pd.DataFrame, metric: str, out_path: Path
) -> None:
    from rarecell.plotting import format_representation_label, format_fraction_label

    fraction_col = "fraction" if "fraction" in results_df.columns else "retain_fraction"
    ctrl_frac_col = "fraction" if "fraction" in control_df.columns else "retain_fraction"

    target_summary = (
        results_df.groupby(["representation", fraction_col], dropna=False)[metric]
        .agg(["mean", "std"])
        .reset_index()
    )
    control_summary = (
        control_df.groupby(["representation", ctrl_frac_col], dropna=False)[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    fractions = sorted(results_df[fraction_col].dropna().unique().astype(float), reverse=True)
    pos_map = {f: i for i, f in enumerate(fractions)}

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors_target = ["#D62728", "#FF7F0E", "#9467BD"]
    colors_control = ["#1F77B4", "#2CA02C", "#8C564B"]

    for idx, (rep, group) in enumerate(target_summary.groupby("representation", sort=False)):
        group = group.sort_values(fraction_col, ascending=False)
        x_pos = group[fraction_col].astype(float).map(pos_map)
        color = colors_target[idx % len(colors_target)]
        ax.errorbar(
            x_pos, group["mean"], yerr=group["std"].fillna(0),
            marker="o", linewidth=1.8, capsize=3, linestyle="-",
            label=f"{format_representation_label(str(rep))} (target)", color=color,
        )

    for idx, (rep, group) in enumerate(control_summary.groupby("representation", sort=False)):
        group = group.sort_values(ctrl_frac_col, ascending=False)
        x_pos = group[ctrl_frac_col].astype(float).map(pos_map)
        color = colors_control[idx % len(colors_control)]
        ax.errorbar(
            x_pos, group["mean"], yerr=group["std"].fillna(0),
            marker="s", linewidth=1.8, capsize=3, linestyle="--",
            label=f"{format_representation_label(str(rep))} (control)", color=color,
        )

    ax.set_xticks(list(pos_map.values()))
    ax.set_xticklabels([format_fraction_label(f) for f in fractions])
    ax.set_xlabel("Retained fraction")
    metric_label = metric.replace("_", " ").title()
    ax.set_ylabel(f"Mean {metric_label}")
    ax.set_title(f"{metric_label}: Target vs. abundant-cell control")
    ax.legend(
        frameon=False, fontsize=8, title_fontsize=9,
        loc="upper left", bbox_to_anchor=(1.02, 1),
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _plot_umap_comparison(config: dict, out_path: Path, severe_fraction: float | None = None) -> None:
    import anndata as ad
    from rarecell.downsampling import downsample_target_cells
    from rarecell.benchmark import rebuild_baseline_representations
    from rarecell.representations import compute_umap_from_embedding
    from rarecell.utils import infer_label_column

    dataset_path = ROOT / str(config.get("dataset_path", ""))
    if not dataset_path.exists():
        print(f"  UMAP skipped: dataset not found at {dataset_path}")
        return

    adata = ad.read_h5ad(dataset_path)
    label_key = config.get("label_column") or infer_label_column(adata)
    if label_key is None:
        print("  UMAP skipped: no label column found.")
        return

    labels = adata.obs[label_key].astype(str)
    target = config.get("target_cell_type") or str(labels.value_counts().index[min(1, len(labels.value_counts()) - 1)])
    fractions = [float(f) for f in config.get("fractions", [1.0, 0.5, 0.25, 0.1, 0.05])]
    seed = int(config.get("seeds", [0])[0])

    configured_severe = config.get("severe_fraction_for_umap")
    if configured_severe is not None:
        use_fraction = float(configured_severe)
    elif severe_fraction is not None:
        use_fraction = float(severe_fraction)
    else:
        use_fraction = min(fractions)

    if str(target) not in set(labels):
        print(f"  UMAP skipped: target '{target}' not in labels.")
        return

    adata_sub = downsample_target_cells(adata, label_key, target, use_fraction, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rebuilt = rebuild_baseline_representations(adata_sub)

    cell_labels = rebuilt.obs[label_key].astype(str)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
    rep_items = [
        ("X_rna_pca", "RNA-only PCA"),
        ("X_protein_pca", "Protein-only PCA"),
        ("X_joint_simple", "RNA + protein PCA"),
    ]
    is_target = cell_labels.to_numpy() == str(target)

    for ax, (key, title) in zip(axes[0], rep_items):
        if key not in rebuilt.obsm:
            ax.set_title(f"{title}\n(not available)")
            ax.set_axis_off()
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            umap_df = compute_umap_from_embedding(rebuilt.obsm[key], random_state=seed)
        values = umap_df.to_numpy()
        ax.scatter(
            values[~is_target, 0], values[~is_target, 1],
            s=5, alpha=0.45, color="#8C8C8C", linewidths=0, label="Other",
        )
        ax.scatter(
            values[is_target, 0], values[is_target, 1],
            s=15, alpha=0.9, color="#D62728", linewidths=0, label=f"Target ({target})",
        )
        ax.set_title(title)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax is axes[0][-1]:
            ax.legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle(
        f"Representation UMAPs at retained fraction {use_fraction:g} "
        f"(target: leiden cluster {target})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_csv = Path(args.raw_csv)
    if not raw_csv.exists():
        print(f"ERROR: Raw benchmark CSV not found: {raw_csv}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading raw benchmark results from {raw_csv}")
    results_df = pd.read_csv(raw_csv)
    fraction_col = "fraction" if "fraction" in results_df.columns else "retain_fraction"
    if fraction_col == "retain_fraction":
        results_df["fraction"] = results_df["retain_fraction"]

    control_df: pd.DataFrame | None = None
    control_csv = Path(args.control_csv)
    if not control_csv.exists():
        fallback = METRICS_DIR / "control_abundant_cell_downsampling.csv"
        if fallback.exists():
            control_csv = fallback
        else:
            control_csv = None

    if control_csv is not None:
        print(f"Reading abundant-cell control from {control_csv}")
        control_df = pd.read_csv(control_csv)
        frac_col_c = "fraction" if "fraction" in control_df.columns else "retain_fraction"
        if frac_col_c == "retain_fraction":
            control_df["fraction"] = control_df["retain_fraction"]

    config = _read_config(ROOT / args.config)

    print("Generating rare_cell_f1_curve.png …")
    _plot_recovery_curve(results_df, "f1", out / "rare_cell_f1_curve.png")

    print("Generating rare_cell_recall_curve_initial.png …")
    _plot_recovery_curve(results_df, "recall", out / "rare_cell_recall_curve_initial.png")

    print("Generating neighborhood_purity_curve.png …")
    _plot_recovery_curve(results_df, "neighborhood_purity", out / "neighborhood_purity_curve.png")

    print("Generating downsampling_cell_counts.png …")
    _plot_cell_counts(results_df, control_df, out / "downsampling_cell_counts.png")

    if control_df is not None:
        print("Generating control_abundant_cell_downsampling.png …")
        _plot_control_comparison(
            results_df, control_df, "f1", out / "control_abundant_cell_downsampling.png"
        )

    if not args.skip_umap:
        print("Generating representation_umap_comparison_rare_fraction.png …")
        _plot_umap_comparison(config, out / "representation_umap_comparison_rare_fraction.png")

    print("Figure generation complete.")


if __name__ == "__main__":
    main()
