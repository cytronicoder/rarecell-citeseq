#!/usr/bin/env python
"""Run the config-driven rare-cell CITE-seq downsampling benchmark."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rarecell.benchmark import (
    rebuild_baseline_representations,
    run_downsampling_benchmark,
)
from rarecell.downsampling import downsample_target_cells
from rarecell.metrics import compute_all_metrics
from rarecell.plotting import plot_recovery_curve
from rarecell.representations import compute_umap_from_embedding
from rarecell.utils import infer_label_column

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
METRICS = RESULTS / "metrics"
MANUSCRIPT = ROOT / "manuscript"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/benchmark_config.yaml", help="Benchmark YAML config.")
    return parser.parse_args()


def _mkdirs() -> None:
    for path in [FIGURES, TABLES, METRICS, MANUSCRIPT]:
        path.mkdir(parents=True, exist_ok=True)


def _read_config(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise SystemExit(f"Expected a mapping in config file: {path}")
    return config


def _choose_target(labels: pd.Series, requested: str | None = None) -> str:
    clean = labels.dropna().astype(str)
    if clean.empty:
        raise ValueError("Cannot choose a target cell type from an empty label column.")
    if requested:
        if requested not in set(clean):
            raise ValueError(f"Requested target_cell_type '{requested}' is absent from labels.")
        return requested
    counts = clean.value_counts()
    return str(counts.index[min(1, len(counts) - 1)])


def _choose_control(labels: pd.Series, target: str, requested: str | None = None) -> str | None:
    clean = labels.dropna().astype(str)
    if requested:
        if requested not in set(clean):
            raise ValueError(f"Requested control_cell_type '{requested}' is absent from labels.")
        return requested
    counts = clean[clean != str(target)].value_counts()
    return str(counts.index[0]) if not counts.empty else None


def _write_dataset_tables(adata: ad.AnnData, label_column: str) -> None:
    labels = adata.obs[label_column].astype(str)
    dataset_summary = pd.DataFrame(
        [
            {
                "dataset": str(adata.uns.get("dataset", "unknown")),
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "n_protein_features": int(adata.obsm["protein_counts"].shape[1])
                if "protein_counts" in adata.obsm
                else 0,
                "label_column": label_column,
                "n_cell_types": int(labels.nunique(dropna=False)),
            }
        ]
    )
    dataset_summary.to_csv(TABLES / "dataset_summary.csv", index=False)
    labels.value_counts(dropna=False).rename_axis("cell_type").reset_index(name="n_cells").to_csv(
        TABLES / "cell_type_counts.csv", index=False
    )


def _plot_umap_comparison(adata: ad.AnnData, label_column: str, target: str, fractions: list[float], seed: int) -> None:
    severe_fraction = min(map(float, fractions))
    adata_sub = downsample_target_cells(adata, label_column, target, severe_fraction, seed=seed)
    rebuilt = rebuild_baseline_representations(adata_sub)
    labels = rebuilt.obs[label_column].astype(str)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
    for ax, (key, title) in zip(axes[0], [
        ("X_rna_pca", "RNA-only PCA"),
        ("X_protein_pca", "Protein-only PCA"),
        ("X_joint_simple", "RNA + protein PCA"),
    ]):
        umap = compute_umap_from_embedding(rebuilt.obsm[key], random_state=seed)
        values = umap.to_numpy()
        is_target = labels.to_numpy() == str(target)
        ax.scatter(values[~is_target, 0], values[~is_target, 1], s=5, alpha=0.45, color="#8C8C8C", linewidths=0)
        ax.scatter(values[is_target, 0], values[is_target, 1], s=9, alpha=0.9, color="#D62728", linewidths=0)
        ax.set_title(title)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"Representation UMAPs at retained fraction {severe_fraction:g}")
    fig.tight_layout()
    fig.savefig(FIGURES / "representation_umap_comparison_rare_fraction.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _random_label_control(
        adata: ad.AnnData,
        label_column: str,
        target: str,
        representations: list[str],
        fraction: float,
        seed: int,
        n_neighbors: int,
) -> pd.DataFrame:
    adata_sub = downsample_target_cells(adata, label_column, target, fraction, seed=seed)
    rebuilt = rebuild_baseline_representations(adata_sub)
    labels = rebuilt.obs[label_column].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    permuted = labels.copy()
    rng.shuffle(permuted)
    key_map = {"rna_pca": "X_rna_pca", "protein_pca": "X_protein_pca", "joint_pca": "X_joint_simple"}
    rows = []
    for representation in representations:
        key = key_map.get(representation, representation)
        if key not in rebuilt.obsm:
            continue
        metrics = compute_all_metrics(rebuilt.obsm[key], permuted, target, seed=seed, n_neighbors=n_neighbors)
        rows.append(
            {
                "control": "random_label",
                "target_cell_type": target,
                "representation": representation,
                "fraction": float(fraction),
                "seed": int(seed),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _write_report(config: dict, target: str, control: str | None, data_available: bool) -> None:
    path = MANUSCRIPT / "mini_report_week4.md"
    if not data_available:
        content = """# Rare-cell CITE-seq Benchmark Mini-Report

## Status
Real benchmark results were not generated because the configured input data file is missing.

## Required Data
Place an internal CITE-seq AnnData file at the configured `dataset_path`, with RNA in `adata.X`, protein counts in `adata.obsm["protein_counts"]`, and protein names in `adata.uns["protein_names"]`.

## TODO
- Import public 10x PBMC CITE-seq data.
- Preprocess RNA and protein.
- Rebuild RNA-only, protein-only, and simple joint representations inside each downsampling condition.
- Generate benchmark metrics, controls, figures, and final interpretation.
"""
    else:
        content = f"""# Rare-cell CITE-seq Benchmark Mini-Report

## Project Question
Do RNA-only, protein-only, and simple RNA+protein representations preserve a target cell type after artificial rarity is introduced?

## Dataset Used
Dataset: `{config.get("dataset", "unknown")}`. Input path: `{config.get("dataset_path")}`.

## Preprocessing
RNA counts were normalized, log-transformed, filtered, scaled, and embedded with PCA. Protein counts were log-transformed, standardized, and embedded with PCA when enough ADT features were available.

## Target Cell Type
Target: `{target}`. Abundant-cell control: `{control or "not available"}`.

## Downsampling Design
Only target cells were downsampled across fractions `{config.get("fractions")}` and seeds `{config.get("seeds")}`. Non-target cells were held fixed.

## Representations Compared
`rna_pca`, `protein_pca`, and `joint_pca`. Representations were recomputed after each downsampling condition.

## Metrics
Rare-cell precision, recall, F1, neighborhood purity, and target-vs-rest silhouette.

## First Results
See `results/metrics/rare_cell_benchmark_raw.csv` and `results/tables/benchmark_summary.csv`.

## Controls
The abundant-cell control and random-label control are saved under `results/metrics` and `results/tables`.

## Limitations
This CPU-only benchmark uses simple PCA-style baselines and does not train totalVI, WNN, or other advanced models.

## Next Steps
Validate on additional public CITE-seq datasets and add biological marker checks for the selected target population.
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    _mkdirs()
    config = _read_config(ROOT / args.config)
    input_path = ROOT / str(config.get("dataset_path", "data/processed/pbmc10k_10x_citeseq_representations.h5ad"))
    if not input_path.exists():
        _write_report(config, target="TODO", control=None, data_available=False)
        raise SystemExit(
            f"Input data file is missing: {input_path}\n"
            "Expected an internal .h5ad with RNA in X and ADT counts in obsm['protein_counts'].\n"
            "Run scripts/import_10x_pbmc10k.py or scripts/import_scvi_totalvi.py first."
        )

    adata = ad.read_h5ad(input_path)
    if "protein_counts" not in adata.obsm:
        raise SystemExit(
            "Input AnnData is missing adata.obsm['protein_counts']; import data into the internal layout first.")
    label_column = config.get("label_column") or infer_label_column(adata)
    if label_column is None:
        raise SystemExit("No label column found. Add label_column to config/benchmark_config.yaml.")
    labels = adata.obs[label_column].astype(str)
    target = _choose_target(labels, config.get("target_cell_type"))
    control = _choose_control(labels, target, config.get("control_cell_type"))
    fractions = [float(value) for value in config.get("fractions", [1.0, 0.5, 0.25, 0.1, 0.05])]
    seeds = [int(value) for value in config.get("seeds", [0, 1, 2, 3, 4])]
    representations = list(config.get("representations", ["rna_pca", "protein_pca", "joint_pca"]))
    n_neighbors = int(config.get("k_neighbors", 15))

    _write_dataset_tables(adata, label_column)
    results = run_downsampling_benchmark(
        adata,
        label_key=label_column,
        target_label=target,
        representation_keys=representations,
        retain_fractions=fractions,
        seeds=seeds,
        n_neighbors=n_neighbors,
        dataset=str(config.get("dataset", input_path.stem)),
    )
    results.to_csv(METRICS / "rare_cell_benchmark_raw.csv", index=False)
    summary = (
        results.groupby(["target_cell_type", "representation", "fraction"], dropna=False)[
            ["precision", "recall", "f1", "neighborhood_purity", "silhouette_target"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(map(str, column)).rstrip("_") for column in summary.columns]
    summary.to_csv(TABLES / "benchmark_summary.csv", index=False)

    plot_recovery_curve(results, "f1", FIGURES / "rare_cell_f1_curve.png")
    plot_recovery_curve(results, "recall", FIGURES / "rare_cell_recall_curve_initial.png")
    plot_recovery_curve(results, "neighborhood_purity", FIGURES / "neighborhood_purity_curve.png")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _plot_umap_comparison(adata, label_column, target, fractions, seeds[0])

    if control is not None:
        control_results = run_downsampling_benchmark(
            adata,
            label_key=label_column,
            target_label=control,
            representation_keys=representations,
            retain_fractions=fractions,
            seeds=seeds,
            n_neighbors=n_neighbors,
            dataset=str(config.get("dataset", input_path.stem)),
        )
        control_results.to_csv(METRICS / "control_abundant_cell_downsampling.csv", index=False)
        plot_recovery_curve(control_results, "f1", FIGURES / "control_abundant_cell_downsampling.png")

    random_control = _random_label_control(
        adata,
        label_column,
        target,
        representations,
        min(fractions),
        seeds[0],
        n_neighbors,
    )
    random_control.to_csv(TABLES / "random_label_control.csv", index=False)
    _write_report(config, target=target, control=control, data_available=True)
    print("Benchmark complete.")
    print(f"Raw metrics: {METRICS / 'rare_cell_benchmark_raw.csv'}")


if __name__ == "__main__":
    main()
