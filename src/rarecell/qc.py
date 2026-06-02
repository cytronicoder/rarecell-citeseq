"""QC and summary-table utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from .io import get_cell_labels, get_protein_matrix, get_rna_adata, validate_citeseq_object
from .utils import ensure_directory, to_dense_array, write_json


def _counts_per_axis(matrix: Any, axis: int) -> np.ndarray:
    values = matrix.to_numpy() if isinstance(matrix, pd.DataFrame) else matrix
    summed = values.sum(axis=axis)
    return np.asarray(summed).reshape(-1)


def _detected_per_axis(matrix: Any, axis: int) -> np.ndarray:
    values = matrix.to_numpy() if isinstance(matrix, pd.DataFrame) else matrix
    detected = (values > 0).sum(axis=axis)
    return np.asarray(detected).reshape(-1)


def summarize_anndata(adata: ad.AnnData) -> pd.DataFrame:
    """Summarize an AnnData object at dataset level."""
    values = to_dense_array(adata.X)
    counts_per_cell = values.sum(axis=1)
    genes_per_cell = (values > 0).sum(axis=1)
    cells_per_gene = (values > 0).sum(axis=0)
    return pd.DataFrame(
        {
            "n_cells": [adata.n_obs],
            "n_genes": [adata.n_vars],
            "total_counts_mean": [float(np.mean(counts_per_cell))],
            "total_counts_median": [float(np.median(counts_per_cell))],
            "genes_per_cell_mean": [float(np.mean(genes_per_cell))],
            "genes_per_cell_median": [float(np.median(genes_per_cell))],
            "detected_cells_per_gene_mean": [float(np.mean(cells_per_gene))],
        }
    )


def summarize_protein_matrix(protein: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Summarize a cell-by-protein matrix."""
    if not isinstance(protein, pd.DataFrame):
        protein = pd.DataFrame(to_dense_array(protein))
    values = protein.to_numpy()
    return pd.DataFrame(
        {
            "n_cells": [protein.shape[0]],
            "n_proteins": [protein.shape[1]],
            "total_counts_mean": [float(np.mean(values.sum(axis=1)))],
            "total_counts_median": [float(np.median(values.sum(axis=1)))],
            "zero_fraction": [float(np.mean(values == 0))],
        }
    )


def summarize_cell_labels(labels: pd.Series | None) -> pd.DataFrame:
    """Summarize cell-label counts."""
    if labels is None:
        return pd.DataFrame(columns=["label", "n_cells", "fraction"])
    counts = labels.astype(str).value_counts(dropna=False)
    return pd.DataFrame(
        {
            "label": counts.index,
            "n_cells": counts.to_numpy(),
            "fraction": counts.to_numpy() / counts.sum(),
        }
    )


def make_protein_feature_table(protein_matrix: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Build a CSV-ready per-protein summary table."""
    protein = protein_matrix if isinstance(protein_matrix, pd.DataFrame) else pd.DataFrame(
        to_dense_array(protein_matrix)
    )
    values = protein.to_numpy()
    total_counts = values.sum(axis=0)
    detected_cells = (values > 0).sum(axis=0)
    n_cells = max(1, protein.shape[0])
    return pd.DataFrame(
        {
            "protein_name": protein.columns.astype(str),
            "total_counts": total_counts.astype(float),
            "mean_counts": values.mean(axis=0).astype(float),
            "detected_cells": detected_cells.astype(int),
            "detected_fraction": detected_cells.astype(float) / n_cells,
        }
    )


def make_candidate_target_population_table(
        labels: pd.Series | None, min_cells: int = 50
) -> pd.DataFrame:
    """Rank label populations for future artificial rarity experiments."""
    columns = ["population", "n_cells", "fraction", "candidate_priority", "notes"]
    if labels is None:
        return pd.DataFrame(columns=columns)

    clean = labels.dropna().astype(str)
    if clean.empty:
        return pd.DataFrame(columns=columns)

    counts = clean.value_counts()
    total = counts.sum()
    preferred_upper = max(min_cells, int(np.ceil(total * 0.25)))
    rows = []
    for population, n_cells in counts.items():
        if n_cells < min_cells:
            priority = 3
            notes = f"Below min_cells={min_cells}; not recommended as first target."
        elif n_cells <= preferred_upper:
            priority = 1
            notes = "Preferred mid-sized population for artificial downsampling."
        else:
            priority = 2
            notes = "Abundant population; usable as an artificial rarity target."
        rows.append(
            {
                "population": population,
                "n_cells": int(n_cells),
                "fraction": float(n_cells / total),
                "candidate_priority": priority,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["candidate_priority", "n_cells", "population"], ascending=[True, True, True]
    ).reset_index(drop=True)


def make_rna_qc_cell_table(rna: ad.AnnData) -> pd.DataFrame:
    """Build a per-cell RNA QC table from counts and existing QC fields."""
    table = pd.DataFrame(index=rna.obs_names)
    table["total_counts"] = _counts_per_axis(rna.X, axis=1).astype(float)
    table["n_genes_by_counts"] = _detected_per_axis(rna.X, axis=1).astype(int)

    if "pct_counts_mt" in rna.obs:
        table["pct_counts_mt"] = pd.to_numeric(rna.obs["pct_counts_mt"], errors="coerce")
    elif "mt" in rna.var and bool(rna.var["mt"].any()):
        mt_mask = rna.var["mt"].to_numpy(dtype=bool)
        mt_counts = _counts_per_axis(rna[:, mt_mask].X, axis=1).astype(float)
        total = table["total_counts"].replace(0, np.nan)
        table["pct_counts_mt"] = (mt_counts / total * 100).fillna(0.0)
    else:
        table["pct_counts_mt"] = np.nan

    qc_terms = ("total_counts", "n_genes", "n_genes_by_counts", "pct_counts", "n_counts")
    for column in rna.obs.columns:
        if column in table.columns:
            continue
        if any(term in str(column) for term in qc_terms):
            table[column] = rna.obs[column].to_numpy()
    table.index.name = "cell_id"
    return table.reset_index()


def make_protein_qc_cell_table(protein: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Build a per-cell protein/ADT QC table."""
    matrix = protein if isinstance(protein, pd.DataFrame) else pd.DataFrame(to_dense_array(protein))
    values = matrix.to_numpy()
    table = pd.DataFrame(
        {
            "cell_id": matrix.index.astype(str),
            "total_protein_counts": values.sum(axis=1).astype(float),
            "detected_proteins": (values > 0).sum(axis=1).astype(int),
            "zero_fraction_per_cell": (values == 0).mean(axis=1).astype(float),
        }
    )
    return table


def make_dataset_summary(obj: Any) -> dict[str, Any]:
    """Build validation, RNA, protein, and label summaries."""
    rna = get_rna_adata(obj)
    labels = get_cell_labels(obj)
    validation = validate_citeseq_object(obj)
    summary: dict[str, Any] = {
        "validation": validation,
        "dataset_summary": summarize_anndata(rna),
        "cell_type_counts": summarize_cell_labels(labels),
    }

    protein = get_protein_matrix(obj)
    summary["protein_summary"] = summarize_protein_matrix(protein)
    return summary


def save_summary_tables(summary: dict[str, Any], out_dir: str | Path) -> dict[str, Path]:
    """Save summary tables and validation metadata to disk."""
    out = ensure_directory(out_dir)
    written: dict[str, Path] = {}
    for key, value in summary.items():
        if isinstance(value, pd.DataFrame):
            path = out / f"{key}.csv"
            value.to_csv(path, index=False)
            written[key] = path
        elif isinstance(value, dict):
            path = out / f"{key}.json"
            write_json(value, path)
            written[key] = path
    return written
