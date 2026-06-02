#!/usr/bin/env python
"""Write dataset and cell-type summaries for processed AnnData files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rarecell.utils import infer_label_column


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed", help="Directory containing .h5ad files.")
    parser.add_argument("--tables-dir", default="results/tables", help="Directory for summary CSVs.")
    return parser.parse_args()


def _protein_count(adata: ad.AnnData) -> int:
    if "protein_names" in adata.uns:
        return len(adata.uns["protein_names"])
    if "protein_counts" in adata.obsm:
        return int(adata.obsm["protein_counts"].shape[1])
    return 0


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    if not processed_dir.exists():
        raise SystemExit(f"Processed data directory does not exist: {processed_dir}")

    dataset_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for path in sorted(processed_dir.glob("*.h5ad")):
        adata = ad.read_h5ad(path)
        label_column = infer_label_column(adata)
        n_proteins = _protein_count(adata)
        dataset_rows.append(
            {
                "dataset": path.stem,
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "n_proteins": n_proteins,
                "has_cell_type_labels": bool(label_column is not None),
                "label_column": label_column or "",
                "notes": "processed AnnData discovered in data/processed",
            }
        )
        if label_column is not None:
            counts = adata.obs[label_column].astype(str).value_counts(dropna=False)
            total = max(1, int(counts.sum()))
            for cell_type, n_cells in counts.items():
                label_rows.append(
                    {
                        "dataset": path.stem,
                        "label_column": label_column,
                        "cell_type": str(cell_type),
                        "n_cells": int(n_cells),
                        "fraction": float(n_cells / total),
                    }
                )

    pd.DataFrame(
        dataset_rows,
        columns=["dataset", "n_cells", "n_genes", "n_proteins", "has_cell_type_labels", "label_column", "notes"],
    ).to_csv(tables_dir / "dataset_summary.csv", index=False)
    pd.DataFrame(
        label_rows,
        columns=["dataset", "label_column", "cell_type", "n_cells", "fraction"],
    ).to_csv(tables_dir / "cell_type_counts.csv", index=False)
    print(f"Saved summaries to {tables_dir}")


if __name__ == "__main__":
    main()
