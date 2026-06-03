#!/usr/bin/env python
"""Import CITE-seq data into the standard benchmark AnnData layout."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rarecell.config import DATA_DIR, FIGURES_DIR, LOGS_DIR, TABLES_DIR
from rarecell.io import (
    get_cell_labels,
    get_protein_matrix,
    get_rna_adata,
    load_citeseq,
    make_candidate_target_population_table,
    make_dataset_summary,
    make_protein_feature_table,
    make_protein_qc_cell_table,
    make_rna_qc_cell_table,
    save_citeseq_object,
    save_summary_tables,
    summarize_cell_labels,
    to_internal_anndata,
)
from rarecell.plotting import plot_bar_counts, plot_histogram, plot_qc_summary
from rarecell.utils import setup_file_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="scvi:5k_pbmc_protein_v3_nextgem",
        help="Path to .h5ad/.h5mu, local 10x HDF5/MEX, or scvi/10x spec.",
    )
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "processed" / "pbmc5k_10x_citeseq_imported.h5ad"),
        help="Output AnnData in the standard benchmark layout.",
    )
    parser.add_argument("--dataset", default="pbmc5k_10x_citeseq", help="Dataset name stored in adata.uns.")
    parser.add_argument("--tables-dir", default=str(TABLES_DIR), help="Directory for QC tables.")
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR), help="Directory for QC figures.")
    parser.add_argument("--label-key", default=None, help="Preferred .obs label column.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_file_logger(LOGS_DIR / "import_data.log")
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading %s", args.input)
    obj = load_citeseq(args.input)
    labels = get_cell_labels(obj, preferred_keys=[args.label_key] if args.label_key else None)
    summary = make_dataset_summary(obj)
    summary["cell_type_counts"] = summarize_cell_labels(labels)
    save_summary_tables(summary, tables_dir)

    rna = get_rna_adata(obj)
    protein = get_protein_matrix(obj)
    make_rna_qc_cell_table(rna).to_csv(tables_dir / "rna_qc_cells.csv", index=False)
    protein_qc = make_protein_qc_cell_table(protein)
    protein_qc.to_csv(tables_dir / "protein_qc_cells.csv", index=False)
    make_protein_feature_table(protein).to_csv(tables_dir / "protein_features.csv", index=False)

    plot_qc_summary(summary, figures_dir / "qc_summary.png")
    rna_qc = make_rna_qc_cell_table(rna)
    plot_histogram(rna_qc["total_counts"], figures_dir / "rna_total_counts_hist.png", "RNA counts per cell",
                   "RNA counts")
    plot_histogram(
        protein_qc["total_protein_counts"],
        figures_dir / "protein_total_counts_hist.png",
        "Protein counts per cell",
        "Protein counts",
    )

    if labels is not None:
        make_candidate_target_population_table(labels).to_csv(tables_dir / "candidate_target_populations.csv",
                                                              index=False)
        plot_bar_counts(labels, figures_dir / "cell_label_counts.png", "Cell counts by label", "Label", "Cells")

    adata = to_internal_anndata(obj, dataset=args.dataset)
    output = Path(args.output)
    save_citeseq_object(adata, output)
    logging.info("Saved imported AnnData: %s", output)
    print(output)


if __name__ == "__main__":
    main()
