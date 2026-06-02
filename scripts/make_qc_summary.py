#!/usr/bin/env python
"""Generate QC summary tables and optional label-count figures."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rarecell.config import FIGURES_DIR, LOGS_DIR, TABLES_DIR
from rarecell.io import get_cell_labels, get_protein_matrix, get_rna_adata, load_citeseq
from rarecell.plotting import plot_bar_counts, plot_histogram, plot_qc_summary
from rarecell.qc import (
    make_candidate_target_population_table,
    make_dataset_summary,
    make_protein_feature_table,
    make_protein_qc_cell_table,
    make_rna_qc_cell_table,
    save_summary_tables,
    summarize_cell_labels,
)
from rarecell.script_utils import setup_file_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="scvi:5k_pbmc_protein_v3_nextgem",
        help=(
            "Path to .h5ad or .h5mu CITE-seq file, "
            "local 10x .h5 or MEX directory, or a scvi/10x spec like "
            "'scvi:5k_pbmc_protein_v3_nextgem'."
        ),
    )
    parser.add_argument("--tables-dir", default=str(TABLES_DIR), help="Directory for summary tables.")
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR), help="Directory for summary figures.")
    parser.add_argument("--label-key", default=None, help="Preferred .obs label column.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_file_logger(LOGS_DIR / "qc_summary.log")
    Path(args.tables_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)

    logging.info("Loading %s", args.input)
    logging.info("Output tables: %s", args.tables_dir)
    logging.info("Output figures: %s", args.figures_dir)
    obj = load_citeseq(args.input)
    rna = get_rna_adata(obj)
    protein = get_protein_matrix(obj)
    logging.info("Dataset dimensions: cells=%d genes=%d protein_features=%d", rna.n_obs, rna.n_vars, protein.shape[1])
    preferred = [args.label_key] if args.label_key else None
    labels = get_cell_labels(obj, preferred_keys=preferred)

    summary = make_dataset_summary(obj)
    summary["cell_type_counts"] = summarize_cell_labels(labels)
    written = save_summary_tables(summary, args.tables_dir)
    for name, path in written.items():
        logging.info("Wrote %s: %s", name, path)

    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    rna_qc = make_rna_qc_cell_table(rna)
    protein_qc = make_protein_qc_cell_table(protein)
    protein_features = make_protein_feature_table(protein)

    rna_qc.to_csv(tables_dir / "rna_qc_cells.csv", index=False)
    protein_qc.to_csv(tables_dir / "protein_qc_cells.csv", index=False)
    protein_features.to_csv(tables_dir / "protein_features.csv", index=False)

    plot_qc_summary(summary, figures_dir / "qc_summary.png")
    plot_histogram(
        rna_qc["total_counts"],
        figures_dir / "rna_total_counts_hist.png",
        "Distribution of RNA UMI counts per cell",
        "RNA UMI counts per cell",
    )
    plot_histogram(
        rna_qc["n_genes_by_counts"],
        figures_dir / "rna_genes_per_cell_hist.png",
        "Distribution of detected genes per cell",
        "Detected genes per cell",
    )
    plot_histogram(
        protein_qc["total_protein_counts"],
        figures_dir / "protein_total_counts_hist.png",
        "Distribution of antibody-derived tag counts per cell",
        "ADT counts per cell",
    )

    if labels is not None:
        candidates = make_candidate_target_population_table(labels)
        candidates.to_csv(tables_dir / "candidate_target_populations.csv", index=False)
        plot_bar_counts(
            labels,
            figures_dir / "cell_label_counts.png",
            "Cell counts by annotated cell type",
            "Cell type",
            "Number of cells",
        )
        logging.info("Wrote cell-label count plot and candidate target table")
    else:
        logging.info("No cell-type labels found; skipping candidate target and label-count outputs")
    logging.info("Completed QC summary.")


if __name__ == "__main__":
    main()
