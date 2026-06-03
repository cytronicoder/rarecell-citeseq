#!/usr/bin/env python
"""Preprocess CITE-seq RNA/protein data and build PCA benchmark representations."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rarecell.config import DATA_DIR, LOGS_DIR, REPRESENTATION_KEY_MAP, TABLES_DIR
from rarecell.io import (
    get_cell_labels,
    get_protein_matrix,
    get_rna_adata,
    load_citeseq,
    make_candidate_target_population_table,
    save_citeseq_object,
)
from rarecell.preprocessing import align_cells_between_modalities, preprocess_protein, preprocess_rna
from rarecell.representations import compute_joint_pca_representation, compute_protein_pca, compute_rna_pca
from rarecell.utils import setup_file_logger, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(DATA_DIR / "processed" / "pbmc5k_10x_citeseq_imported.h5ad"),
        help="Imported .h5ad/.h5mu, 10x HDF5/MEX path, or scvi/10x spec.",
    )
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "processed" / "pbmc5k_10x_citeseq_representations.h5ad"),
        help="Output AnnData with benchmark representations.",
    )
    parser.add_argument("--tables-dir", default=str(TABLES_DIR), help="Directory for small summary tables.")
    parser.add_argument("--rna-pcs", type=int, default=30)
    parser.add_argument("--protein-pcs", type=int, default=10)
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--label-key", default=None, help="Preferred .obs label column.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_file_logger(LOGS_DIR / "preprocess_data.log")
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading %s", args.input)
    obj = load_citeseq(args.input)
    labels = get_cell_labels(obj, preferred_keys=[args.label_key] if args.label_key else None)
    rna = get_rna_adata(obj)
    protein = get_protein_matrix(obj)
    rna, protein = align_cells_between_modalities(rna, protein)

    if "counts" not in rna.layers:
        rna.layers["counts"] = rna.X.copy()
    raw_protein = protein.copy()
    if labels is not None:
        labels = labels.reindex(rna.obs_names)

    logging.info("Preprocessing RNA")
    adata = preprocess_rna(rna, n_top_genes=args.n_top_genes, n_pcs=args.rna_pcs)
    logging.info("Preprocessing protein")
    protein_processed = preprocess_protein(protein).loc[adata.obs_names].copy()
    raw_protein = raw_protein.loc[adata.obs_names].copy()

    logging.info("Building baseline representations")
    rna_pca = compute_rna_pca(adata, n_components=args.rna_pcs)
    protein_pca = compute_protein_pca(protein_processed, n_components=args.protein_pcs)
    joint_pca = compute_joint_pca_representation(rna_pca, protein_pca, scale_blocks=True)

    if labels is not None:
        adata.obs["cell_type_simple"] = labels.reindex(adata.obs_names)
    if "leiden" not in adata.obs and "neighbors" in adata.uns:
        try:
            import scanpy as sc

            sc.tl.leiden(adata, random_state=0)
        except Exception as exc:
            logging.warning("Leiden clustering failed; continuing without leiden labels. %s", exc)

    adata.obsm["X_rna_pca"] = rna_pca.reindex(adata.obs_names).to_numpy()
    adata.obsm["X_protein_pca"] = protein_pca.reindex(adata.obs_names).to_numpy()
    adata.obsm[REPRESENTATION_KEY_MAP["joint_pca"]] = joint_pca.reindex(adata.obs_names).to_numpy()
    adata.obsm["protein_counts"] = raw_protein.reindex(adata.obs_names).to_numpy()
    adata.uns["protein_names"] = list(raw_protein.columns.astype(str))

    candidate_labels = adata.obs["cell_type_simple"] if "cell_type_simple" in adata.obs else adata.obs.get("leiden")
    candidates = make_candidate_target_population_table(candidate_labels)
    candidates.to_csv(tables_dir / "candidate_target_populations.csv", index=False)

    run_parameters = {
        "input": args.input,
        "output": args.output,
        "rna_pcs": args.rna_pcs,
        "protein_pcs": args.protein_pcs,
        "n_top_genes": args.n_top_genes,
        "label_key": args.label_key,
        "n_cells": int(adata.n_obs),
        "n_rna_features": int(adata.n_vars),
        "n_protein_features": int(raw_protein.shape[1]),
        "representations": list(REPRESENTATION_KEY_MAP),
    }
    write_json(run_parameters, tables_dir / "preprocess_run_parameters.json")

    output = Path(args.output)
    save_citeseq_object(adata, output)
    logging.info("Saved benchmark-ready AnnData: %s", output)
    print(output)


if __name__ == "__main__":
    main()
