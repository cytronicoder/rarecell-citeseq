#!/usr/bin/env python
"""Compute RNA, protein, and joint baseline CITE-seq representations."""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import anndata as ad
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rarecell.config import DATA_DIR, FIGURES_DIR, LOGS_DIR, REPRESENTATION_KEY_MAP, TABLES_DIR
from rarecell.io import get_cell_labels, get_protein_matrix, get_rna_adata, load_citeseq, save_citeseq_object
from rarecell.plotting import plot_bar_counts, plot_embedding
from rarecell.preprocessing import align_cells_between_modalities, preprocess_protein, preprocess_rna
from rarecell.io import make_candidate_target_population_table
from rarecell.representations import (
    compute_joint_pca_representation,
    compute_protein_pca,
    compute_rna_pca,
    compute_umap_from_embedding,
    save_embedding,
)
from rarecell.utils import setup_file_logger
from rarecell.utils import write_json


def _cluster_counts(rna_processed) -> pd.DataFrame:
    if "leiden" not in rna_processed.obs:
        return pd.DataFrame(columns=["cluster", "n_cells", "fraction"])
    counts = rna_processed.obs["leiden"].astype(str).value_counts().sort_index()
    total = counts.sum()
    return pd.DataFrame(
        {
            "cluster": counts.index,
            "n_cells": counts.to_numpy(),
            "fraction": counts.to_numpy() / total,
        }
    )


def _rank_genes_groups_table(rna_processed, top_n: int = 20) -> pd.DataFrame:
    result = rna_processed.uns.get("rank_genes_groups")
    if result is None or "names" not in result:
        return pd.DataFrame(columns=["cluster", "rank", "gene", "score", "pval", "pval_adj", "logfoldchange"])

    groups = result["names"].dtype.names
    if groups is None:
        return pd.DataFrame(columns=["cluster", "rank", "gene", "score", "pval", "pval_adj", "logfoldchange"])

    rows = []
    optional = {
        "scores": "score",
        "pvals": "pval",
        "pvals_adj": "pval_adj",
        "logfoldchanges": "logfoldchange",
    }
    for group in groups:
        genes = result["names"][group][:top_n]
        for rank, gene in enumerate(genes, start=1):
            row = {"cluster": group, "rank": rank, "gene": gene}
            for source, target in optional.items():
                row[target] = result[source][group][rank - 1] if source in result else pd.NA
            rows.append(row)
    return pd.DataFrame(rows)


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
    parser.add_argument("--tables-dir", default=str(TABLES_DIR), help="Directory for embeddings.")
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR), help="Directory for figures.")
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "processed" / "pbmc5k_10x_citeseq_representations.h5ad"),
        help="Path to save the processed object with benchmark representations.",
    )
    parser.add_argument("--rna-pcs", type=int, default=30, help="Number of RNA PCs.")
    parser.add_argument("--protein-pcs", type=int, default=10, help="Number of protein PCs.")
    parser.add_argument("--n-top-genes", type=int, default=2000, help="Number of highly variable genes.")
    parser.add_argument("--label-key", default=None, help="Preferred .obs label column.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_file_logger(LOGS_DIR / "baseline_representations.log")
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading %s", args.input)
    logging.info("Output object: %s", args.output)
    logging.info("Output tables: %s", tables_dir)
    logging.info("Output figures: %s", figures_dir)
    obj = load_citeseq(args.input)
    preferred = [args.label_key] if args.label_key else None
    labels = get_cell_labels(obj, preferred_keys=preferred)

    rna = get_rna_adata(obj)
    protein = get_protein_matrix(obj)
    logging.info("Input dimensions: cells=%d genes=%d protein_features=%d", rna.n_obs, rna.n_vars, protein.shape[1])
    rna, protein = align_cells_between_modalities(rna, protein)
    if "counts" not in rna.layers:
        rna.layers["counts"] = rna.X.copy()
    raw_protein = protein.copy()
    if labels is not None:
        labels = labels.reindex(rna.obs_names)

    logging.info("Preprocessing RNA")
    rna_processed = preprocess_rna(rna, n_top_genes=args.n_top_genes, n_pcs=args.rna_pcs)
    logging.info("Preprocessing protein")
    protein_processed = preprocess_protein(protein)
    protein_processed = protein_processed.loc[rna_processed.obs_names].copy()
    raw_protein = raw_protein.loc[rna_processed.obs_names].copy()

    logging.info("Computing baseline representations")
    rna_pca = compute_rna_pca(rna_processed, n_components=args.rna_pcs)
    protein_pca = compute_protein_pca(protein_processed, n_components=args.protein_pcs)
    joint = compute_joint_pca_representation(rna_pca, protein_pca, scale_blocks=True)

    logging.info("Computing UMAPs")
    rna_umap = compute_umap_from_embedding(rna_pca)
    protein_umap = compute_umap_from_embedding(protein_pca)
    joint_umap = compute_umap_from_embedding(joint)

    try:
        import scanpy as sc

        if "neighbors" not in rna_processed.uns and "X_pca" in rna_processed.obsm:
            n_pcs = min(args.rna_pcs, rna_processed.obsm["X_pca"].shape[1])
            sc.pp.neighbors(rna_processed, n_neighbors=min(15, rna_processed.n_obs - 1), n_pcs=n_pcs)
        if "leiden" not in rna_processed.obs and "neighbors" in rna_processed.uns:
            sc.tl.leiden(rna_processed, random_state=0)
    except Exception as exc:
        warnings.warn(f"Leiden clustering failed; continuing without clusters. {exc}")

    label_candidates = labels.dropna().astype(str) if labels is not None else None
    if label_candidates is not None and not label_candidates.empty:
        candidate_labels = label_candidates
    elif "leiden" in rna_processed.obs:
        candidate_labels = "cluster_" + rna_processed.obs["leiden"].astype(str)
        candidate_labels.name = "cell_type_candidate"
    else:
        candidate_labels = None

    cluster_counts = _cluster_counts(rna_processed)
    cluster_counts.to_csv(tables_dir / "cluster_counts.csv", index=False)
    candidates = make_candidate_target_population_table(candidate_labels)
    candidates.to_csv(tables_dir / "candidate_target_populations.csv", index=False)

    marker_table = pd.DataFrame()
    if "leiden" in rna_processed.obs:
        try:
            import scanpy as sc

            sc.tl.rank_genes_groups(rna_processed, groupby="leiden", method="wilcoxon")
            marker_table = _rank_genes_groups_table(rna_processed)
            marker_table.to_csv(tables_dir / "rna_cluster_markers.csv", index=False)
        except Exception as exc:
            warnings.warn(f"Marker gene export failed; continuing without marker table. {exc}")

    save_embedding(rna_pca, tables_dir / "rna_pca.csv")
    save_embedding(protein_pca, tables_dir / "protein_pca.csv")
    save_embedding(joint, tables_dir / "joint_pca.csv")
    save_embedding(rna_umap, tables_dir / "rna_umap.csv")
    save_embedding(protein_umap, tables_dir / "protein_umap.csv")
    save_embedding(joint_umap, tables_dir / "joint_umap.csv")

    if labels is not None:
        rna_processed.obs["cell_type_simple"] = labels.reindex(rna_processed.obs_names)
    rna_processed.obsm["X_rna_pca"] = rna_pca.reindex(rna_processed.obs_names).to_numpy()
    rna_processed.obsm["X_protein_pca"] = protein_pca.reindex(rna_processed.obs_names).to_numpy()
    rna_processed.obsm[REPRESENTATION_KEY_MAP["joint_pca"]] = joint.reindex(rna_processed.obs_names).to_numpy()
    rna_processed.obsm["X_rna_umap"] = rna_umap.reindex(rna_processed.obs_names).to_numpy()
    rna_processed.obsm["X_protein_umap"] = protein_umap.reindex(rna_processed.obs_names).to_numpy()
    rna_processed.obsm["X_joint_umap"] = joint_umap.reindex(rna_processed.obs_names).to_numpy()
    rna_processed.obsm["protein_counts"] = raw_protein.reindex(rna_processed.obs_names).to_numpy()
    rna_processed.uns["protein_names"] = list(raw_protein.columns.astype(str))

    color_labels = labels if labels is not None else candidate_labels
    plot_embedding(rna_umap, color_labels, "RNA-only PCA representation", figures_dir / "rna_umap.png")
    plot_embedding(protein_umap, color_labels, "Protein-only PCA representation", figures_dir / "protein_umap.png")
    plot_embedding(joint_umap, color_labels, "Joint RNA–protein representation", figures_dir / "joint_umap.png")
    if labels is not None:
        plot_bar_counts(
            labels,
            figures_dir / "cell_label_counts.png",
            "Cell counts by annotated cell type",
            "Cell type",
            "Number of cells",
        )
    if "leiden" in rna_processed.obs:
        plot_bar_counts(
            rna_processed.obs["leiden"].astype(str),
            figures_dir / "cluster_counts.png",
            "Cell counts by cluster",
            "Cluster",
            "Number of cells",
        )

    run_parameters = {
        "input": args.input,
        "output": args.output,
        "tables_dir": str(tables_dir),
        "figures_dir": str(figures_dir),
        "rna_pcs": args.rna_pcs,
        "protein_pcs": args.protein_pcs,
        "n_top_genes": args.n_top_genes,
        "label_key": args.label_key,
        "n_aligned_cells": int(rna_processed.n_obs),
        "n_rna_features": int(rna_processed.n_vars),
        "n_protein_features": int(protein_processed.shape[1]),
        "leiden_succeeded": bool("leiden" in rna_processed.obs),
        "marker_export_succeeded": bool(not marker_table.empty),
    }
    write_json(run_parameters, tables_dir / "run_parameters.json")

    output = Path(args.output)
    if output.suffix.lower() == ".h5mu":
        import mudata as md

        prot = ad.AnnData(
            protein_processed.to_numpy(),
            obs=pd.DataFrame(index=protein_processed.index),
            var=pd.DataFrame(index=protein_processed.columns),
        )
        prot.layers["counts"] = raw_protein.reindex(protein_processed.index).to_numpy()
        mdata = md.MuData({"rna": rna_processed, "prot": prot})
        save_citeseq_object(mdata, output)
    else:
        save_citeseq_object(rna_processed, output)
    logging.info("Saved processed object: %s", output)
    logging.info("Completed baseline representation generation.")


if __name__ == "__main__":
    main()
