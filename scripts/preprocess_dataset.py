#!/usr/bin/env python
"""Preprocess an internal CITE-seq AnnData file for baseline representations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rarecell.io import load_citeseq, save_citeseq_object, to_internal_anndata
from rarecell.preprocessing import preprocess_rna


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input .h5ad/.h5mu/10x file in CITE-seq format.")
    parser.add_argument("--output", required=True, help="Output preprocessed .h5ad path.")
    parser.add_argument("--n-top-genes", type=int, default=3000, help="Number of highly variable genes.")
    parser.add_argument("--rna-pcs", type=int, default=50, help="Number of RNA PCs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists() and not str(args.input).startswith(("scvi:", "10x:")):
        raise SystemExit(
            f"Input file not found: {input_path}. Run an import script first or pass an existing CITE-seq input."
        )
    raw = to_internal_anndata(load_citeseq(args.input), dataset=input_path.stem if input_path.exists() else args.input)
    protein_counts = raw.obsm["protein_counts"].copy()
    protein_names = list(raw.uns.get("protein_names", []))
    processed = preprocess_rna(raw, n_top_genes=args.n_top_genes, n_pcs=args.rna_pcs)
    raw_index = list(raw.obs_names)
    keep_positions = [raw_index.index(cell) for cell in processed.obs_names]
    processed.obsm["protein_counts"] = protein_counts[keep_positions, :]
    processed.uns["protein_names"] = protein_names
    save_citeseq_object(processed, args.output)
    print(f"Saved preprocessed AnnData: {args.output}")


if __name__ == "__main__":
    main()
