#!/usr/bin/env python
"""Import a 10x CITE-seq HDF5 file into the internal AnnData layout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rarecell.io import load_citeseq, save_citeseq_object, to_internal_anndata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="10x filtered_feature_bc_matrix.h5 path.")
    parser.add_argument("--output", required=True, help="Output internal .h5ad path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(
            f"Input file not found: {input_path}\n"
            "Download the 10x PBMC CITE-seq filtered_feature_bc_matrix.h5 file and place it at this path."
        )
    adata = to_internal_anndata(load_citeseq(input_path), dataset=input_path.parent.name)
    save_citeseq_object(adata, args.output)
    print(f"Saved internal AnnData: {args.output}")


if __name__ == "__main__":
    main()
