#!/usr/bin/env python
"""Import a scvi-tools totalVI tutorial dataset or local MuData file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rarecell.io import load_citeseq, save_citeseq_object, to_internal_anndata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="scvi:5k_pbmc_protein_v3_nextgem",
        help="Local .h5mu/.h5ad/.h5 input or scvi: dataset spec.",
    )
    parser.add_argument("--output", required=True, help="Output internal .h5ad path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        adata = to_internal_anndata(load_citeseq(args.input), dataset=str(args.input))
    except ImportError as exc:
        raise SystemExit(
            f"Could not import scvi/MuData input: {exc}\n"
            "Install optional dependencies such as scvi-tools, mudata, or muon, or pass a local .h5ad/.h5mu file."
        ) from exc
    save_citeseq_object(adata, args.output)
    print(f"Saved internal AnnData: {args.output}")


if __name__ == "__main__":
    main()
