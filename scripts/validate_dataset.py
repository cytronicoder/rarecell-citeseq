#!/usr/bin/env python
"""Validate a local CITE-seq dataset object."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rarecell.config import DATA_DIR, LOGS_DIR, TABLES_DIR
from rarecell.io import load_citeseq, save_citeseq_object, validate_citeseq_object
from rarecell.script_utils import setup_file_logger
from rarecell.utils import write_json


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
    parser.add_argument("--out-dir", default=str(TABLES_DIR), help="Directory for validation outputs.")
    parser.add_argument(
        "--save-loaded-object",
        nargs="?",
        const="__default__",
        default=None,
        help=(
            "Optional path for the loaded object. If provided without a value, "
            "uses data/processed/pbmc5k_raw.h5ad or .h5mu."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_file_logger(LOGS_DIR / "dataset_validation.log")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading %s", args.input)
    logging.info("Output directory: %s", out_dir)
    obj = load_citeseq(args.input)
    validation = validate_citeseq_object(obj)
    logging.info("Dataset dimensions: cells=%s genes=%s", validation.get("n_cells"), validation.get("n_genes"))

    json_path = write_json(validation, out_dir / "dataset_validation.json")
    csv_path = out_dir / "dataset_validation.csv"
    pd.DataFrame([validation]).to_csv(csv_path, index=False)

    if args.save_loaded_object is not None:
        if args.save_loaded_object == "__default__":
            suffix = ".h5mu" if validation["object_type"] == "MuData" else ".h5ad"
            save_path = DATA_DIR / "processed" / f"pbmc5k_raw{suffix}"
        else:
            save_path = Path(args.save_loaded_object)
        saved = save_citeseq_object(obj, save_path)
        logging.info("Saved loaded object: %s", saved)

    logging.info("Validation written to %s and %s", json_path, csv_path)
    print("\nCITE-seq validation summary")
    print("--------------------------")
    for key in [
        "object_type",
        "modalities",
        "n_cells",
        "n_genes",
        "protein_modality_exists",
        "n_protein_features",
        "cell_type_labels_exist",
        "cell_type_label_key",
        "barcodes_align",
    ]:
        value = validation.get(key)
        print(f"{key}: {value}")

    if not validation["protein_modality_exists"]:
        logging.warning("Validation failed: no protein/ADT modality was found.")
        raise SystemExit("Validation failed: no protein/ADT modality was found.")
    logging.info("Completed dataset validation.")


if __name__ == "__main__":
    main()
