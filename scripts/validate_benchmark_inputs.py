#!/usr/bin/env python
"""Validate the processed AnnData object required by the benchmark pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rarecell.config import DATA_DIR, REPRESENTATION_KEY_MAP
from rarecell.io import load_anndata
from rarecell.script_utils import resolve_representations, setup_file_logger
from rarecell.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DATA_DIR / "processed" / "pbmc5k_representations.h5ad"))
    parser.add_argument("--label-key", default="cell_type_simple")
    parser.add_argument("--target-label", default=None)
    parser.add_argument(
        "--representations",
        nargs="+",
        default=list(REPRESENTATION_KEY_MAP),
        help="Public representation names required for the benchmark.",
    )
    parser.add_argument("--min-target-cells", type=int, default=100)
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    return parser.parse_args()


def _record(rows: list[dict], check: str, passed: bool, message: str, value="") -> None:
    rows.append(
        {
            "check": check,
            "status": "pass" if passed else "fail",
            "message": message,
            "value": value,
        }
    )


def validate_benchmark_inputs(
        input_path: str | Path,
        label_key: str | None = None,
        target_label: str | None = None,
        representations: list[str] | None = None,
        min_target_cells: int = 100,
        output_dir: str | Path = ROOT / "results",
) -> dict:
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    logs_dir = output_dir / "logs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    input_path = Path(input_path)
    _record(rows, "file_exists", input_path.exists(), f"Input path: {input_path}", str(input_path))

    adata = None
    if input_path.exists():
        try:
            adata = load_anndata(input_path)
            _record(rows, "load_anndata", True, "Loaded input as AnnData.", f"{adata.n_obs} x {adata.n_vars}")
        except Exception as exc:
            _record(rows, "load_anndata", False, f"Could not load AnnData: {exc}", type(exc).__name__)

    detected_label_key = None
    available_embeddings: list[str] = []
    if adata is not None:
        _record(rows, "rna_matrix", adata.X is not None, "adata.X is present.", str(getattr(adata.X, "shape", "")))
        _record(rows, "counts_layer", "counts" in adata.layers, "adata.layers['counts'] exists.",
                "counts" in adata.layers)
        _record(
            rows,
            "protein_counts",
            "protein_counts" in adata.obsm,
            "adata.obsm['protein_counts'] exists.",
            "protein_counts" in adata.obsm,
        )
        _record(
            rows,
            "protein_names",
            "protein_names" in adata.uns,
            "adata.uns['protein_names'] exists.",
            "protein_names" in adata.uns,
        )
        requested_representations = representations or list(REPRESENTATION_KEY_MAP)
        try:
            resolved_representations = resolve_representations(adata, requested_representations)
        except KeyError:
            resolved_representations = []
        resolved_names = {name for name, _ in resolved_representations}
        for standard_name in requested_representations:
            key = REPRESENTATION_KEY_MAP.get(standard_name, standard_name)
            display_name = standard_name if standard_name in REPRESENTATION_KEY_MAP else key
            exists = key in adata.obsm
            resolved = display_name in resolved_names or exists
            _record(rows, f"embedding_{display_name}", resolved,
                    f"Representation '{display_name}' resolves to adata.obsm['{key}'].", key)
            if resolved:
                available_embeddings.append(display_name)
        available_labels = list(adata.obs.columns)
        if label_key in adata.obs:
            detected_label_key = label_key
            _record(rows, "label_key", True, f"Using label key '{detected_label_key}'.", detected_label_key)
        else:
            _record(
                rows,
                "label_key",
                False,
                f"Label key '{label_key}' is not present. Available columns: {available_labels or '<none>'}.",
                label_key or "",
            )
        if detected_label_key and target_label is not None:
            labels = adata.obs[detected_label_key].astype(str)
            count = int((labels == str(target_label)).sum())
            _record(
                rows,
                "target_label",
                count >= min_target_cells,
                f"Target '{target_label}' has {count} cells; required at least {min_target_cells}.",
                count,
            )

    table = pd.DataFrame(rows, columns=["check", "status", "message", "value"])
    csv_path = tables_dir / "benchmark_input_validation.csv"
    json_path = tables_dir / "benchmark_input_validation.json"
    table.to_csv(csv_path, index=False)
    overall_pass = bool(not table.empty and (table["status"] == "pass").all())
    payload = {
        "input_path": str(input_path),
        "dataset_shape": list(adata.shape) if adata is not None else None,
        "detected_label_key": detected_label_key,
        "available_embeddings": available_embeddings,
        "checks": table.to_dict(orient="records"),
        "overall_status": "pass" if overall_pass else "fail",
    }
    write_json(payload, json_path)
    return payload


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    setup_file_logger(output_dir / "logs" / "benchmark_input_validation.log")
    logging.info("Input path: %s", args.input)
    logging.info(
        "Output paths: %s and %s",
        output_dir / "tables" / "benchmark_input_validation.csv",
        output_dir / "logs" / "benchmark_input_validation.log",
    )
    result = validate_benchmark_inputs(
        args.input,
        label_key=args.label_key,
        target_label=args.target_label,
        representations=args.representations,
        min_target_cells=args.min_target_cells,
        output_dir=args.output_dir,
    )
    logging.info("Dataset shape: %s", result.get("dataset_shape"))
    logging.info("Completed with status: %s", result["overall_status"])
    if result["overall_status"] != "pass":
        raise SystemExit("Benchmark input validation failed.")


if __name__ == "__main__":
    main()
