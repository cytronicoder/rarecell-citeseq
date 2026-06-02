#!/usr/bin/env python
"""Run a quick smoke test for the rare-cell downsampling benchmark.

Subsamples the first available processed CITE-seq dataset to at most
--max-cells cells and runs the benchmark pipeline with a reduced grid
(two fractions, two seeds) to verify the full output pipeline works.

Input:
    Automatically selects from configured candidates:
        data/processed/pbmc5k_representations.h5ad

Outputs (under results/):
    intermediate/{output_prefix}_input.h5ad
    tables/{output_prefix}__benchmark_results.csv
    tables/{output_prefix}__metric_summary.csv
    figures/{output_prefix}__rare_cell_f1_curve.{png,pdf}
    logs/{output_prefix}__run_summary.json
    reports/{output_prefix}__benchmark_report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import anndata as ad
import numpy as np

from rarecell.benchmark import INTERMEDIATE_DIR, ensure_project_directories, validate_representations
from rarecell.io_utils import resolve_input_file
from run_rare_cell_benchmark import run_pipeline
from rarecell.downsampling import validate_target_label
from rarecell.validation import resolve_label_key, resolve_target_label

SMOKE_INPUT_CANDIDATES = [
    "data/processed/pbmc5k_representations.h5ad",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to a processed .h5ad file. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--label-key",
        default="cell_type_simple",
        help="adata.obs column containing cell-type labels.",
    )
    parser.add_argument(
        "--target-label",
        default="B cell",
        help="Cell-type label of the rare target population.",
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        default=["rna_pca", "protein_pca", "joint_pca"],
        help="Public representation names to compare.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="Number of nearest neighbors for kNN-based metrics.",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=1000,
        help="Maximum cells to retain in the smoke-test subsample.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output prefix. Auto-inferred from input filename if omitted.",
    )
    return parser.parse_args()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _infer_smoke_prefix(input_path: Path, explicit_prefix: str | None) -> str:
    if explicit_prefix:
        return explicit_prefix
    display = _display_path(input_path).lower()
    if "pbmc5k" in display:
        return "smoke_test_pbmc5k"
    return "smoke_test"


def _subsample_adata(adata, max_cells: int, seed: int = 0):
    if adata.n_obs <= max_cells:
        return adata.copy()
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(np.arange(adata.n_obs), size=max_cells, replace=False))
    return adata[keep, :].copy()


def main() -> None:
    args = parse_args()
    ensure_project_directories()
    input_path = resolve_input_file(
        args.input,
        default_candidates=SMOKE_INPUT_CANDIDATES,
    )
    print("Smoke test input selected:")
    print(_display_path(input_path))
    adata = ad.read_h5ad(input_path)

    label_key = resolve_label_key(adata, preferred=args.label_key)
    target_label = resolve_target_label(adata, label_key, preferred=args.target_label)
    validate_target_label(adata, label_key, target_label)
    validate_representations(adata, args.representations)
    representations = list(args.representations)

    smoke_adata = _subsample_adata(adata, args.max_cells, seed=0)
    if target_label not in set(smoke_adata.obs[label_key].astype(str)):
        target_indices = np.flatnonzero(adata.obs[label_key].astype(str).to_numpy() == str(target_label))
        other_indices = np.flatnonzero(adata.obs[label_key].astype(str).to_numpy() != str(target_label))
        n_target_keep = min(max(2, args.max_cells // 5), target_indices.size)
        n_other_keep = min(args.max_cells - n_target_keep, other_indices.size)
        keep = np.sort(np.concatenate([target_indices[:n_target_keep], other_indices[:n_other_keep]]))
        smoke_adata = adata[keep, :].copy()

    output_prefix = _infer_smoke_prefix(input_path, args.output_prefix)
    smoke_input = INTERMEDIATE_DIR / f"{output_prefix}_input.h5ad"
    smoke_adata.write_h5ad(smoke_input)
    print(f"Smoke-test input cells: {smoke_adata.n_obs}")

    pipeline_args = SimpleNamespace(
        input=str(smoke_input.relative_to(ROOT)),
        label_key=label_key,
        target_label=target_label,
        output_prefix=output_prefix,
        retain_fractions=[1.0, 0.25],
        seeds=[0, 1],
        n_neighbors=args.n_neighbors,
        representations=representations,
    )
    run_pipeline(pipeline_args)
    print("Smoke test complete.")


if __name__ == "__main__":
    main()
