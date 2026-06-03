#!/usr/bin/env python
"""Run benchmark controls: abundant-cell and random-label."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import anndata as ad
import yaml

METRICS_DIR = ROOT / "results" / "metrics"
TABLES_DIR = ROOT / "results" / "tables"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default="config/benchmark_config.yaml",
        help="Benchmark YAML config.",
    )
    p.add_argument(
        "--skip-abundant",
        action="store_true",
        help="Skip the abundant-cell control.",
    )
    p.add_argument(
        "--skip-random-label",
        action="store_true",
        help="Skip the random-label control.",
    )
    return p.parse_args()


def _read_config(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"Expected a mapping in config file: {path}")
    return cfg


def main() -> None:
    args = parse_args()
    config = _read_config(ROOT / args.config)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    input_path = ROOT / str(
        config.get("dataset_path", "data/processed/pbmc5k_10x_citeseq_representations.h5ad")
    )
    if not input_path.exists():
        print(f"ERROR: Data file not found: {input_path}", file=sys.stderr)
        print(
            "Place an .h5ad file at the configured dataset_path and run "
            "scripts/preprocess_dataset.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading data from {input_path} …")
    adata = ad.read_h5ad(input_path)

    from rarecell.utils import infer_label_column
    from rarecell.benchmark import select_control_cell_type, run_random_label_control
    from rarecell.benchmark import run_downsampling_benchmark

    label_key = config.get("label_column") or infer_label_column(adata)
    if label_key is None:
        raise SystemExit("No label column found. Set label_column in the config.")

    labels = adata.obs[label_key].astype(str)
    counts = labels.value_counts()
    target = str(config.get("target_cell_type") or counts.index[min(1, len(counts) - 1)])
    fractions = [float(f) for f in config.get("fractions", [1.0, 0.5, 0.25, 0.1, 0.05])]
    seeds = [int(s) for s in config.get("seeds", [0, 1, 2, 3, 4])]
    representations = list(config.get("representations", ["rna_pca", "protein_pca", "joint_pca"]))
    n_neighbors = int(config.get("k_neighbors", 15))

    print(f"Target cell type: {target}")

    if not args.skip_abundant and config.get("run_abundant_control", True):
        control = select_control_cell_type(
            labels, target, config.get("control_cell_type")
        )
        if control is None:
            print("No non-target cell type found; skipping abundant-cell control.")
        else:
            print(f"Running abundant-cell control with cell type: {control}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                control_results = run_downsampling_benchmark(
                    adata,
                    label_key=label_key,
                    target_label=control,
                    representation_keys=representations,
                    retain_fractions=fractions,
                    seeds=seeds,
                    n_neighbors=n_neighbors,
                    dataset=str(config.get("dataset", input_path.stem)),
                )
            raw_path = METRICS_DIR / "abundant_cell_control_raw.csv"
            control_results.to_csv(raw_path, index=False)
            print(f"  Saved raw control metrics: {raw_path}")

    if not args.skip_random_label and config.get("run_random_label_control", True):
        fraction_for_control = min(fractions)
        seed_for_control = seeds[0]
        print(
            f"Running random-label control at fraction={fraction_for_control}, seed={seed_for_control} …"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rl_df = run_random_label_control(
                adata,
                label_key=label_key,
                target_label=target,
                representations=representations,
                retain_fraction=fraction_for_control,
                seed=seed_for_control,
                n_neighbors=n_neighbors,
            )
        rl_path = TABLES_DIR / "random_label_control.csv"
        rl_df.to_csv(rl_path, index=False)
        print(f"  Saved random-label control: {rl_path}")

    if config.get("run_marker_removal_sensitivity", False):
        print("Marker-removal sensitivity analysis is enabled in config but not implemented here.")
        print("Set run_marker_removal_sensitivity: false to suppress this message.")

    print("Controls complete.")


if __name__ == "__main__":
    main()
