#!/usr/bin/env python
"""Run the core rare-cell downsampling benchmark."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import anndata as ad
import yaml

from rarecell.benchmark import (
    INTERMEDIATE_DIR,
    LOGS_DIR,
    METRICS_DIR,
    TABLES_DIR,
    cell_counts_by_fraction,
    ensure_project_directories,
    make_metric_summary,
    run_downsampling_benchmark,
    run_random_label_control,
    save_benchmark_results,
    select_control_cell_type,
    validate_representations,
    validate_results_table,
)
from rarecell.downsampling import summarize_downsampling_grid, validate_target_label
from rarecell.io import resolve_input_file
from rarecell.utils import make_output_prefix, resolve_label_key, resolve_target_label, write_json

DEFAULT_CONFIG = ROOT / "config" / "benchmark_config.yaml"
DEFAULT_INPUTS = [
    "data/processed/pbmc5k_10x_citeseq_representations.h5ad",
    "data/processed/pbmc5k_10x_citeseq_processed.h5ad",
    "data/processed/pbmc5k_representations.h5ad",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Benchmark YAML config.")
    parser.add_argument("--input", default=None, help="Override config dataset_path.")
    parser.add_argument("--label-key", default=None, help="Override config label_column.")
    parser.add_argument("--target-label", default=None, help="Override config target_cell_type.")
    parser.add_argument("--output-prefix", default=None, help="Override output filename prefix.")
    parser.add_argument("--retain-fractions", nargs="+", type=float, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--n-neighbors", type=int, default=None)
    parser.add_argument("--representations", nargs="+", default=None)
    parser.add_argument("--skip-controls", action="store_true", help="Skip abundant-cell and random-label controls.")
    return parser.parse_args()


def _read_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _relative(path: str | Path) -> str:
    value = Path(path)
    try:
        return str(value.relative_to(ROOT))
    except ValueError:
        return str(value)


def _setup_logging(output_prefix: str) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{output_prefix}__run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return log_path


def _parameters(args: argparse.Namespace, config: dict) -> dict:
    return {
        "dataset": config.get("dataset", "citeseq"),
        "input": args.input or config.get("dataset_path"),
        "label_key": args.label_key or config.get("label_column", "leiden"),
        "target_label": args.target_label if args.target_label is not None else config.get("target_cell_type"),
        "control_label": config.get("control_cell_type"),
        "fractions": args.retain_fractions or config.get("fractions", [1.0, 0.5, 0.25, 0.1, 0.05]),
        "seeds": args.seeds or config.get("seeds", [0, 1, 2, 3, 4]),
        "representations": args.representations or config.get("representations",
                                                              ["rna_pca", "protein_pca", "joint_pca"]),
        "n_neighbors": args.n_neighbors or config.get("k_neighbors", 15),
        "run_abundant_control": bool(config.get("run_abundant_control", True)) and not args.skip_controls,
        "run_random_label_control": bool(config.get("run_random_label_control", True)) and not args.skip_controls,
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, Path | list[Path]]:
    config = _read_config(args.config)
    params = _parameters(args, config)
    input_path = resolve_input_file(params["input"], DEFAULT_INPUTS)
    adata = ad.read_h5ad(input_path)

    label_key = resolve_label_key(adata, preferred=params["label_key"])
    target_label = resolve_target_label(adata, label_key, preferred=params["target_label"])
    representations = list(params["representations"])
    fractions = [float(value) for value in params["fractions"]]
    seeds = [int(value) for value in params["seeds"]]
    n_neighbors = int(params["n_neighbors"])
    output_prefix = args.output_prefix or make_output_prefix(params["dataset"], target_label)

    ensure_project_directories()
    log_path = _setup_logging(output_prefix)
    logging.info("Input: %s", input_path)
    logging.info("Label key: %s; target label: %s", label_key, target_label)

    validate_target_label(adata, label_key, target_label)
    validate_representations(adata, representations)
    labels = adata.obs[label_key].astype(str)
    original_target_count = int((labels == str(target_label)).sum())

    grid = summarize_downsampling_grid(adata, label_key, target_label, fractions, seeds)
    grid_path = TABLES_DIR / f"{output_prefix}__downsampling_grid.csv"
    grid.to_csv(grid_path, index=False)

    results_df = run_downsampling_benchmark(
        adata,
        label_key=label_key,
        target_label=target_label,
        representation_keys=representations,
        retain_fractions=fractions,
        seeds=seeds,
        n_neighbors=n_neighbors,
        dataset=str(params["dataset"]),
        logger=logging.getLogger(__name__),
    )
    validate_results_table(results_df, representations, fractions, seeds)
    save_benchmark_results(results_df, output_prefix)
    results_csv = TABLES_DIR / f"{output_prefix}__benchmark_results.csv"
    raw_csv = METRICS_DIR / f"{output_prefix}__benchmark_raw.csv"
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(raw_csv, index=False)

    metric_summary = make_metric_summary(results_df)
    metric_summary_path = TABLES_DIR / f"{output_prefix}__metric_summary.csv"
    metric_summary.to_csv(metric_summary_path, index=False)

    counts = cell_counts_by_fraction(adata, label_key, target_label, fractions, seeds)
    counts_path = INTERMEDIATE_DIR / f"{output_prefix}__cell_counts_by_fraction.csv"
    counts.to_csv(counts_path, index=False)

    output_files: list[Path] = [log_path, grid_path, results_csv, raw_csv, metric_summary_path, counts_path]

    if params["run_abundant_control"]:
        control_label = select_control_cell_type(labels, target=str(target_label), requested=params["control_label"])
        if control_label is not None:
            control_df = run_downsampling_benchmark(
                adata,
                label_key=label_key,
                target_label=control_label,
                representation_keys=representations,
                retain_fractions=fractions,
                seeds=seeds,
                n_neighbors=n_neighbors,
                dataset=f"{params['dataset']}_abundant_control",
                logger=logging.getLogger(__name__),
            )
            control_path = METRICS_DIR / f"{output_prefix}__abundant_control_raw.csv"
            control_df.to_csv(control_path, index=False)
            output_files.append(control_path)

    if params["run_random_label_control"]:
        random_df = run_random_label_control(
            adata,
            label_key=label_key,
            target_label=target_label,
            representations=representations,
            retain_fraction=min(fractions),
            seed=seeds[0],
            n_neighbors=n_neighbors,
        )
        random_path = TABLES_DIR / f"{output_prefix}__random_label_control.csv"
        random_df.to_csv(random_path, index=False)
        output_files.append(random_path)

    run_summary = {
        "input_file": _relative(input_path),
        "label_key": label_key,
        "target_label": str(target_label),
        "output_prefix": output_prefix,
        "representations": representations,
        "retain_fractions": fractions,
        "seeds": seeds,
        "n_neighbors": n_neighbors,
        "n_cells": int(adata.n_obs),
        "target_cell_count_original": original_target_count,
        "n_cell_types": int(labels.nunique(dropna=False)),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_files": [_relative(path) for path in output_files],
    }
    summary_path = LOGS_DIR / f"{output_prefix}__run_summary.json"
    write_json(run_summary, summary_path)
    output_files.append(summary_path)

    for path in output_files:
        print(_relative(path))
    return {
        "results": results_csv,
        "metric_summary": metric_summary_path,
        "run_summary": summary_path,
        "output_files": output_files,
    }


def main() -> None:
    outputs = run_pipeline(parse_args())
    print(f"Benchmark complete: {_relative(outputs['results'])}")


if __name__ == "__main__":
    main()
