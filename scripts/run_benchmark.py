#!/usr/bin/env python
"""Run the rare-cell CITE-seq downsampling benchmark.

Loads a processed AnnData (.h5ad) file containing RNA, protein, and joint
representations, then runs a controlled target-cell downsampling benchmark
across a grid of retain fractions and random seeds.

Input:
    A processed AnnData file with adata.obsm keys for each requested
    representation (rna_pca, protein_pca, joint_pca).

Outputs (under results/):
    tables/{output_prefix}__benchmark_results.csv
    tables/{output_prefix}__metric_summary.csv
    tables/{output_prefix}__downsampling_grid.csv
    tables/{output_prefix}__figure_index.csv
    figures/{output_prefix}__rare_cell_f1_curve.{png,pdf}
    figures/{output_prefix}__rare_cell_recall_curve.{png,pdf}
    figures/{output_prefix}__neighborhood_purity_curve.{png,pdf}
    figures/{output_prefix}__target_cell_counts.{png,pdf}
    logs/{output_prefix}__run_summary.json
    reports/{output_prefix}__benchmark_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import anndata as ad

from rarecell.benchmark import (
    INTERMEDIATE_DIR,
    LOGS_DIR,
    METRICS_DIR,
    TABLES_DIR,
    cell_counts_by_fraction,
    ensure_project_directories,
    make_metric_summary,
    run_downsampling_benchmark,
    save_benchmark_results,
    validate_representations,
    validate_results_table,
    write_markdown_report,
)
from rarecell.downsampling import summarize_downsampling_grid, validate_target_label
from rarecell.io import resolve_input_file
from rarecell.plotting import save_all_standard_plots

DEFAULT_INPUT = "data/processed/pbmc5k_10x_citeseq_representations.h5ad"
# Fallback for older processed files produced by preprocess_data.py
_FALLBACK_INPUTS = [
    "data/processed/pbmc5k_10x_citeseq_representations.h5ad",
    "data/processed/pbmc5k_representations.h5ad",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=(
            "Path to the processed AnnData .h5ad input file. "
            "Must contain protein_counts in obsm and benchmark representations."
        ),
    )
    parser.add_argument(
        "--label-key",
        default="leiden",
        help=(
            "adata.obs column containing cell-type labels. "
            "Defaults to 'leiden' (Leiden cluster IDs from the current preprocessing pipeline). "
            "Use 'cell_type_simple' for objects built with preprocess_data.py."
        ),
    )
    parser.add_argument(
        "--target-label",
        default=None,
        help=(
            "Cell-type label of the rare target population. "
            "If omitted, the second-smallest class in label-key is used automatically."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Prefix for all output files, e.g. 'pbmc5k_14'. "
            "Outputs follow the pattern {output_prefix}__{descriptor}.{ext}. "
            "If omitted, auto-generated from dataset name and target label."
        ),
    )
    parser.add_argument(
        "--retain-fractions",
        nargs="+",
        type=float,
        default=[1.0, 0.5, 0.25, 0.1, 0.05],
        help="Target-cell retained fractions to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Random seeds for downsampling reproducibility.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="Number of nearest neighbors for kNN-based metrics.",
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        default=["rna_pca", "protein_pca", "joint_pca"],
        help="Public representation names to compare.",
    )
    return parser.parse_args()


def _relative(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _setup_logging(output_prefix: str) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{output_prefix}__run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return log_path


def _load_and_validate(args: argparse.Namespace):
    try:
        input_path = resolve_input_file(
            args.input,
            default_candidates=_FALLBACK_INPUTS,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Processed input file not found:\n"
            f"  {args.input}\n\n"
            "Expected location: data/processed/pbmc5k_10x_citeseq_representations.h5ad\n"
            "This file is created by:\n"
            "  python scripts/preprocess_data.py\n"
            "or the notebooks/baseline_representations.ipynb notebook.\n\n"
            "For a quick validation run without preprocessing:\n"
            "  python scripts/run_smoke_test.py"
        ) from exc

    logging.info("Loading AnnData: %s", input_path)
    adata = ad.read_h5ad(input_path)

    # Resolve label_key from adata if the default doesn't exist
    from rarecell.utils import resolve_label_key, resolve_target_label
    resolved_label_key = resolve_label_key(adata, preferred=args.label_key)
    if resolved_label_key != args.label_key:
        logging.info(
            "Label key '%s' not found; using '%s' instead.", args.label_key, resolved_label_key
        )
        args.label_key = resolved_label_key

    # Auto-select target label if not specified
    if args.target_label is None:
        args.target_label = resolve_target_label(adata, args.label_key, preferred=None)
        logging.info("Auto-selected target label: '%s'", args.target_label)

    # Auto-generate output prefix if not specified
    if args.output_prefix is None:
        from rarecell.utils import make_output_prefix
        dataset_name = Path(input_path).stem.replace("_representations", "").replace("_processed", "")
        args.output_prefix = make_output_prefix(dataset_name, args.target_label)
        logging.info("Auto-generated output prefix: '%s'", args.output_prefix)

    validate_target_label(adata, args.label_key, args.target_label)
    validate_representations(adata, args.representations)
    return input_path, adata


def _write_json(data: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def run_pipeline(args: argparse.Namespace) -> dict[str, Path | list[Path]]:
    ensure_project_directories()
    log_path = _setup_logging(args.output_prefix)
    input_path, adata = _load_and_validate(args)

    labels = adata.obs[args.label_key].astype(str)
    original_target_count = int((labels == str(args.target_label)).sum())
    n_cell_types = int(labels.nunique(dropna=False))

    logging.info("Saving downsampling grid summary.")
    grid = summarize_downsampling_grid(
        adata,
        args.label_key,
        args.target_label,
        args.retain_fractions,
        args.seeds,
    )
    grid_path = TABLES_DIR / f"{args.output_prefix}__downsampling_grid.csv"
    grid.to_csv(grid_path, index=False)
    print(f"Saved grid summary: {_relative(grid_path)}")

    logging.info("Running benchmark.")
    results_df = run_downsampling_benchmark(
        adata,
        label_key=args.label_key,
        target_label=args.target_label,
        representation_keys=args.representations,
        retain_fractions=args.retain_fractions,
        seeds=args.seeds,
        n_neighbors=args.n_neighbors,
    )
    validate_results_table(results_df, args.representations, args.retain_fractions, args.seeds)
    save_benchmark_results(results_df, args.output_prefix)
    results_csv = TABLES_DIR / f"{args.output_prefix}__benchmark_results.csv"
    results_parquet = METRICS_DIR / f"{args.output_prefix}__benchmark_results.parquet"
    print(f"Saved results CSV: {_relative(results_csv)}")
    if results_parquet.exists():
        print(f"Saved results parquet: {_relative(results_parquet)}")
    else:
        print("Parquet output skipped or unavailable; CSV results were saved.")

    logging.info("Saving cell-count metadata.")
    counts = cell_counts_by_fraction(
        adata,
        args.label_key,
        args.target_label,
        args.retain_fractions,
        args.seeds,
    )
    counts_path = INTERMEDIATE_DIR / f"{args.output_prefix}__cell_counts_by_fraction.csv"
    counts.to_csv(counts_path, index=False)
    print(f"Saved cell-count metadata: {_relative(counts_path)}")

    logging.info("Saving metric summary.")
    metric_summary = make_metric_summary(results_df)
    metric_summary_path = TABLES_DIR / f"{args.output_prefix}__metric_summary.csv"
    metric_summary.to_csv(metric_summary_path, index=False)
    print(f"Saved metric summary: {_relative(metric_summary_path)}")

    logging.info("Saving standard plots.")
    plot_paths = save_all_standard_plots(results_df, args.output_prefix)
    if not plot_paths:
        raise AssertionError("No figure files were created.")
    for path in plot_paths:
        print(f"Saved figure: {_relative(path)}")

    output_files = [
        grid_path,
        results_csv,
        metric_summary_path,
        counts_path,
        *plot_paths,
        log_path,
    ]
    run_summary_path = LOGS_DIR / f"{args.output_prefix}__run_summary.json"
    output_files.append(run_summary_path)
    if results_parquet.exists():
        output_files.append(results_parquet)

    report_path = write_markdown_report(
        output_prefix=args.output_prefix,
        input_file=_relative(input_path),
        label_key=args.label_key,
        target_label=args.target_label,
        representation_keys=args.representations,
        retain_fractions=args.retain_fractions,
        seeds=args.seeds,
        n_neighbors=args.n_neighbors,
        n_cells=int(adata.n_obs),
        original_target_cells=original_target_count,
        n_cell_types=n_cell_types,
        output_files=[_relative(path) for path in output_files],
        metric_summary=metric_summary,
    )
    output_files.append(report_path)
    print(f"Saved report: {_relative(report_path)}")

    run_summary = {
        "input_file": _relative(input_path),
        "label_key": args.label_key,
        "target_label": args.target_label,
        "output_prefix": args.output_prefix,
        "representations": args.representations,
        "retain_fractions": args.retain_fractions,
        "seeds": args.seeds,
        "n_neighbors": int(args.n_neighbors),
        "n_cells": int(adata.n_obs),
        "target_cell_count_original": original_target_count,
        "n_cell_types": n_cell_types,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_files": [_relative(path) for path in output_files],
    }
    _write_json(run_summary, run_summary_path)
    output_files.append(run_summary_path)
    print(f"Saved run summary: {_relative(run_summary_path)}")

    if not run_summary_path.exists():
        raise AssertionError(f"Run summary JSON was not created: {run_summary_path}.")
    if not any(path.exists() for path in plot_paths):
        raise AssertionError("At least one figure file should have been created.")

    logging.info("Benchmark complete with %d rows.", len(results_df))
    return {
        "results": results_csv,
        "metric_summary": metric_summary_path,
        "plots": plot_paths,
        "run_summary": run_summary_path,
        "report": report_path,
        "output_files": output_files,
    }


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args)
    print("Benchmark complete.")
    print(f"Main results: {_relative(outputs['results'])}")


if __name__ == "__main__":
    main()
