"""Filename helpers for benchmark output naming conventions."""

from __future__ import annotations

import re
from pathlib import Path


def slugify_label(label: str) -> str:
    """Convert cell-type labels to safe lowercase snake_case filename slugs.

    Examples: 'B cell' -> 'b_cell', 'CD14+ Monocyte' -> 'cd14_monocyte',
    'NK/T cell' -> 'nk_t_cell'
    """
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(label))
    return text.strip("_").lower()


def make_output_prefix(dataset_name: str, target_label: str) -> str:
    """Return a canonical output prefix: {dataset}_{target_slug}.

    Example: make_output_prefix("pbmc5k", "B cell") -> "pbmc5k_b_cell"
    """
    return f"{slugify_label(dataset_name)}_{slugify_label(target_label)}"


def result_path(
        results_dir: Path,
        category: str,
        output_prefix: str,
        descriptor: str,
        suffix: str,
) -> Path:
    """Return a standardized result path using the double-underscore separator.

    Example:
        result_path(RESULTS_DIR, "tables", "pbmc5k_b_cell", "benchmark_results", ".csv")
        -> results/tables/pbmc5k_b_cell__benchmark_results.csv
    """
    filename = f"{output_prefix}__{descriptor}{suffix}"
    return Path(results_dir) / category / filename


def resolve_existing_output_prefix(
        tables_dir: Path,
        preferred_prefix: str | None = None,
        descriptor: str = "benchmark_results",
        suffix: str = ".csv",
) -> str:
    """Return an output prefix with an existing table for descriptor.

    The preferred prefix wins when its table exists. Otherwise, use the most
    recently modified matching table in tables_dir.
    """
    tables_path = Path(tables_dir)
    if preferred_prefix:
        preferred_path = tables_path / f"{preferred_prefix}__{descriptor}{suffix}"
        if preferred_path.exists():
            return preferred_prefix

    matches = sorted(
        tables_path.glob(f"*__{descriptor}{suffix}"),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    if matches:
        marker = f"__{descriptor}{suffix}"
        return matches[0].name[: -len(marker)]

    expected = f"{preferred_prefix}__{descriptor}{suffix}" if preferred_prefix else f"*__{descriptor}{suffix}"
    raise FileNotFoundError(
        f"Results file not found in {tables_path}: {expected}. "
        "Run notebooks/rare_cell_downsampling_benchmark.ipynb first."
    )
