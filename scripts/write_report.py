#!/usr/bin/env python
"""Write or update a compact benchmark mini-report with current results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import yaml

TABLES_DIR = ROOT / "results" / "tables"
METRICS_DIR = ROOT / "results" / "metrics"
MANUSCRIPT_DIR = ROOT / "manuscript"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default="config/benchmark_config.yaml",
        help="Benchmark YAML config.",
    )
    p.add_argument(
        "--out",
        default=str(MANUSCRIPT_DIR / "rare_cell_benchmark_mini_report.md"),
        help="Output Markdown file path.",
    )
    return p.parse_args()


def _read_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _fmt(value, decimals: int = 3) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _build_results_block(summary_path: Path, raw_path: Path) -> str:
    lines: list[str] = []

    if not summary_path.exists():
        lines.append("Summary table not yet generated. Run `scripts/generate_summary_tables.py` first.")
        return "\n".join(lines)

    summary = pd.read_csv(summary_path)

    if "metric" in summary.columns:
        f1_rows = summary[summary["metric"] == "f1"].copy()
        if not f1_rows.empty:
            lines.append("### F1 score by representation and fraction\n")
            lines.append("| target_cell_type | representation | fraction | mean_F1 | std_F1 | n_seeds |")
            lines.append("|---|---|---|---|---|---|")
            for _, row in f1_rows.sort_values(["target_cell_type", "representation", "fraction"]).iterrows():
                lines.append(
                    f"| {row['target_cell_type']} | {row['representation']} "
                    f"| {_fmt(row['fraction'])} | {_fmt(row['mean'])} | {_fmt(row['std'])} "
                    f"| {int(row.get('n_seeds', 0))} |"
                )
        recall_rows = summary[summary["metric"] == "recall"].copy()
        if not recall_rows.empty:
            lines.append("\n### Recall by representation and fraction\n")
            lines.append("| target_cell_type | representation | fraction | mean_recall | std_recall | n_seeds |")
            lines.append("|---|---|---|---|---|---|")
            for _, row in recall_rows.sort_values(["target_cell_type", "representation", "fraction"]).iterrows():
                lines.append(
                    f"| {row['target_cell_type']} | {row['representation']} "
                    f"| {_fmt(row['fraction'])} | {_fmt(row['mean'])} | {_fmt(row['std'])} "
                    f"| {int(row.get('n_seeds', 0))} |"
                )
    else:
        lines.append("*(Summary table uses legacy wide format — run generate_summary_tables.py to update.)*")

    best_path = TABLES_DIR / "best_method_by_fraction.csv"
    if best_path.exists():
        best = pd.read_csv(best_path)
        f1_best = best[best["metric"] == "f1"].sort_values(["fraction", "best_mean"], ascending=[True, False])
        if not f1_best.empty:
            lines.append("\n### Best representation by fraction (F1)\n")
            lines.append("| fraction | best_representation | best_mean_F1 | second_best | delta |")
            lines.append("|---|---|---|---|---|")
            for _, row in f1_best.iterrows():
                lines.append(
                    f"| {_fmt(row['fraction'])} | {row['best_representation']} "
                    f"| {_fmt(row['best_mean'])} | {row.get('second_best_representation', '')} "
                    f"| {_fmt(row.get('delta', float('nan')))} |"
                )

    return "\n".join(lines)


def _build_control_block() -> str:
    lines: list[str] = []

    rl_path = TABLES_DIR / "random_label_control.csv"
    if rl_path.exists():
        rl = pd.read_csv(rl_path)
        lines.append("**Random-label control** (permuted labels destroy geometry-label relationship):\n")
        cols = [c for c in ["representation", "fraction", "seed", "precision", "recall", "f1", "neighborhood_purity"]
                if c in rl.columns]
        if not rl.empty:
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("|" + "|".join(["---"] * len(cols)) + "|")
            for _, row in rl[cols].iterrows():
                def _fmtval(v):
                    try:
                        return f"{float(v):.3f}"
                    except Exception:
                        return str(v)

                lines.append("| " + " | ".join(_fmtval(row[c]) for c in cols) + " |")
        lines.append(
            "\n*Expected result: random-label control should give near-zero F1 and "
            "near-random neighborhood purity.*"
        )
    else:
        lines.append(
            "Random-label control not yet generated. "
            "Run `scripts/run_controls.py` to generate `results/tables/random_label_control.csv`."
        )

    lines.append("")
    ctrl_summary_path = TABLES_DIR / "abundant_cell_control_summary.csv"
    if ctrl_summary_path.exists():
        lines.append(
            "**Abundant-cell control** summary saved to "
            "`results/tables/abundant_cell_control_summary.csv`. "
            "Raw metrics at `results/metrics/abundant_cell_control_raw.csv`."
        )
    else:
        lines.append(
            "Abundant-cell control summary not yet generated. "
            "Run `scripts/run_controls.py` then `scripts/generate_summary_tables.py`."
        )

    return "\n".join(lines)


def _build_error_analysis_block() -> str:
    lines: list[str] = []
    err_path = TABLES_DIR / "target_cell_error_analysis.csv"
    if err_path.exists():
        err = pd.read_csv(err_path)
        if not err.empty:
            top = err.sort_values("count", ascending=False).head(5)
            lines.append("Top confusion partners for target cells (highest error count):\n")
            lines.append("| representation | fraction | confused_with | count | fraction_of_target_errors |")
            lines.append("|---|---|---|---|---|")
            for _, row in top.iterrows():
                lines.append(
                    f"| {row.get('representation', '')} | {_fmt(row.get('fraction', float('nan')))} "
                    f"| {row.get('confused_with', '')} | {int(row.get('count', 0))} "
                    f"| {_fmt(row.get('fraction_of_target_errors', float('nan')))} |"
                )
        else:
            lines.append(
                "Error analysis table is empty (no prediction errors observed at evaluated conditions)."
            )
    else:
        lines.append(
            "Error analysis not yet generated. "
            "Run `scripts/generate_summary_tables.py --run-error-analysis` (requires data)."
        )

    neighbor_path = TABLES_DIR / "target_neighbor_composition.csv"
    if neighbor_path.exists():
        lines.append(
            "\nNeighbor composition table saved to `results/tables/target_neighbor_composition.csv`."
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = _read_config(ROOT / args.config)
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    out = Path(args.out)
    raw_path = METRICS_DIR / "rare_cell_benchmark_raw.csv"
    summary_path = TABLES_DIR / "benchmark_summary.csv"
    data_available = raw_path.exists()

    dataset = config.get("dataset", "pbmc5k_10x_citeseq")
    dataset_path = config.get("dataset_path", "data/processed/pbmc5k_10x_citeseq_representations.h5ad")
    fractions = config.get("fractions", [1.0, 0.5, 0.25, 0.1, 0.05])
    seeds = config.get("seeds", [0, 1, 2, 3, 4])
    representations = config.get("representations", ["rna_pca", "protein_pca", "joint_pca"])

    target = "TODO"
    control = "TODO"
    n_cells = "TODO"
    n_cell_types = "TODO"

    if raw_path.exists():
        raw = pd.read_csv(raw_path, nrows=1)
        if "target_cell_type" in raw.columns:
            target = str(raw["target_cell_type"].iloc[0])
        if "n_cells" in raw.columns:
            n_cells = str(int(raw["n_cells"].iloc[0]))

    ctrl_raw = METRICS_DIR / "abundant_cell_control_raw.csv"
    if not ctrl_raw.exists():
        ctrl_raw = METRICS_DIR / "control_abundant_cell_downsampling.csv"
    if ctrl_raw.exists():
        ctrl_first = pd.read_csv(ctrl_raw, nrows=1)
        for col in ["target_cell_type", "target_label"]:
            if col in ctrl_first.columns:
                control = str(ctrl_first[col].iloc[0])
                break

    cell_counts_path = TABLES_DIR / "cell_type_counts.csv"
    if cell_counts_path.exists():
        n_cell_types = str(len(pd.read_csv(cell_counts_path)))

    if not data_available:
        content = f"""# Rare-cell CITE-seq Benchmark Mini-Report

## Status

Real benchmark results were not generated because the raw results CSV is missing.

Run the full pipeline:

```bash
python scripts/import_data.py
python scripts/preprocess_data.py
python scripts/run_benchmark.py --config config/benchmark_config.yaml
python scripts/run_controls.py --config config/benchmark_config.yaml
python scripts/generate_summary_tables.py
python scripts/generate_figures.py
python scripts/write_report.py
```

## Required Data

Place a preprocessed CITE-seq AnnData file at `{dataset_path}` with:
- RNA counts in `adata.X`
- Protein counts in `adata.obsm["protein_counts"]`
- Cell-type labels in `adata.obs`

## TODO

- [ ] Import public 10x PBMC CITE-seq data
- [ ] Preprocess RNA and protein
- [ ] Run benchmark and controls
- [ ] Generate figures and mini-report with real numbers
"""
    else:
        results_block = _build_results_block(summary_path, raw_path)
        control_block = _build_control_block()
        error_block = _build_error_analysis_block()

        content = f"""# Rare-cell CITE-seq Benchmark Mini-Report

## Project Question

Do RNA-only, protein-only, and simple RNA+protein representations preserve a target cell type
after artificial rarity is introduced through controlled downsampling?

## Dataset Used

Dataset: `{dataset}`. Input path: `{dataset_path}`.
Total cells: {n_cells}. Cell types: {n_cell_types}.

## Preprocessing Summary

RNA counts were normalized (scran-style or library-size), log-transformed, filtered to
highly-variable genes, scaled, and embedded with PCA (50 components).
Protein counts were log-transformed, standardized, and embedded with PCA (20 components).
Representations were **recomputed from scratch** after each downsampling condition to avoid
information leakage from the full dataset.

## Target Cell Type

Target: leiden cluster `{target}` (selected as a minority population).
Abundant-cell control: leiden cluster `{control}`.

## Downsampling Design

Only target cells were downsampled. Non-target cells were held fixed.
Retained fractions: `{fractions}`.
Seeds: `{seeds}` (independent random draws per fraction).

## Representations Compared

- `rna_pca`: RNA-only PCA (50 components on log-normalized, HVG-filtered counts)
- `protein_pca`: Protein-only PCA (20 components on CLR-transformed ADT)
- `joint_pca`: Concatenation of scaled RNA-PCA and protein-PCA blocks

Representations were recomputed inside each downsampling condition.

## Metrics

- **Rare-cell precision**: Fraction of kNN-predicted target cells that are truly target
- **Rare-cell recall**: Fraction of true target cells recovered by kNN prediction
- **Rare-cell F1**: Harmonic mean of precision and recall
- **Neighborhood purity**: Mean fraction of true target-cell neighbors that are also target cells
- **Target silhouette**: Mean silhouette score for target vs. non-target cells in the embedding

kNN uses k={config.get("k_neighbors", 15)} neighbors; self-exclusion is applied.

## Main Benchmark Results

{results_block}

## Control Results

{control_block}

## Error Analysis Summary

{error_block}

## Limitations

- This benchmark uses simple linear (PCA-style) representations only. Advanced methods
  (totalVI, WNN, MOFA+) are not included.
- The benchmark is CPU-only and does not train deep models.
- Target cell type is defined by Leiden clustering (leiden cluster `{target}`), not
  validated biological cell-type annotation.
- With very few target cells (for example, fraction <= 0.05), some metric estimates become unstable
  (n_valid may drop below n_seeds due to NaN propagation).
- Confusion and neighbor composition analyses require rerunning representations on each
  downsampling condition, which is computationally intensive.

## Next Steps

1. Validate on additional public CITE-seq datasets.
2. Add confirmed biological cell-type annotations for more interpretable target selection.
3. Evaluate marker discrimination (ROC-AUC) for the chosen target population.
4. Consider extending to additional representations (e.g., totalVI latent space) in a
   clearly labeled optional extension.

## Output Files

| File | Description |
|---|---|
| `results/metrics/rare_cell_benchmark_raw.csv` | Per-condition raw metrics |
| `results/metrics/abundant_cell_control_raw.csv` | Abundant-cell control raw metrics |
| `results/tables/benchmark_summary.csv` | Seed-aggregated metric summary |
| `results/tables/best_method_by_fraction.csv` | Best representation per fraction/metric |
| `results/tables/random_label_control.csv` | Random-label control results |
| `results/tables/abundant_cell_control_summary.csv` | Abundant-cell control summary |
| `results/tables/target_cell_error_analysis.csv` | Confusion partner analysis |
| `results/tables/target_neighbor_composition.csv` | Neighbor label distribution |
| `results/figures/rare_cell_f1_curve.png` | F1 vs. retained fraction |
| `results/figures/rare_cell_recall_curve_initial.png` | Recall vs. retained fraction |
| `results/figures/neighborhood_purity_curve.png` | Purity vs. retained fraction |
| `results/figures/control_abundant_cell_downsampling.png` | Target vs. control comparison |
| `results/figures/representation_umap_comparison_rare_fraction.png` | UMAP at severe rarity |
| `results/figures/downsampling_cell_counts.png` | Cell counts per fraction |
"""

    out.write_text(content, encoding="utf-8")
    print(f"Mini-report written to {out}")


if __name__ == "__main__":
    main()
