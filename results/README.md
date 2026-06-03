### Results

This directory contains all outputs from the rare-cell CITE-seq downsampling benchmark.

#### Directory layout

```
results/
  metrics/        Raw per-condition benchmark metrics
  tables/         Aggregated summary tables and control outputs
  figures/        Publication-quality figures
  intermediate/   Intermediate artifacts (AnnData snapshots, cell-count grids)
  logs/           Script run logs
  reports/        Auto-generated per-run Markdown reports
```

#### Figures

| File                                                       | Description                                                           |
|------------------------------------------------------------|-----------------------------------------------------------------------|
| `figures/rare_cell_f1_curve.png`                           | Rare-cell F1 score vs. retained fraction, one line per representation |
| `figures/rare_cell_recall_curve_initial.png`               | Rare-cell recall vs. retained fraction                                |
| `figures/neighborhood_purity_curve.png`                    | Target-cell neighborhood purity vs. retained fraction                 |
| `figures/control_abundant_cell_downsampling.png`           | Target vs. abundant-cell control comparison (F1)                      |
| `figures/representation_umap_comparison_rare_fraction.png` | UMAP at severe rarity (fraction 0.05), colored by target vs. other    |
| `figures/downsampling_cell_counts.png`                     | Cell counts per fraction (target and control)                         |
| `figures/pbmc5k_14__*.png`                                 | Per-run figures from the pbmc5k run 14 benchmark                      |
| `figures/marker_auc_by_fraction.png`                       | Mean marker ROC-AUC by retained fraction                              |
| `figures/target_marker_expression_by_fraction.png`         | Mean target-cell marker expression by retained fraction               |
| `figures/target_neighbor_composition_heatmap.png`          | Neighbor-label composition around target cells                        |
| `figures/target_error_absorption_heatmap.png`              | Labels absorbing target-cell prediction errors                        |
| `figures/marker_vs_recovery_scatter.png`                   | Marker strength versus recovery                                       |
| `figures/multi_target_f1_curve.png`                        | Multi-target F1 curves                                                |
| `figures/multi_target_neighborhood_purity_curve.png`       | Multi-target neighborhood-purity curves                               |
| `figures/cross_target_method_ranking.png`                  | Cross-target method ranking                                           |
| `figures/final_figure_1_study_design.png`                  | Study-design schematic                                                |
| `figures/final_figure_2_main_benchmark.png`                | Main benchmark F1 figure                                              |
| `figures/final_figure_3_failure_modes.png`                 | Failure-mode interpretation figure                                    |

All curves use x-axis = retained target-cell fraction (left = severe rarity, right = full retention).
Error bars or shaded regions represent variability across seeds.

#### Tables

| File                                           | Description                                                                                         |
|------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| `tables/benchmark_summary.csv`                 | Seed-aggregated metrics: mean, std, n_seeds, n_valid per (target, representation, fraction, metric) |
| `tables/best_method_by_fraction.csv`           | Best representation per (target, fraction, metric) with delta to second-best                        |
| `tables/abundant_cell_control_summary.csv`     | Same schema as benchmark_summary, for the abundant-cell control cell type                           |
| `tables/random_label_control.csv`              | Random-label permutation control results                                                            |
| `tables/target_cell_error_analysis.csv`        | Which non-target labels most often absorb target cells under kNN prediction                         |
| `tables/target_neighbor_composition.csv`       | Distribution of neighbor labels for target cells at each fraction and seed                          |
| `tables/marker_gene_summary.csv`               | RNA marker availability, AUC, target mean, rest mean, and log2 ratio                                |
| `tables/marker_protein_summary.csv`            | Protein marker availability, AUC, target mean, rest mean, and log2 ratio                            |
| `tables/marker_auc_by_fraction.csv`            | Marker AUC and expression by target, representation, fraction, seed, modality, and marker           |
| `tables/biological_interpretation_summary.csv` | Marker, recovery, purity, and conservative failure-mode summary                                     |
| `tables/multi_target_benchmark_summary.csv`    | Multi-target seed-aggregated metric summary                                                         |
| `tables/cross_target_method_ranking.csv`       | Average rank, best count, and worst count across target conditions                                  |
| `tables/final_results_table.csv`               | Compact final results table for manuscript-style reporting                                          |
| `tables/dataset_summary.csv`                   | Dataset-level metadata (n_cells, n_genes, n_protein_features)                                       |
| `tables/cell_type_counts.csv`                  | Per-cell-type counts in the full dataset                                                            |

#### Metrics

| Raw metric          | Column name           | Interpretation                                                              |
|---------------------|-----------------------|-----------------------------------------------------------------------------|
| Precision           | `precision`           | Fraction of kNN-predicted target calls that are truly target                |
| Recall              | `recall`              | Fraction of true target cells recovered by kNN prediction                   |
| F1 score            | `f1`                  | Harmonic mean of precision and recall                                       |
| Neighborhood purity | `neighborhood_purity` | Mean fraction of k nearest neighbors of a target cell that are also target  |
| Target silhouette   | `silhouette_target`   | Mean silhouette score for target cells vs. all other cells in the embedding |

kNN uses k=15 and excludes the cell itself (self-exclusion applied via NearestNeighbors with n_neighbors=k+1).

#### Metrics raw CSV

`metrics/rare_cell_benchmark_raw.csv` — one row per (dataset, target, fraction, seed, representation).
`metrics/abundant_cell_control_raw.csv` — same schema for the abundant-cell control.
`metrics/multi_target_benchmark_raw.csv` — one row per target, fraction, seed, and representation for the multi-target
benchmark.

#### Reports

| File                                      | Description                                                                                                |
|-------------------------------------------|------------------------------------------------------------------------------------------------------------|
| `reports/biological_interpretation.md`    | Marker availability, marker strength, recovery, failure modes, conservative interpretation, and next steps |
| `reports/final_results_interpretation.md` | Main question, design, results, cross-target comparison, limitations, and candidate abstract paragraph     |

#### Regenerating outputs

`scripts/` is the canonical reproducible pipeline. `notebooks/` contains interactive
companions for inspection and review that call the same scripts — they are not required
for regeneration. Run scripts from the project root with the conda environment active:

```bash
python scripts/run_downsampling_benchmark.py --config config/benchmark_config.yaml
python scripts/run_controls.py --config config/benchmark_config.yaml
python scripts/generate_summary_tables.py
python scripts/generate_benchmark_figures.py
python scripts/run_marker_analysis.py --config config/benchmark_config.yaml \
    --marker-config config/marker_config.yaml
python scripts/run_error_analysis.py --config config/benchmark_config.yaml
python scripts/run_multi_target_benchmark.py --config config/multi_target_config.yaml
python scripts/generate_final_figures.py --config config/multi_target_config.yaml
python scripts/write_interpretation_reports.py --config config/multi_target_config.yaml
```

The scripts are idempotent — re-running overwrites existing outputs with fresh results.

#### Companion notebooks

Each notebook mirrors one or more scripts and imports the same reusable functions from
`src/rarecell/`. They delegate execution to the scripts via `subprocess.run()` so the
canonical code path is exercised even during interactive review.

| Notebook                                           | Mirrors                                                                         |
|----------------------------------------------------|---------------------------------------------------------------------------------|
| `notebooks/data_loading_and_qc.ipynb`              | `scripts/make_qc_summary.py`                                                    |
| `notebooks/baseline_representations.ipynb`         | `scripts/make_baseline_representations.py`                                      |
| `notebooks/rare_cell_downsampling_benchmark.ipynb` | `scripts/run_downsampling_benchmark.py`                                         |
| `notebooks/benchmark_results_and_figures.ipynb`    | `scripts/generate_benchmark_figures.py`                                         |
| `notebooks/controls.ipynb`                         | `scripts/run_controls.py` + `scripts/generate_summary_tables.py`                |
| `notebooks/marker_analysis.ipynb`                  | `scripts/run_marker_analysis.py`                                                |
| `notebooks/error_analysis.ipynb`                   | `scripts/run_error_analysis.py`                                                 |
| `notebooks/multi_target_benchmark.ipynb`           | `scripts/run_multi_target_benchmark.py`                                         |
| `notebooks/final_figures_and_interpretation.ipynb` | `scripts/generate_final_figures.py` + `scripts/write_interpretation_reports.py` |

Notebooks that require data files (`baseline_representations.ipynb`,
`rare_cell_downsampling_benchmark.ipynb`, `controls.ipynb`, `marker_analysis.ipynb`,
`error_analysis.ipynb`, `multi_target_benchmark.ipynb`) cannot be executed unless the
processed `.h5ad` is present at the path configured in `config/benchmark_config.yaml`
(default: `data/processed/pbmc5k_10x_citeseq_representations.h5ad`).
