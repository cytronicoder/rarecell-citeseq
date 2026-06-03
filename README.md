### Benchmarking rare-cell recovery in RNA-protein single-cell representations

Single-cell multimodal technologies such as CITE-seq measure both transcriptomes and surface-protein abundance in the same cells. This makes them powerful for distinguishing immune cell types and fine-grained cellular states.

However, multimodal integration methods are usually evaluated on broad objectives such as clustering, batch correction, modality prediction, or global cell-type separation. Large benchmarks already compare many single-cell multimodal integration algorithms, including broad studies across RNA, protein, ATAC, DNA, and spatial modalities.

In this project, we determine whether common multimodal representations still preserve rare cell types or instead absorb them into nearby abundant populations. This question is especially important because rare immune populations and cellular states can be biologically meaningful, disease-associated, or central to downstream discovery, even when they represent only a small fraction of the dataset.

#### Datasets

We use two CITE-seq resources for the main benchmark and optional multimodal extension:

1. [10x PBMC 10k CITE-seq dataset](https://www.10xgenomics.com/datasets/10-k-peripheral-blood-mononuclear-cells-pbm-cs-from-a-healthy-donor-single-indexed-3-1-standard-4-0-0)
2. [scvi-tools PBMC5k + PBMC10k totalVI tutorial dataset](https://docs.scvi-tools.org/en/1.2.2/tutorials/notebooks/multimodal/totalVI.html)

We use the [PBMC5k dataset](https://muon-tutorials.readthedocs.io/en/latest/cite-seq/1-CITE-seq-PBMC-5k.html) primarily for development, debugging, and reproducibility checks.

#### Methods

We construct several downsampled versions of the labeled CITE-seq dataset in which one target cell type is retained at decreasing fractions while all other cell types are kept fixed. For each downsampling condition, we compare three simple representation strategies:

- an RNA-only representation,
- a protein-only representation, and
- a combined RNA-protein representation.

The RNA-only baseline is computed from normalized gene-expression values using highly variable genes followed by principal component analysis. The protein-only baseline is computed from normalized antibody-derived tag measurements. The combined representation is built by concatenating reduced RNA and protein feature spaces after appropriate scaling.

For each dataset, we select one or more target cell types with sufficient initial abundance and known biological marker support. We then create a sequence of downsampled datasets in which the target population is retained at predefined fractions, such as 50%, 25%, 10%, 5%, and, where feasible, 2%. All non-target cell types are retained during this procedure.

We evaluate rare-cell recovery using several complementary metrics:

1. **Rare-cell recall** measures the fraction of target cells that remain recoverable under a given representation.
2. **Rare-cell precision** measures whether cells predicted or grouped with the target population are truly target cells.
3. The **F1 score** combines precision and recall and serves as a primary summary metric.

We also compute **neighborhood purity**, defined as the fraction of nearest neighbors of a target cell that share the same cell-type label, to directly test whether rare cells remain locally organized or become absorbed into nearby abundant populations.

Furthermore, the **Silhouette score** is used to evaluate geometric separability of the target population in the learned representation. Where marker information is available, we also compute **marker-based AUC-ROC** to test whether known RNA or protein markers still distinguish the target population after downsampling.

The benchmark includes several controls to reduce the risk of misleading conclusions:

1. We repeat all downsampling experiments across multiple random seeds and report mean performance with variability estimates.
2. We include an abundant-cell control in which a common cell type is artificially downsampled to the same rarity levels; this helps determine whether the observed behavior is specific to the selected target population or simply a generic consequence of low sample size.
3. We include a random-label control in which target-cell labels are permuted to ensure that the recovery metrics collapse when biological label structure is removed.
4. Finally, we perform marker sanity checks using known immune-cell markers to confirm that the selected target population is biologically coherent; if one marker dominates the result, an optional marker-removal sensitivity analysis can be performed to test whether the representation preserves broader cell identity or merely one highly informative feature.

#### Repository layout

```
config/                         Configuration YAML files for benchmark runs
data/                           Raw and processed data (gitignored except README and .gitkeep)
  5k_pbmc_protein_v3_nextgem/   Raw 10x HDF5 data (downloaded automatically on first run)
  processed/                    Preprocessed .h5ad files (generated by the pipeline scripts)
manuscript/                     Manuscript drafts and mini-reports
notebooks/                      Interactive companion notebooks, one per pipeline step
results/                        All generated outputs (gitignored except .gitkeep placeholders)
  figures/                      Publication-quality PNG and PDF figures
  intermediate/                 Intermediate AnnData snapshots and cell-count grids
  logs/                         Per-run log files
  metrics/                      Raw per-condition benchmark results (CSV and optional Parquet)
  reports/                      Auto-generated Markdown benchmark reports
  tables/                       Aggregated summary tables and control outputs
scripts/                        Canonical reproducible pipeline scripts
src/rarecell/                   Reusable benchmark library installed via pip
tests/                          Unit and integration test suite
```

#### Environment setup

```bash
conda env create -f environment.yml
conda activate rarecell-citeseq
pip install -e .
```

#### Canonical script workflow

Run scripts from the project root with the conda environment active. Each step writes its outputs to a subdirectory under `results/`.

```bash
python scripts/import_data.py
python scripts/preprocess_data.py
python scripts/run_benchmark.py
python scripts/run_controls.py --config config/benchmark_config.yaml
python scripts/generate_summary_tables.py
python scripts/generate_figures.py
python scripts/write_report.py
```

For a quick end-to-end smoke test using a synthetic dataset, run:

```bash
python scripts/run_smoke_test.py
```

#### Canonical notebook workflow

| Notebook                               | Mirrors script         |
|----------------------------------------|------------------------|
| `01_data_import_qc.ipynb`              | `import_data.py`       |
| `02_preprocessing_representations.ipynb` | `preprocess_data.py` |
| `03_downsampling_benchmark.ipynb`      | `run_benchmark.py`     |
| `04_results_figures.ipynb`             | `generate_figures.py`  |

Exploratory and extension notebooks are in `notebooks/archive/` and are not part of the canonical workflow.

Notebooks require the processed `.h5ad` file to be present at the path configured in `config/benchmark_config.yaml` (default: `data/processed/pbmc5k_10x_citeseq_representations.h5ad`).

#### Expected data locations

The pipeline expects CITE-seq input data in one of the following locations:

- `data/5k_pbmc_protein_v3_nextgem/filtered_feature_bc_matrix.h5` — downloaded automatically when running `make_qc_summary.py` with the default `scvi:5k_pbmc_protein_v3_nextgem` spec, or placed manually.
- Any `.h5ad` or `.h5mu` file containing RNA in `.X` and protein counts in `.obsm["protein_counts"]`; pass it explicitly with `--input <path>`.

Processed files written by the pipeline are stored under `data/processed/` and are gitignored. The canonical filenames are `pbmc5k_10x_citeseq_processed.h5ad` and `pbmc5k_10x_citeseq_representations.h5ad`.

#### Configuration files

| File                                   | Purpose                                                                                       |
|----------------------------------------|-----------------------------------------------------------------------------------------------|
| `config/benchmark_config.yaml`         | Primary config: dataset path, label column, target cell type, fractions, seeds, control flags |
| `config/example_benchmark_config.yaml` | Annotated template; copy and edit for new datasets                                            |

The `target_cell_type` field in `benchmark_config.yaml` defaults to `null`, which causes the benchmark to auto-select the second most abundant cell type as the target population. Set it explicitly to a Leiden cluster label or a cell-type name to pin the target.

#### Generated outputs

Selected key outputs written to `results/`:

| Output                                  | Description                                                                 |
|-----------------------------------------|-----------------------------------------------------------------------------|
| `metrics/rare_cell_benchmark_raw.csv`   | One row per (dataset, target, fraction, seed, representation)               |
| `metrics/abundant_cell_control_raw.csv` | Same schema for the abundant-cell control                                   |
| `tables/benchmark_summary.csv`          | Seed-aggregated mean, std, n per (target, representation, fraction, metric) |
| `tables/best_method_by_fraction.csv`    | Best and second-best representation per (target, fraction, metric)          |
| `tables/random_label_control.csv`       | Permuted-label control results                                              |
| `figures/rare_cell_f1_curve.png`        | F1 score vs. retained fraction, one line per representation                 |
| `figures/neighborhood_purity_curve.png` | Neighborhood purity vs. retained fraction                                   |
| `reports/`                              | Auto-generated Markdown run reports                                         |

All figure files are saved in both PNG (screen, 200 dpi) and PDF (print-ready) formats. File names follow the convention `{output_prefix}__{descriptor}.{ext}` where the double underscore separates the dataset prefix from the figure descriptor.

#### Testing

```bash
conda activate rarecell-citeseq
python -m compileall src/ scripts/ -q   # syntax check
python -m pytest tests/ -q              # 90 tests, ~25 s on a laptop CPU
```

Tests are in `tests/` and use `pytest`. The test suite covers importability, preprocessing and representation shape checks, downsampling correctness (including self-exclusion and fraction-boundary cases), metric correctness (precision, recall, F1, neighborhood purity), summary-table aggregation, plotting smoke tests, configuration loading, and report generation. Tests do not require the real PBMC dataset; they use synthetic AnnData fixtures generated in `tests/conftest.py`.

#### Troubleshooting

- **`scvi` download fails.** Place `filtered_feature_bc_matrix.h5` manually at `data/5k_pbmc_protein_v3_nextgem/filtered_feature_bc_matrix.h5` and pass `--input data/5k_pbmc_protein_v3_nextgem/filtered_feature_bc_matrix.h5`.
- **`KeyError: protein_counts`.** Run `preprocess_data.py` before the downsampling benchmark; the representations file must have `.obsm["protein_counts"]`.
- **`KeyError: X_rna_pca`.** The representations file is missing expected embeddings. Run `preprocess_data.py` again to rebuild them.
- **Logs and intermediate outputs.** All scripts write logs to `results/logs/` and intermediate files to `results/intermediate/`. Check these directories first when diagnosing unexpected results.
