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

---

#### Latest workflow

This section documents the current reproducible pipeline as of 2026-06-02.

**Project objective.** We benchmark whether RNA-only, protein-only, and joint RNA–protein representations preserve rare immune cell populations in CITE-seq data. Target cells are artificially downsampled to predefined fractions and rare-cell recall, precision, F1, neighborhood purity, and silhouette score are reported for each representation.

**Repository structure.**

```
rarecell-citeseq/
├── config/
│   ├── benchmark_config.yaml            # main config (pbmc5k_10x_citeseq)
│   └── pbmc5k_benchmark_config.yaml     # pbmc5k dev config
├── data/
│   ├── README.md                        # data placement instructions
│   └── .gitkeep
├── notebooks/
│   ├── data_loading_and_qc.ipynb
│   ├── baseline_representations.ipynb
│   ├── rare_cell_downsampling_benchmark.ipynb
│   └── benchmark_results_and_figures.ipynb
├── scripts/
│   ├── make_qc_summary.py               # QC tables and figures
│   ├── make_baseline_representations.py # RNA/protein/joint PCA + UMAP + save .h5ad
│   ├── preprocess_dataset.py            # RNA-only preprocessing helper
│   ├── run_rare_cell_benchmark.py       # primary benchmark runner
│   ├── run_smoke_test.py                # quick validation (≤1000 cells)
│   ├── run_downsampling_benchmark.py    # config-driven benchmark
│   ├── validate_benchmark_inputs.py     # pre-flight input check
│   ├── validate_dataset.py              # dataset validation helper
│   ├── import_10x_pbmc10k.py           # import 10x PBMC10k CITE-seq
│   ├── import_scvi_totalvi.py          # import scvi-tools PBMC5k/10k data
│   └── summarize_processed_datasets.py  # dataset-level summary tables
├── src/rarecell/
│   ├── benchmark.py                     # benchmark runner, result validation, reports
│   ├── benchmark_plots.py               # save_all_standard_plots()
│   ├── config.py                        # project-level paths and representation key map
│   ├── downsampling.py                  # downsample_target_cells(), grid utilities
│   ├── figure_utils.py                  # set_plot_style(), save_figure(), constants
│   ├── io.py, io_utils.py              # CITE-seq loaders and input resolution
│   ├── markers.py                       # marker gene/protein utilities
│   ├── metrics.py                       # compute_all_metrics() (kNN-based)
│   ├── naming.py                        # make_output_prefix(), result_path()
│   ├── plotting.py                      # plot_recovery_curve(), plot_embedding()
│   ├── preprocessing.py                 # preprocess_rna(), preprocess_protein()
│   ├── qc.py                            # QC table builders
│   ├── representations.py               # compute_rna_pca(), compute_protein_pca(), joint
│   ├── reporting.py                     # write_benchmark_report()
│   ├── script_utils.py                  # shared logger, label detection, repr resolution
│   ├── utils.py                         # infer_label_column(), write_json()
│   └── validation.py                    # validate_adata_fields(), find_label_key()
├── results/
│   ├── figures/                         # PNG + PDF publication figures
│   ├── tables/                          # CSV metric tables and summaries
│   ├── metrics/                         # Parquet benchmark results
│   ├── logs/                            # run logs and JSON summaries
│   ├── reports/                         # Markdown benchmark reports
│   └── intermediate/                    # cached subsample inputs
├── tests/                               # pytest test suite (71 tests)
├── manuscript/                          # draft report
├── environment.yml                      # conda environment definition
├── requirements.txt                     # pip requirements
└── pyproject.toml                       # package metadata and pytest config
```

**Data placement.** Raw and processed data are excluded from GitHub (see `.gitignore`). Place data files as follows before running the pipeline:

```
data/5k_pbmc_protein_v3_nextgem/filtered_feature_bc_matrix.h5   # 10x raw .h5
data/processed/pbmc5k_representations.h5ad                       # pre-built representations
```

Alternatively, the import scripts download and cache public datasets automatically on first run.

**Environment setup.**

```bash
conda env create -f environment.yml
conda activate rarecell-citeseq
pip install -e .
```

Or with pip only:

```bash
pip install -r requirements.txt
pip install -e .
```

**Exact command sequence.**

```bash
# 1. Import raw CITE-seq data (downloads on first run)
python scripts/import_scvi_totalvi.py

# 2. Generate QC tables and figures
python scripts/make_qc_summary.py

# 3. Compute RNA, protein, and joint baseline representations
python scripts/make_baseline_representations.py \
    --output data/processed/pbmc5k_representations.h5ad

# 4. Validate inputs before the main benchmark
python scripts/validate_benchmark_inputs.py \
    --input data/processed/pbmc5k_representations.h5ad

# 5. Run the benchmark (full grid: 5 fractions × 5 seeds)
python scripts/run_rare_cell_benchmark.py \
    --input data/processed/pbmc5k_representations.h5ad \
    --label-key cell_type_simple \
    --target-label "B cell" \
    --output-prefix pbmc5k_b_cell

# 5a. Or run a quick smoke test (≤1000 cells, 2 fractions × 2 seeds)
python scripts/run_smoke_test.py
```

**Expected outputs.** After the full benchmark run, output files follow the pattern `{output_prefix}__{descriptor}.{ext}`:

```
results/tables/pbmc5k_b_cell__benchmark_results.csv
results/tables/pbmc5k_b_cell__metric_summary.csv
results/tables/pbmc5k_b_cell__downsampling_grid.csv
results/tables/pbmc5k_b_cell__figure_index.csv
results/figures/pbmc5k_b_cell__rare_cell_f1_curve.{png,pdf}
results/figures/pbmc5k_b_cell__rare_cell_recall_curve.{png,pdf}
results/figures/pbmc5k_b_cell__neighborhood_purity_curve.{png,pdf}
results/figures/pbmc5k_b_cell__target_silhouette_curve.{png,pdf}
results/metrics/pbmc5k_b_cell__benchmark_results.parquet
results/logs/pbmc5k_b_cell__run.log
results/logs/pbmc5k_b_cell__run_summary.json
results/reports/pbmc5k_b_cell__benchmark_report.md
results/intermediate/pbmc5k_b_cell__cell_counts_by_fraction.csv
```

**Troubleshooting.**

- *`FileNotFoundError` for data files:* Run `python scripts/import_scvi_totalvi.py` first to download and cache PBMC5k data.
- *`KeyError: 'protein_counts'`:* Re-run `make_baseline_representations.py` to embed protein counts in `adata.obsm`.
- *Leiden clustering fails:* Requires `leidenalg` and `igraph`. Install via `pip install leidenalg python-igraph`.
- *Parquet output skipped:* Install `pyarrow` for parquet support; CSV output is always saved.
- *`UMAP` slow or missing:* Install `umap-learn`. UMAP is optional; PCA embeddings are used for metrics.
- *Tests fail to import modules:* Ensure the package is installed with `pip install -e .` and that `src/` is on the Python path.

**What is excluded from GitHub.**

```
data/raw/             # raw 10x .h5 files
data/processed/       # processed .h5ad AnnData objects
*.h5ad, *.h5mu, *.h5 # all large binary data files
results/figures/      # generated figures (PNG, PDF)
results/tables/       # generated CSV tables
results/metrics/      # generated Parquet metrics
results/logs/         # generated log and JSON files
results/reports/      # generated Markdown reports
results/intermediate/ # cached subsample inputs
__pycache__/          # Python bytecode
.venv/, env/, venv/   # virtual environments
.idea/, .vscode/      # editor settings
```

Only `.gitkeep` placeholder files are retained in `results/` subdirectories so the directory structure can be cloned without data.

**Current project status and remaining TODOs.**

- All 71 unit tests pass (`python -m pytest`).
- The PBMC5k downsampling benchmark runs end-to-end from `import_scvi_totalvi.py` through figure and report generation.
- CPU-only baselines only: `rna_pca`, `protein_pca`, and a simple concatenated `joint_pca`. Advanced methods (totalVI, WNN, MOFA) are not yet included.
- Validation on the PBMC10k dataset is pending.
- Biological marker sanity checks (marker AUC-ROC) are implemented in `src/rarecell/markers.py` but not yet integrated into the main pipeline.
- Abundant-cell and random-label controls are run by `run_downsampling_benchmark.py` but not yet by the primary `run_rare_cell_benchmark.py` script.
