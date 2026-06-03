# Rare-cell CITE-seq Benchmark Mini-Report

## Project Question

Do RNA-only, protein-only, and simple RNA+protein representations preserve a target cell type
after artificial rarity is introduced through controlled downsampling?

## Dataset Used

Dataset: `pbmc5k_10x_citeseq`. Input path: `data/processed/pbmc5k_10x_citeseq_representations.h5ad`.
Total cells: 5527. Cell types: 0.

## Preprocessing Summary

RNA counts were normalized (scran-style or library-size), log-transformed, filtered to
highly-variable genes, scaled, and embedded with PCA (50 components).
Protein counts were log-transformed, standardized, and embedded with PCA (20 components).
Representations were **recomputed from scratch** after each downsampling condition to avoid
information leakage from the full dataset.

## Target Cell Type

Target: leiden cluster `2` (selected as a minority population).
Abundant-cell control: leiden cluster `3`.

## Downsampling Design

Only target cells were downsampled. Non-target cells were held fixed.
Retained fractions: `[1.0, 0.5, 0.25, 0.1, 0.05]`.
Seeds: `[0, 1, 2, 3, 4]` (independent random draws per fraction).

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

kNN uses k=15 neighbors; self-exclusion is applied.

## Main Benchmark Results

### F1 score by representation and fraction

| target_cell_type | representation | fraction | mean_F1 | std_F1 | n_seeds |
|---|---|---|---|---|---|
| 2 | joint_pca | 0.050 | 0.412 | 0.051 | 5 |
| 2 | joint_pca | 0.100 | 0.627 | 0.031 | 5 |
| 2 | joint_pca | 0.250 | 0.726 | 0.013 | 5 |
| 2 | joint_pca | 0.500 | 0.702 | 0.006 | 5 |
| 2 | joint_pca | 1.000 | 0.736 | 0.000 | 5 |
| 2 | protein_pca | 0.050 | 0.000 | 0.000 | 5 |
| 2 | protein_pca | 0.100 | 0.047 | 0.032 | 5 |
| 2 | protein_pca | 0.250 | 0.289 | 0.024 | 5 |
| 2 | protein_pca | 0.500 | 0.516 | 0.016 | 5 |
| 2 | protein_pca | 1.000 | 0.685 | 0.000 | 5 |
| 2 | rna_pca | 0.050 | 0.376 | 0.084 | 5 |
| 2 | rna_pca | 0.100 | 0.675 | 0.047 | 5 |
| 2 | rna_pca | 0.250 | 0.821 | 0.024 | 5 |
| 2 | rna_pca | 0.500 | 0.809 | 0.004 | 5 |
| 2 | rna_pca | 1.000 | 0.873 | 0.000 | 5 |

### Recall by representation and fraction

| target_cell_type | representation | fraction | mean_recall | std_recall | n_seeds |
|---|---|---|---|---|---|
| 2 | joint_pca | 0.050 | 0.292 | 0.042 | 5 |
| 2 | joint_pca | 0.100 | 0.565 | 0.042 | 5 |
| 2 | joint_pca | 0.250 | 0.793 | 0.025 | 5 |
| 2 | joint_pca | 0.500 | 0.882 | 0.005 | 5 |
| 2 | joint_pca | 1.000 | 0.938 | 0.000 | 5 |
| 2 | protein_pca | 0.050 | 0.000 | 0.000 | 5 |
| 2 | protein_pca | 0.100 | 0.025 | 0.017 | 5 |
| 2 | protein_pca | 0.250 | 0.203 | 0.020 | 5 |
| 2 | protein_pca | 0.500 | 0.474 | 0.024 | 5 |
| 2 | protein_pca | 1.000 | 0.735 | 0.000 | 5 |
| 2 | rna_pca | 0.050 | 0.258 | 0.067 | 5 |
| 2 | rna_pca | 0.100 | 0.602 | 0.057 | 5 |
| 2 | rna_pca | 0.250 | 0.857 | 0.031 | 5 |
| 2 | rna_pca | 0.500 | 0.932 | 0.010 | 5 |
| 2 | rna_pca | 1.000 | 0.958 | 0.000 | 5 |

### Best representation by fraction (F1)

| fraction | best_representation | best_mean_F1 | second_best | delta |
|---|---|---|---|---|
| 0.050 | joint_pca | 0.412 | rna_pca | 0.036 |
| 0.100 | rna_pca | 0.675 | joint_pca | 0.048 |
| 0.250 | rna_pca | 0.821 | joint_pca | 0.095 |
| 0.500 | rna_pca | 0.809 | joint_pca | 0.107 |
| 1.000 | rna_pca | 0.873 | joint_pca | 0.137 |

## Control Results

**Random-label control** (permuted labels destroy geometry-label relationship):

| representation | fraction | seed | precision | recall | f1 | neighborhood_purity |
|---|---|---|---|---|---|---|
| rna_pca | 0.050 | 0.000 | 0.000 | 0.000 | 0.000 | 0.008 |
| protein_pca | 0.050 | 0.000 | 0.000 | 0.000 | 0.000 | 0.007 |
| joint_pca | 0.050 | 0.000 | 0.000 | 0.000 | 0.000 | 0.008 |

*Expected result: random-label control should give near-zero F1 and near-random neighborhood purity.*

**Abundant-cell control** summary saved to `results/tables/abundant_cell_control_summary.csv`. Raw metrics at `results/metrics/abundant_cell_control_raw.csv`.

## Error Analysis Summary

Top confusion partners for target cells (highest error count):

| representation | fraction | confused_with | count | fraction_of_target_errors |
|---|---|---|---|---|
| protein_pca | 0.100 | 3 | 84 | 0.884 |
| protein_pca | 0.100 | 3 | 82 | 0.882 |
| protein_pca | 0.050 | 3 | 46 | 0.958 |
| protein_pca | 0.050 | 3 | 44 | 0.917 |
| joint_pca | 0.100 | 4 | 30 | 0.652 |

Neighbor composition table saved to `results/tables/target_neighbor_composition.csv`.

## Limitations

- This benchmark uses simple linear (PCA-style) representations only. Advanced methods
  (totalVI, WNN, MOFA+) are not included.
- The benchmark is CPU-only and does not train deep models.
- Target cell type is defined by Leiden clustering (leiden cluster `2`), not
  validated biological cell-type annotation.
- With very few target cells (fraction ≤ 0.05), some metric estimates become unstable
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
