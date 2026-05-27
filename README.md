### Benchmarking rare-cell recovery in RNA–protein single-cell representations

Single-cell multimodal technologies such as CITE-seq measure both transcriptomes and surface-protein abundance in the same cells. This makes them powerful for distinguishing immune cell types and fine-grained cellular states. However, multimodal integration methods are usually evaluated on broad objectives such as clustering, batch correction, modality prediction, or global cell-type separation. Large benchmarks already compare many single-cell multimodal integration algorithms, including broad studies across RNA, protein, ATAC, DNA, and spatial modalities.

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
- a combined RNA–protein representation.

The RNA-only baseline is computed from normalized gene-expression values using highly variable genes followed by principal component analysis. The protein-only baseline is computed from normalized antibody-derived tag measurements. The combined representation is built by concatenating reduced RNA and protein feature spaces after appropriate scaling.

For each dataset, we select one or more target cell types with sufficient initial abundance and known biological marker support. We then create a sequence of downsampled datasets in which the target population is retained at predefined fractions, such as 50%, 25%, 10%, 5%, and, where feasible, 2%.  All non-target cell types are retained during this procedure.

We evaluate rare-cell recovery using several complementary metrics:

1. **Rare-cell recall** measures the fraction of target cells that remain recoverable under a given representation.
2. **Rare-cell precision** measures whether cells predicted or grouped with the target population are truly target cells.
3. The **F1 score** combines precision and recall and serves as a primary summary metric.

We also compute **neighborhood purity**, defined as the fraction of nearest neighbors of a target cell that share the same cell-type label, to directly test whether rare cells remain locally organized or become absorbed into nearby abundant populations. **Silhouette score** is used to evaluate geometric separability of the target population in the learned representation. Where marker information is available, we also compute **marker-based AUC-ROC** to test whether known RNA or protein markers still distinguish the target population after downsampling.

The benchmark includes several controls to reduce the risk of misleading conclusions. First, we repeat all downsampling experiments across multiple random seeds and report mean performance with variability estimates. Second, we include an abundant-cell control in which a common cell type is artificially downsampled to the same rarity levels. This helps determine whether the observed behavior is specific to the selected target population or simply a generic consequence of low sample size.

Third, we include a random-label control in which target-cell labels are permuted to ensure that the recovery metrics collapse when biological label structure is removed. Finally, we perform marker sanity checks using known immune-cell markers to confirm that the selected target population is biologically coherent; if one marker dominates the result, an optional marker-removal sensitivity analysis can be performed to test whether the representation preserves broader cell identity or merely one highly informative feature.
