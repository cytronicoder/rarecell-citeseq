### Data Directory

Place local public CITE-seq data files here. The canonical pipeline expects one paired RNA and ADT/protein dataset.

Recommended first dataset:

- 10x Genomics PBMC 5k CITE-seq / protein dataset used in Scanpy and Muon tutorials.
- Access it with the scvi 10x dataset spec `scvi:5k_pbmc_protein_v3_nextgem` (downloads into this directory).

Supported input shapes:

- `.h5ad` AnnData with RNA in `.X` and protein counts in `.obsm["protein_expression"]`, `.obsm["protein_counts"]`, or `.obsm["adt"]`.
- `.h5ad` AnnData containing combined 10x features with `adata.var["feature_types"]` values such as `Gene Expression` and `Antibody Capture`.
- `.h5mu` MuData with modalities named `rna` plus one of `prot`, `protein`, `adt`, or `modality2`.
- scvi 10x dataset specs like `scvi:5k_pbmc_protein_v3_nextgem`, loaded via `scvi.data.dataset_10x`.
- The short compatibility alias `scvi:pbmc5k` resolves to `scvi:5k_pbmc_protein_v3_nextgem`.

Use `scvi:` specs directly or point to local 10x files.

This repository does not commit downloaded data or generated analysis outputs.
