"""Lightweight RNA and protein preprocessing pipelines."""

from __future__ import annotations

import warnings
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .utils import matrix_to_dataframe, safe_n_components, to_dense_array


def _normalize_total_log1p(adata: ad.AnnData, target_sum: float) -> None:
    """Normalize rows to target_sum and apply log1p without relying on Scanpy numba paths."""
    matrix = adata.X
    if sparse.issparse(matrix):
        matrix = matrix.tocsr(copy=True)
        totals = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
        scale = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals > 0)
        matrix = sparse.diags(scale).dot(matrix).tocsr()
        matrix.data = np.log1p(matrix.data)
        adata.X = matrix
    else:
        values = np.asarray(matrix, dtype=float)
        totals = values.sum(axis=1)
        scale = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals > 0)
        adata.X = np.log1p(values * scale[:, None])
    adata.uns["log1p"] = {"base": None}


def safe_standardize(matrix: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
    """Standardize features while preserving DataFrame metadata when present."""
    is_dataframe = isinstance(matrix, pd.DataFrame)
    index = matrix.index if is_dataframe else None
    columns = matrix.columns if is_dataframe else None
    values = matrix.to_numpy() if is_dataframe else to_dense_array(matrix)
    values = np.nan_to_num(values, copy=False)
    scaled = StandardScaler().fit_transform(values)
    if is_dataframe:
        return pd.DataFrame(scaled, index=index, columns=columns)
    return scaled


def preprocess_rna(
        adata: ad.AnnData,
        n_top_genes: int = 3000,
        target_sum: float = 1e4,
        n_pcs: int = 50,
        min_genes: int = 1,
        min_cells: int = 1,
        max_pct_mt: float | None = None,
        max_genes: int | None = None,
        max_counts: float | None = None,
) -> ad.AnnData:
    """Run a compact Scanpy RNA preprocessing workflow and store RNA PCA."""
    import scanpy as sc

    rna = adata.copy()
    if rna.n_obs == 0 or rna.n_vars == 0:
        raise ValueError("RNA AnnData must contain at least one cell and one gene.")

    if "counts" in rna.layers:
        rna.X = rna.layers["counts"].copy()
    else:
        rna.layers["counts"] = rna.X.copy()

    rna.var["mt"] = rna.var_names.astype(str).str.upper().str.startswith("MT-")
    try:
        sc.pp.calculate_qc_metrics(rna, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    except Exception as exc:
        warnings.warn(f"RNA QC metric calculation failed; continuing with basic filters. {exc}")

    sc.pp.filter_cells(rna, min_genes=min_genes)
    sc.pp.filter_genes(rna, min_cells=min_cells)
    if rna.n_obs == 0 or rna.n_vars == 0:
        raise ValueError("RNA filtering removed all cells or genes.")

    keep_cells = pd.Series(True, index=rna.obs_names)
    if max_pct_mt is not None and "pct_counts_mt" in rna.obs:
        keep_cells &= pd.to_numeric(rna.obs["pct_counts_mt"], errors="coerce").fillna(0) <= max_pct_mt
    if max_genes is not None and "n_genes_by_counts" in rna.obs:
        keep_cells &= pd.to_numeric(rna.obs["n_genes_by_counts"], errors="coerce").fillna(0) <= max_genes
    if max_counts is not None and "total_counts" in rna.obs:
        keep_cells &= pd.to_numeric(rna.obs["total_counts"], errors="coerce").fillna(0) <= max_counts
    if not bool(keep_cells.any()):
        raise ValueError("RNA optional QC filters removed all cells.")
    if not bool(keep_cells.all()):
        rna = rna[keep_cells.to_numpy(), :].copy()

    if rna.n_obs == 0 or rna.n_vars == 0:
        raise ValueError("RNA filtering removed all cells or genes.")

    _normalize_total_log1p(rna, target_sum=target_sum)

    n_hvg = min(int(n_top_genes), rna.n_vars)
    run_hvg = rna.n_vars > 1 and rna.n_obs > 2 and rna.n_obs >= 5 and rna.n_vars >= 10
    if run_hvg:
        try:
            sc.pp.highly_variable_genes(rna, n_top_genes=n_hvg, flavor="seurat")
        except Exception as exc:
            warnings.warn(f"Highly variable gene selection failed; using all genes. {exc}")
            rna.var["highly_variable"] = True
    else:
        rna.var["highly_variable"] = True

    if "highly_variable" not in rna.var or not bool(rna.var["highly_variable"].any()):
        rna.var["highly_variable"] = True

    run_scale = rna.n_obs >= 5 and rna.n_vars >= 5
    if run_scale:
        zero_center = not sparse.issparse(rna.X)
        sc.pp.scale(rna, max_value=10, zero_center=zero_center)
        if sparse.issparse(rna.X):
            if np.isnan(rna.X.data).any() or np.isinf(rna.X.data).any():
                rna.X.data = np.nan_to_num(rna.X.data, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            if np.isnan(rna.X).any() or np.isinf(rna.X).any():
                rna.X = np.nan_to_num(rna.X, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        if sparse.issparse(rna.X):
            if np.isnan(rna.X.data).any() or np.isinf(rna.X.data).any():
                rna.X.data = np.nan_to_num(rna.X.data, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            if np.isnan(rna.X).any() or np.isinf(rna.X).any():
                rna.X = np.nan_to_num(rna.X, nan=0.0, posinf=0.0, neginf=0.0)
    n_components = safe_n_components(n_pcs, rna.n_obs, int(rna.var["highly_variable"].sum()))
    try:
        sc.tl.pca(
            rna,
            n_comps=n_components,
            mask_var="highly_variable",
            svd_solver="arpack" if min(rna.n_obs, rna.n_vars) > n_components else "auto",
        )
    except TypeError:
        sc.tl.pca(
            rna,
            n_comps=n_components,
            use_highly_variable=True,
            svd_solver="arpack" if min(rna.n_obs, rna.n_vars) > n_components else "auto",
        )

    rna.obsm["X_rna_pca"] = np.asarray(rna.obsm["X_pca"])
    if rna.n_obs >= 3 and n_components >= 2:
        n_neighbors = min(15, rna.n_obs - 1)
        try:
            # SparseEfficiencyWarning is raised by scipy when scanpy sets diagonal
            # elements of the CSR distances matrix it builds internally. The
            # mutation is inside scanpy/scipy, not our code, and the result is
            # still correct, so we suppress only that warning here.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=sparse.SparseEfficiencyWarning)
                sc.pp.neighbors(rna, n_neighbors=n_neighbors, n_pcs=n_components)
            try:
                sc.tl.umap(rna, random_state=0)
            except Exception as exc:
                warnings.warn(f"RNA UMAP failed; PCA is still available. {exc}")
        except Exception as exc:
            warnings.warn(f"RNA neighbor graph construction failed; PCA is still available. {exc}")
    return rna


def _preprocess_protein_matrix(protein_matrix: Any) -> pd.DataFrame:
    """Log-transform non-negative protein counts and standardize features."""
    protein = matrix_to_dataframe(protein_matrix, prefix="protein")
    values = protein.to_numpy(dtype=float)
    if np.nanmin(values) >= 0:
        values = np.log1p(values)
    transformed = pd.DataFrame(values, index=protein.index, columns=protein.columns)
    standardized = safe_standardize(transformed)
    assert isinstance(standardized, pd.DataFrame)
    return standardized


def preprocess_protein(protein_matrix: Any, n_components: int = 20) -> pd.DataFrame | ad.AnnData:
    """Preprocess protein counts or add a protein representation to AnnData.

    Matrix/DataFrame input returns a standardized protein feature DataFrame for
    backward compatibility. AnnData input reads ``.obsm["protein_counts"]`` and
    stores ``.obsm["X_protein_pca"]``.
    """
    if isinstance(protein_matrix, ad.AnnData):
        adata = protein_matrix.copy()
        if "protein_counts" not in adata.obsm:
            raise KeyError("Expected adata.obsm['protein_counts'] before protein preprocessing.")
        names = adata.uns.get("protein_names")
        protein = matrix_to_dataframe(
            adata.obsm["protein_counts"],
            index=adata.obs_names,
            columns=names,
            prefix="protein",
        )
        standardized = _preprocess_protein_matrix(protein)
        n_proteins = standardized.shape[1]
        if n_proteins <= 2:
            values = standardized.to_numpy()
        else:
            n_pcs = safe_n_components(n_components, standardized.shape[0], n_proteins)
            values = PCA(n_components=n_pcs, random_state=0).fit_transform(standardized.to_numpy())
        adata.obsm["X_protein_pca"] = values
        return adata
    return _preprocess_protein_matrix(protein_matrix)


def align_cells_between_modalities(
        rna_adata: ad.AnnData, protein_matrix: pd.DataFrame | np.ndarray
) -> tuple[ad.AnnData, pd.DataFrame]:
    """Align RNA and protein matrices by shared cell barcodes when possible."""
    protein = matrix_to_dataframe(protein_matrix, prefix="protein")
    rna_index = pd.Index(rna_adata.obs_names)
    protein_index = pd.Index(protein.index)

    if len(rna_index) == len(protein_index) and rna_index.equals(protein_index):
        return rna_adata.copy(), protein.copy()

    shared = rna_index.intersection(protein_index)
    if len(shared) == 0:
        if rna_adata.n_obs == protein.shape[0]:
            protein.index = rna_index
            return rna_adata.copy(), protein
        raise ValueError("RNA and protein matrices have no shared cell barcodes.")

    return rna_adata[shared, :].copy(), protein.loc[shared].copy()
