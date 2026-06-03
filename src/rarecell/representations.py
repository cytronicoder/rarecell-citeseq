"""Baseline RNA, protein, and joint PCA representations."""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .utils import component_names, safe_n_components, to_dense_array


def compute_rna_pca(adata: ad.AnnData, n_components: int = 30) -> pd.DataFrame:
    """Return an RNA PCA embedding, using Scanpy PCA if already present."""
    n_components = safe_n_components(n_components, adata.n_obs, adata.n_vars)
    if "X_pca" in adata.obsm and adata.obsm["X_pca"].shape[1] >= n_components:
        values = np.asarray(adata.obsm["X_pca"][:, :n_components])
    else:
        try:
            import scanpy as sc

            tmp = adata.copy()
            sc.tl.pca(tmp, n_comps=n_components, svd_solver="arpack")
            values = np.asarray(tmp.obsm["X_pca"][:, :n_components])
        except Exception:
            values = PCA(n_components=n_components, random_state=0).fit_transform(to_dense_array(adata.X))
    return pd.DataFrame(values, index=adata.obs_names, columns=component_names("RNA_PC", values.shape[1]))


def compute_protein_pca(protein_matrix: Any, n_components: int = 10) -> pd.DataFrame:
    """Compute PCA on a preprocessed protein matrix."""
    protein = protein_matrix if isinstance(protein_matrix, pd.DataFrame) else pd.DataFrame(
        to_dense_array(protein_matrix))
    n_components = safe_n_components(n_components, protein.shape[0], protein.shape[1])
    values = PCA(n_components=n_components, random_state=0).fit_transform(protein.to_numpy())
    return pd.DataFrame(values, index=protein.index, columns=component_names("Protein_PC", values.shape[1]))


def compute_joint_pca_representation(
        rna_embedding: pd.DataFrame | np.ndarray,
        protein_embedding: pd.DataFrame | np.ndarray,
        scale_blocks: bool = True,
) -> pd.DataFrame:
    """Compute a joint RNA+protein PCA embedding via scaled concatenation."""
    rna = rna_embedding if isinstance(rna_embedding, pd.DataFrame) else pd.DataFrame(to_dense_array(rna_embedding))
    protein = (
        protein_embedding
        if isinstance(protein_embedding, pd.DataFrame)
        else pd.DataFrame(to_dense_array(protein_embedding))
    )

    if rna.shape[0] != protein.shape[0]:
        shared = pd.Index(rna.index).intersection(pd.Index(protein.index))
        if len(shared) == 0:
            raise ValueError("RNA and protein embeddings have different cell counts and no shared index.")
        rna = rna.loc[shared]
        protein = protein.loc[shared]
    elif not pd.Index(rna.index).equals(pd.Index(protein.index)):
        shared = pd.Index(rna.index).intersection(pd.Index(protein.index))
        if len(shared) == rna.shape[0]:
            rna = rna.loc[shared]
            protein = protein.loc[shared]
        else:
            raise ValueError("RNA and protein embeddings have mismatched cell indices.")

    rna_values = rna.to_numpy()
    protein_values = protein.to_numpy()
    if scale_blocks:
        rna_values = StandardScaler().fit_transform(rna_values)
        protein_values = StandardScaler().fit_transform(protein_values)

    values = np.hstack([rna_values, protein_values])
    columns = [f"joint_rna_{col}" for col in rna.columns] + [
        f"joint_protein_{col}" for col in protein.columns
    ]
    return pd.DataFrame(values, index=rna.index, columns=columns)


def build_joint_representation(
        adata: ad.AnnData,
        rna_key: str = "X_rna_pca",
        protein_key: str = "X_protein_pca",
        output_key: str = "X_joint_pca",
        scale_blocks: bool = True,
) -> ad.AnnData:
    """Store a standardized RNA+protein PCA concatenation in ``adata.obsm``."""
    missing = [key for key in [rna_key, protein_key] if key not in adata.obsm]
    if missing:
        raise KeyError(
            f"Missing required adata.obsm keys for joint representation: {missing}. "
            f"Available keys: {list(adata.obsm.keys())}."
        )
    rna = pd.DataFrame(to_dense_array(adata.obsm[rna_key]), index=adata.obs_names)
    protein = pd.DataFrame(to_dense_array(adata.obsm[protein_key]), index=adata.obs_names)
    joint = compute_joint_pca_representation(rna, protein, scale_blocks=scale_blocks)
    values = joint.reindex(adata.obs_names).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"Joint representation '{output_key}' contains NaN or infinite values.")
    adata.obsm[output_key] = values
    return adata
