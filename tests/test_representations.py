import numpy as np
import pandas as pd
import pytest

ad = pytest.importorskip("anndata")

from rarecell.representations import (
    build_joint_representation,
    compute_joint_pca_representation,
    compute_protein_pca,
)


def test_protein_pca_returns_expected_shape():
    rng = np.random.default_rng(0)
    protein = pd.DataFrame(rng.normal(size=(8, 5)), index=[f"cell{i}" for i in range(8)])
    embedding = compute_protein_pca(protein, n_components=3)
    assert embedding.shape == (8, 3)
    assert list(embedding.index) == list(protein.index)


def test_joint_representation_aligns_by_cell_index():
    rna = pd.DataFrame(np.ones((3, 2)), index=["a", "b", "c"], columns=["r1", "r2"])
    protein = pd.DataFrame(np.ones((3, 2)), index=["c", "a", "b"], columns=["p1", "p2"])
    joint = compute_joint_pca_representation(rna, protein, scale_blocks=False)
    assert list(joint.index) == ["a", "b", "c"]
    assert list(joint.columns) == ["joint_rna_r1", "joint_rna_r2", "joint_protein_p1", "joint_protein_p2"]


def test_build_joint_representation_writes_canonical_key():
    obs = pd.DataFrame(index=["a", "b", "c", "d"])
    var = pd.DataFrame(index=["g1", "g2"])
    adata = ad.AnnData(np.ones((4, 2)), obs=obs, var=var)
    adata.obsm["X_rna_pca"] = np.arange(8).reshape(4, 2)
    adata.obsm["X_protein_pca"] = np.arange(12).reshape(4, 3)
    result = build_joint_representation(adata)
    assert "X_joint_pca" in result.obsm
    assert "X_joint_simple" not in result.obsm
    assert result.obsm["X_joint_pca"].shape == (4, 5)
