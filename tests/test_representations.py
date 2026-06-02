import numpy as np
import pandas as pd

from rarecell.representations import (
    compute_joint_pca_representation,
    compute_protein_pca,
    compute_umap_from_embedding,
    save_embedding,
)


def test_protein_pca_returns_expected_shape():
    rng = np.random.default_rng(0)
    protein = pd.DataFrame(rng.normal(size=(8, 5)), index=[f"cell{i}" for i in range(8)])
    embedding = compute_protein_pca(protein, n_components=3)
    assert embedding.shape == (8, 3)
    assert list(embedding.index) == list(protein.index)


def test_joint_representation_aligns_cell_counts():
    rna = pd.DataFrame(np.ones((5, 3)), index=[f"cell{i}" for i in range(5)])
    protein = pd.DataFrame(np.ones((5, 2)), index=[f"cell{i}" for i in range(5)])
    joint = compute_joint_pca_representation(rna, protein)
    assert joint.shape == (5, 5)


def test_joint_representation_aligns_by_cell_index():
    rna = pd.DataFrame(np.ones((3, 2)), index=["a", "b", "c"], columns=["r1", "r2"])
    protein = pd.DataFrame(np.ones((3, 2)), index=["c", "a", "b"], columns=["p1", "p2"])
    joint = compute_joint_pca_representation(rna, protein, scale_blocks=False)
    assert list(joint.index) == ["a", "b", "c"]
    assert list(joint.columns) == ["joint_rna_r1", "joint_rna_r2", "joint_protein_p1", "joint_protein_p2"]


def test_umap_from_embedding_preserves_index_for_tiny_data():
    embedding = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=["cell_a", "cell_b"],
        columns=["PC1", "PC2"],
    )
    umap = compute_umap_from_embedding(embedding)
    assert list(umap.columns) == ["UMAP1", "UMAP2"]
    assert list(umap.index) == ["cell_a", "cell_b"]


def test_save_embedding_csv(tmp_path):
    embedding = pd.DataFrame(np.ones((3, 2)), index=["a", "b", "c"], columns=["UMAP1", "UMAP2"])
    out = save_embedding(embedding, tmp_path / "embedding.csv")
    assert out.exists()
