import numpy as np
import pandas as pd
import pytest

ad = pytest.importorskip("anndata")
pytest.importorskip("scanpy")

from rarecell.preprocessing import align_cells_between_modalities, preprocess_protein, preprocess_rna


def test_rna_preprocessing_runs_on_tiny_data():
    rng = np.random.default_rng(0)
    counts = rng.poisson(2, size=(12, 20))
    adata = ad.AnnData(
        counts,
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(12)]),
        var=pd.DataFrame(index=[f"gene{i}" for i in range(20)]),
    )
    processed = preprocess_rna(adata, n_top_genes=10, n_pcs=5)
    assert "X_pca" in processed.obsm
    assert processed.obsm["X_pca"].shape[0] == processed.n_obs
    assert "counts" in processed.layers


def test_protein_preprocessing_returns_dataframe():
    protein = pd.DataFrame(
        [[0, 1, 2], [2, 3, 4], [5, 1, 0]],
        index=["a", "b", "c"],
        columns=["CD3", "CD19", "CD14"],
    )
    processed = preprocess_protein(protein)
    assert isinstance(processed, pd.DataFrame)
    assert processed.shape == protein.shape
    assert np.isfinite(processed.to_numpy()).all()


def test_align_cells_between_modalities_by_barcode():
    adata = ad.AnnData(np.ones((3, 2)), obs=pd.DataFrame(index=["a", "b", "c"]))
    protein = pd.DataFrame(np.ones((3, 2)), index=["c", "a", "b"])
    aligned_rna, aligned_protein = align_cells_between_modalities(adata, protein)
    assert list(aligned_rna.obs_names) == ["a", "b", "c"]
    assert list(aligned_protein.index) == ["a", "b", "c"]


def test_rna_preprocessing_preserves_existing_counts_layer():
    counts = np.arange(20).reshape(4, 5) + 1
    adata = ad.AnnData(
        counts.copy(),
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(4)]),
        var=pd.DataFrame(index=[f"gene{i}" for i in range(5)]),
    )
    adata.layers["counts"] = counts.copy()
    processed = preprocess_rna(adata, n_top_genes=5, n_pcs=2)
    np.testing.assert_array_equal(processed.layers["counts"], counts)


def test_rna_optional_filters_raise_clear_error_when_all_cells_removed():
    counts = np.ones((4, 5))
    adata = ad.AnnData(
        counts,
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(4)]),
        var=pd.DataFrame(index=[f"gene{i}" for i in range(5)]),
    )
    with pytest.raises(ValueError, match="optional QC filters removed all cells"):
        preprocess_rna(adata, max_counts=0)
