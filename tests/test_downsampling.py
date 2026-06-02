import numpy as np
import pandas as pd
import pytest

ad = pytest.importorskip("anndata")

from rarecell.downsampling import downsample_target_cells, get_target_counts


def make_adata():
    obs = pd.DataFrame(
        {"cell_type": ["B"] * 6 + ["T"] * 4},
        index=[f"cell{i}" for i in range(10)],
    )
    var = pd.DataFrame(index=[f"gene{i}" for i in range(3)])
    adata = ad.AnnData(np.ones((10, 3)), obs=obs, var=var)
    adata.obsm["X_test"] = np.arange(20).reshape(10, 2)
    return adata


def test_target_cells_decrease_and_non_target_remain():
    adata = make_adata()
    result = downsample_target_cells(adata, "cell_type", "B", 0.5, seed=0)
    counts = result.obs["cell_type"].value_counts().to_dict()
    assert counts["B"] == 3
    assert counts["T"] == 4
    assert set(adata.obs_names[adata.obs["cell_type"] == "T"]).issubset(set(result.obs_names))
    assert result.obsm["X_test"].shape == (7, 2)


def test_same_seed_retains_same_cells():
    adata = make_adata()
    first = downsample_target_cells(adata, "cell_type", "B", 0.5, seed=3)
    second = downsample_target_cells(adata, "cell_type", "B", 0.5, seed=3)
    assert list(first.obs_names) == list(second.obs_names)


def test_different_seeds_usually_retain_different_target_cells():
    adata = make_adata()
    first = downsample_target_cells(adata, "cell_type", "B", 0.5, seed=1)
    second = downsample_target_cells(adata, "cell_type", "B", 0.5, seed=2)
    first_targets = set(first.obs_names[first.obs["is_target_cell"]])
    second_targets = set(second.obs_names[second.obs["is_target_cell"]])
    assert first_targets != second_targets


def test_invalid_label_key_raises_key_error():
    with pytest.raises(KeyError):
        downsample_target_cells(make_adata(), "missing", "B", 0.5)


def test_invalid_target_label_raises_value_error():
    with pytest.raises(ValueError, match="not present"):
        downsample_target_cells(make_adata(), "cell_type", "NK", 0.5)


def test_invalid_retain_fraction_raises_value_error():
    with pytest.raises(ValueError):
        downsample_target_cells(make_adata(), "cell_type", "B", 0)
    with pytest.raises(ValueError):
        downsample_target_cells(make_adata(), "cell_type", "B", 1.1)


def test_min_target_cells_behavior():
    result = downsample_target_cells(make_adata(), "cell_type", "B", 0.05, seed=0, min_target_cells=2)
    assert (result.obs["cell_type"].astype(str) == "B").sum() == 2
    with pytest.raises(ValueError, match="fewer than min_target_cells"):
        downsample_target_cells(make_adata(), "cell_type", "T", 0.5, seed=0, min_target_cells=5)


def test_downsampling_metadata_and_counts_table():
    result = downsample_target_cells(make_adata(), "cell_type", "B", 0.5, seed=0)
    assert result.uns["downsampling"]["n_target_before"] == 6
    assert result.uns["downsampling"]["n_target_after"] == 3
    for column in ["downsample_target", "retain_fraction", "downsample_seed", "is_target_cell"]:
        assert column in result.obs
    assert result.obs["is_target_cell"].dtype == bool
    expected = result.obs["cell_type"].astype(str) == "B"
    assert result.obs["is_target_cell"].to_numpy().tolist() == expected.to_numpy().tolist()
    counts = get_target_counts(result, "cell_type")
    assert list(counts.columns) == ["label", "n_cells", "fraction"]
