import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ad = pytest.importorskip("anndata")

from rarecell.config import DATA_DIR
from rarecell.io import (
    detect_modalities,
    get_protein_matrix,
    load_citeseq,
    save_citeseq_object,
    validate_citeseq_object,
)


def make_adata_with_obsm(key="protein_expression", feature_types: bool = False):
    obs = pd.DataFrame({"cell_type": ["T", "B", "T", "Mono"]}, index=[f"cell{i}" for i in range(4)])
    var = pd.DataFrame(index=[f"gene{i}" for i in range(5)])
    if feature_types:
        var["feature_types"] = ["Gene Expression", "Gene Expression", "Gene Expression", "Antibody Capture",
                                "Antibody Capture"]
    adata = ad.AnnData(np.ones((4, 5)), obs=obs, var=var)
    adata.obsm[key] = pd.DataFrame(
        np.arange(12).reshape(4, 3),
        index=adata.obs_names,
        columns=["CD3", "CD19", "CD14"],
    )
    return adata


@pytest.mark.parametrize("key", ["protein_expression", "protein_counts", "adt"])
def test_anndata_protein_obsm_detection(key):
    adata = make_adata_with_obsm(key)
    info = detect_modalities(adata)
    assert info["protein_location"] == "obsm"
    assert info["protein_key"] == key
    protein = get_protein_matrix(adata)
    assert protein.shape == (4, 3)


def test_combined_feature_type_detection():
    obs = pd.DataFrame(index=[f"cell{i}" for i in range(3)])
    var = pd.DataFrame(
        {"feature_types": ["Gene Expression", "Gene Expression", "Antibody Capture"]},
        index=["G1", "G2", "CD3"],
    )
    adata = ad.AnnData(np.ones((3, 3)), obs=obs, var=var)
    protein = get_protein_matrix(adata)
    validation = validate_citeseq_object(adata)
    assert protein.shape == (3, 1)
    assert validation["n_genes"] == 2
    assert validation["n_protein_features"] == 1


def test_missing_protein_raises_clear_error():
    adata = ad.AnnData(np.ones((3, 4)))
    with pytest.raises(ValueError, match="No protein/ADT data found"):
        get_protein_matrix(adata)


def test_mudata_modality_detection_if_available():
    md = pytest.importorskip("mudata")
    md.set_options(pull_on_update=False)
    rna = ad.AnnData(
        np.ones((3, 4)),
        obs=pd.DataFrame(index=["a", "b", "c"]),
        var=pd.DataFrame(index=[f"gene{i}" for i in range(4)]),
    )
    prot = ad.AnnData(
        np.ones((3, 2)),
        obs=pd.DataFrame(index=["a", "b", "c"]),
        var=pd.DataFrame(index=["CD3", "CD19"]),
    )
    mdata = md.MuData({"rna": rna, "prot": prot})
    info = detect_modalities(mdata)
    protein = get_protein_matrix(mdata)
    assert info["rna_key"] == "rna"
    assert info["protein_key"] == "prot"
    assert protein.shape == (3, 2)


def test_scvi_10x_spec_uses_dataset_10x(monkeypatch):
    calls = []
    expected = make_adata_with_obsm()

    def fake_dataset_10x(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    scvi = types.ModuleType("scvi")
    scvi_data = types.ModuleType("scvi.data")
    scvi_data.dataset_10x = fake_dataset_10x
    monkeypatch.setitem(sys.modules, "scvi", scvi)
    monkeypatch.setitem(sys.modules, "scvi.data", scvi_data)

    assert load_citeseq("scvi:pbmc5k") is expected
    assert calls == [
        (("5k_pbmc_protein_v3_nextgem",), {"save_path": str(DATA_DIR), "gex_only": False})
    ]


def test_local_10x_h5_uses_scanpy_gex_only_false(monkeypatch):
    calls = []
    expected = make_adata_with_obsm(feature_types=True)

    def fake_read_10x_h5(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    scanpy = types.ModuleType("scanpy")
    scanpy.read_10x_h5 = fake_read_10x_h5
    monkeypatch.setitem(sys.modules, "scanpy", scanpy)

    path = "data/5k_pbmc_protein_v3_filtered_feature_bc_matrix.h5"
    assert load_citeseq(path) is expected
    assert calls[0][0] == (Path(path),)
    assert calls[0][1]["gex_only"] is False


def test_local_10x_mex_directory_uses_muon(monkeypatch, tmp_path):
    calls = []
    expected = make_adata_with_obsm()
    mex = tmp_path / "filtered_feature_bc_matrix"
    mex.mkdir()
    for filename in ["matrix.mtx", "barcodes.tsv", "features.tsv"]:
        (mex / filename).write_text("", encoding="utf-8")

    def fake_read_10x_mtx(path):
        calls.append(path)
        return expected

    muon = types.ModuleType("muon")
    muon.read_10x_mtx = fake_read_10x_mtx
    monkeypatch.setitem(sys.modules, "muon", muon)

    assert load_citeseq(mex) is expected
    assert calls == [mex]


def test_save_citeseq_object_requires_matching_suffix(monkeypatch, tmp_path):
    adata = make_adata_with_obsm()
    calls = []

    def fake_write_h5ad(path):
        calls.append(path)

    monkeypatch.setattr(adata, "write_h5ad", fake_write_h5ad)
    out = save_citeseq_object(adata, tmp_path / "object.h5ad")
    assert out == tmp_path / "object.h5ad"
    assert calls == [tmp_path / "object.h5ad"]
    with pytest.raises(ValueError, match=".h5ad"):
        save_citeseq_object(adata, tmp_path / "object.h5mu")
