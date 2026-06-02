import pandas as pd
import pytest
from scipy import sparse

ad = pytest.importorskip("anndata")

from rarecell.markers import (
    MARKER_AUC_COLUMNS,
    compute_marker_auc_table,
    find_available_protein_markers,
    find_available_rna_markers,
    get_protein_feature_names,
    match_marker_to_features,
    marker_matching_diagnostics,
)


def make_marker_adata():
    labels = ["B cell", "B cell", "B cell", "Other", "Other", "Other"]
    obs = pd.DataFrame({"cell_type": labels}, index=[f"cell{i}" for i in range(6)])
    var = pd.DataFrame(index=["MS4A1", "CD79A", "GeneX"])
    x = sparse.csr_matrix(
        [
            [10, 9, 0],
            [9, 8, 0],
            [8, 7, 1],
            [1, 0, 2],
            [0, 1, 3],
            [2, 0, 1],
        ]
    )
    adata = ad.AnnData(x, obs=obs, var=var)
    adata.obsm["protein_counts"] = sparse.csr_matrix(
        [
            [10, 1],
            [9, 1],
            [8, 2],
            [1, 8],
            [0, 7],
            [2, 6],
        ]
    )
    adata.uns["protein_names"] = ["TotalSeqC anti-human CD19", "CD3"]
    return adata


def test_exact_rna_marker_matching():
    matches = find_available_rna_markers(make_marker_adata(), "B cell")
    assert any(match["marker"] == "MS4A1" and match["matched_feature_name"] == "MS4A1" for match in matches)


def test_partial_protein_marker_matching():
    matches = find_available_protein_markers(make_marker_adata(), "B cell")
    assert matches[0]["marker"] == "CD19"
    assert matches[0]["matched_feature_name"] == "TotalSeqC anti-human CD19"


def test_missing_marker_returns_empty_gracefully():
    assert find_available_rna_markers(make_marker_adata(), "Plasma") == []
    adata = make_marker_adata()
    del adata.uns["protein_names"]
    assert find_available_protein_markers(adata, "B cell") == []


def test_marker_auc_table_expected_columns_and_sparse_handling():
    table = compute_marker_auc_table(make_marker_adata(), "cell_type", "B cell")
    assert list(table.columns) == MARKER_AUC_COLUMNS
    assert {"rna", "protein"}.issubset(set(table["modality"]))
    assert {"predefined_biological_marker"} == set(table["marker_source"])
    assert table["auc"].max() > 0.9


def test_protein_marker_name_variants_match():
    assert match_marker_to_features("CD19", ["CD19_TotalSeqB"], "protein")["matched_feature_name"] == "CD19_TotalSeqB"
    assert match_marker_to_features("CD3", ["anti-human CD3"], "protein")["matched_feature_name"] == "anti-human CD3"
    assert match_marker_to_features("CD14", ["ADT_CD14"], "protein")["matched_feature_name"] == "ADT_CD14"


def test_rna_marker_matching_is_strict():
    assert match_marker_to_features("MS4A1", ["MS4A1", "CD19P"], "rna")["matched_feature_name"] == "MS4A1"
    assert match_marker_to_features("CD19", ["CD19P", "XCD19"], "rna") is None


def test_cluster_data_driven_markers_return_high_auc():
    labels = ["3", "3", "3", "1", "1", "1"]
    obs = pd.DataFrame({"cluster": labels}, index=[f"cell{i}" for i in range(6)])
    var = pd.DataFrame(index=["GENE_HIGH", "GENE_OTHER"])
    x = sparse.csr_matrix(
        [
            [10, 0],
            [9, 1],
            [8, 0],
            [0, 5],
            [1, 4],
            [0, 6],
        ]
    )
    adata = ad.AnnData(x, obs=obs, var=var)
    table = compute_marker_auc_table(adata, "cluster", "3")
    assert not table.empty
    assert table.iloc[0]["marker"] == "GENE_HIGH"
    assert table.iloc[0]["marker_source"] == "data_driven_cluster_marker"
    assert table.iloc[0]["auc"] > 0.9


def test_empty_marker_case_with_data_driven_disabled():
    adata = make_marker_adata()
    table = compute_marker_auc_table(
        adata,
        "cell_type",
        "Unmapped population",
        allow_data_driven_cluster_markers=False,
    )
    assert list(table.columns) == MARKER_AUC_COLUMNS
    assert table.empty


def test_missing_protein_names_does_not_crash_and_is_diagnostic():
    adata = make_marker_adata()
    del adata.uns["protein_names"]
    assert get_protein_feature_names(adata) == ["protein_0", "protein_1"]
    table = compute_marker_auc_table(adata, "cell_type", "B cell")
    assert not table.empty
    diagnostics = marker_matching_diagnostics(adata, "B cell")
    assert "interpretable protein feature names are unavailable" in " ".join(diagnostics["reason"])
