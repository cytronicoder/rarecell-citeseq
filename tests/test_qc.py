import numpy as np
import pandas as pd
import pytest

ad = pytest.importorskip("anndata")

from rarecell.qc import (
    make_candidate_target_population_table,
    make_protein_feature_table,
    make_protein_qc_cell_table,
    make_rna_qc_cell_table,
)


def test_make_protein_feature_table():
    protein = pd.DataFrame(
        [[1, 0], [3, 4], [0, 2]],
        index=["cell1", "cell2", "cell3"],
        columns=["CD3", "CD19"],
    )
    table = make_protein_feature_table(protein)
    assert list(table.columns) == [
        "protein_name",
        "total_counts",
        "mean_counts",
        "detected_cells",
        "detected_fraction",
    ]
    assert table.loc[table["protein_name"] == "CD3", "detected_cells"].item() == 2


def test_candidate_target_population_priorities():
    labels = pd.Series(["small"] * 10 + ["mid"] * 60 + ["large"] * 200)
    table = make_candidate_target_population_table(labels, min_cells=50)
    assert list(table["population"]) == ["mid", "large", "small"]
    assert list(table["candidate_priority"]) == [1, 2, 3]


def test_candidate_target_population_empty_without_labels():
    table = make_candidate_target_population_table(None)
    assert table.empty
    assert list(table.columns) == ["population", "n_cells", "fraction", "candidate_priority", "notes"]


def test_rna_qc_cell_table_computes_basic_fields():
    rna = ad.AnnData(
        np.array([[1, 0, 2], [0, 3, 0]]),
        obs=pd.DataFrame(index=["cell1", "cell2"]),
        var=pd.DataFrame(index=["MT-CO1", "GeneA", "GeneB"]),
    )
    rna.var["mt"] = [True, False, False]
    table = make_rna_qc_cell_table(rna)
    assert list(table["cell_id"]) == ["cell1", "cell2"]
    assert list(table["total_counts"]) == [3.0, 3.0]
    assert list(table["n_genes_by_counts"]) == [2, 1]
    assert table.loc[0, "pct_counts_mt"] == pytest.approx(100 / 3)


def test_protein_qc_cell_table_computes_basic_fields():
    protein = pd.DataFrame([[1, 0, 2], [0, 0, 0]], index=["cell1", "cell2"])
    table = make_protein_qc_cell_table(protein)
    assert list(table["cell_id"]) == ["cell1", "cell2"]
    assert list(table["total_protein_counts"]) == [3.0, 0.0]
    assert list(table["detected_proteins"]) == [2, 0]
    assert list(table["zero_fraction_per_cell"]) == [pytest.approx(1 / 3), 1.0]
