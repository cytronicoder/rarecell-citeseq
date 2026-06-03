import numpy as np
import pandas as pd
import pytest

ad = pytest.importorskip("anndata")

from rarecell.benchmark import make_metric_summary, run_downsampling_benchmark, run_random_label_control, \
    validate_results_table
from rarecell.io import get_protein_matrix, to_internal_anndata
from rarecell.plotting import save_all_standard_plots
from rarecell.reporting import write_benchmark_report


def _standard_adata() -> ad.AnnData:
    rng = np.random.default_rng(0)
    n_cells = 36
    n_genes = 12
    n_proteins = 5
    obs = pd.DataFrame(
        {"cell_type": ["target"] * 12 + ["other"] * 24},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    x = rng.poisson(2.0, size=(n_cells, n_genes)).astype(float)
    x[:12, :3] += 5
    protein = rng.poisson(2.0, size=(n_cells, n_proteins)).astype(float)
    protein[:12, :2] += 4
    adata = ad.AnnData(x, obs=obs, var=var)
    adata.layers["counts"] = x.copy()
    adata.obsm["protein_counts"] = protein
    adata.uns["protein_names"] = [f"adt_{i}" for i in range(n_proteins)]
    return adata


def test_standard_anndata_preserves_protein_modality():
    raw = _standard_adata()
    converted = to_internal_anndata(raw, dataset="synthetic")
    protein = get_protein_matrix(converted)
    assert converted.uns["dataset"] == "synthetic"
    assert converted.layers["counts"].shape == converted.X.shape
    assert protein.shape == (converted.n_obs, 5)
    assert list(protein.columns) == [f"adt_{i}" for i in range(5)]


def test_benchmark_plot_and_report_smoke(tmp_path):
    adata = _standard_adata()
    representations = ["rna_pca", "protein_pca", "joint_pca"]
    fractions = [1.0, 0.5]
    seeds = [0]

    results = run_downsampling_benchmark(
        adata,
        label_key="cell_type",
        target_label="target",
        representation_keys=representations,
        retain_fractions=fractions,
        seeds=seeds,
        n_neighbors=3,
        dataset="synthetic",
        rna_pcs=4,
        protein_pcs=3,
        n_top_genes=8,
    )
    validate_results_table(results, representations, fractions, seeds)
    assert set(results["representation"]) == set(representations)
    assert set(results["retain_fraction"].astype(float)) == set(fractions)

    metric_summary = make_metric_summary(results)
    figure_paths = save_all_standard_plots(
        results,
        output_prefix="synthetic_target",
        figures_dir=tmp_path / "figures",
        tables_dir=tmp_path / "tables",
    )
    assert figure_paths
    assert (tmp_path / "tables" / "synthetic_target__figure_index.csv").exists()

    run_summary = {
        "input_file": "synthetic",
        "label_key": "cell_type",
        "target_label": "target",
        "output_prefix": "synthetic_target",
        "representations": representations,
        "retain_fractions": fractions,
        "seeds": seeds,
        "n_neighbors": 3,
        "n_cells": adata.n_obs,
        "target_cell_count_original": 12,
        "n_cell_types": 2,
        "output_files": [str(path) for path in figure_paths],
    }
    report = write_benchmark_report(tmp_path / "report.md", run_summary, metric_summary)
    assert report.exists()
    assert "Rare-cell downsampling benchmark report" in report.read_text()

    random_control = run_random_label_control(
        adata,
        label_key="cell_type",
        target_label="target",
        representations=representations,
        retain_fraction=0.5,
        seed=0,
        n_neighbors=3,
    )
    assert {"control", "representation", "f1"}.issubset(random_control.columns)
