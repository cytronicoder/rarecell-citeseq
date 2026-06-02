import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ad = pytest.importorskip("anndata")

from rarecell import benchmark
from rarecell.benchmark_plots import save_all_standard_plots
from rarecell.config import REPRESENTATION_KEY_MAP


def make_benchmark_adata():
    labels = np.array(["B cell"] * 20 + ["Other"] * 30)
    obs = pd.DataFrame({"cell_type_simple": labels}, index=[f"cell{i}" for i in range(50)])
    var = pd.DataFrame(index=["MS4A1", "CD79A", "GeneX"])
    x = np.column_stack(
        [
            np.r_[np.ones(20) * 8, np.zeros(30)],
            np.r_[np.ones(20) * 6, np.zeros(30)],
            np.arange(50),
        ]
    )
    adata = ad.AnnData(x, obs=obs, var=var)
    adata.layers["counts"] = adata.X.copy()
    adata.obsm["protein_counts"] = np.column_stack(
        [np.r_[np.ones(20) * 10, np.zeros(30)], np.r_[np.zeros(20), np.ones(30) * 8]]
    )
    adata.uns["protein_names"] = ["CD19", "CD3"]
    rna = np.column_stack([np.r_[np.zeros(20), np.ones(30) * 10], np.arange(50) / 50])
    protein = np.column_stack([np.r_[np.zeros(20), np.ones(30) * 8], np.arange(50) / 10])
    adata.obsm["X_rna_pca"] = rna
    adata.obsm["X_protein_pca"] = protein
    adata.obsm[REPRESENTATION_KEY_MAP["joint_pca"]] = np.hstack([rna, protein])
    return adata


def test_benchmark_outputs(tmp_path, monkeypatch):
    tables_dir = tmp_path / "tables"
    metrics_dir = tmp_path / "metrics"
    reports_dir = tmp_path / "reports"
    figures_dir = tmp_path / "figures"
    logs_dir = tmp_path / "logs"

    monkeypatch.setattr(benchmark, "TABLES_DIR", tables_dir)
    monkeypatch.setattr(benchmark, "METRICS_DIR", metrics_dir)
    monkeypatch.setattr(benchmark, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(benchmark, "FIGURES_DIR", figures_dir)
    monkeypatch.setattr(benchmark, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(benchmark, "RESULTS_DIR", tmp_path)

    def _ensure_dirs():
        for path in [tables_dir, metrics_dir, reports_dir, figures_dir, logs_dir]:
            path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(benchmark, "ensure_project_directories", _ensure_dirs)

    adata = make_benchmark_adata()
    results = benchmark.run_downsampling_benchmark(
        adata,
        label_key="cell_type_simple",
        target_label="B cell",
        representation_keys=["rna_pca", "protein_pca", "joint_pca"],
        retain_fractions=[1.0, 0.5],
        seeds=[0],
        n_neighbors=3,
    )
    benchmark.validate_results_table(results, ["rna_pca", "protein_pca", "joint_pca"], [1.0, 0.5], [0])
    assert set(results["representation"]) == {"rna_pca", "protein_pca", "joint_pca"}

    output_prefix = "synthetic_b_cell"
    benchmark.save_benchmark_results(results, output_prefix)
    assert (tables_dir / f"{output_prefix}__benchmark_results.csv").exists()

    metric_summary = benchmark.make_metric_summary(results)
    metric_summary_path = tables_dir / f"{output_prefix}__metric_summary.csv"
    metric_summary.to_csv(metric_summary_path, index=False)
    assert metric_summary_path.exists()

    saved_plots = save_all_standard_plots(
        results,
        output_prefix,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
    )
    assert saved_plots
    assert (tables_dir / f"{output_prefix}__figure_index.csv").exists()

    report_path = benchmark.write_markdown_report(
        output_prefix=output_prefix,
        input_file="synthetic.h5ad",
        label_key="cell_type_simple",
        target_label="B cell",
        representation_keys=["rna_pca", "protein_pca", "joint_pca"],
        retain_fractions=[1.0, 0.5],
        seeds=[0],
        n_neighbors=3,
        n_cells=adata.n_obs,
        original_target_cells=int((adata.obs["cell_type_simple"] == "B cell").sum()),
        n_cell_types=int(adata.obs["cell_type_simple"].nunique()),
        output_files=[str(p) for p in saved_plots],
        metric_summary=metric_summary,
    )
    assert report_path.exists()
