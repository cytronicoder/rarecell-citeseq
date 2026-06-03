"""Tests for src/rarecell/summaries.py."""

import pandas as pd

from rarecell.benchmark import (
    make_benchmark_summary,
    make_best_method_summary,
    make_abundant_control_summary,
)


def _make_raw_df(
        n_seeds: int = 3,
        fractions: tuple[float, ...] = (1.0, 0.5, 0.1),
        reps: tuple[str, ...] = ("rna_pca", "protein_pca"),
) -> pd.DataFrame:
    rows = []
    for frac in fractions:
        for seed in range(n_seeds):
            for rep in reps:
                rows.append(
                    {
                        "target_cell_type": "B",
                        "representation": rep,
                        "fraction": frac,
                        "seed": seed,
                        "precision": 0.8 + 0.05 * seed + 0.1 * frac,
                        "recall": 0.7 + 0.03 * seed + 0.15 * frac,
                        "f1": 0.75 + 0.04 * seed + 0.12 * frac,
                        "neighborhood_purity": 0.6 + 0.02 * seed,
                        "silhouette_target": 0.1 + 0.01 * seed,
                    }
                )
    return pd.DataFrame(rows)


def test_benchmark_summary_shape():
    raw = _make_raw_df(n_seeds=3, fractions=(1.0, 0.5, 0.1), reps=("rna_pca", "protein_pca"))
    summary = make_benchmark_summary(raw)
    # 2 reps × 3 fractions × 5 metrics = 30 rows
    assert len(summary) == 2 * 3 * 5
    assert set(summary.columns) >= {"target_cell_type", "representation", "fraction", "metric", "mean", "std",
                                    "n_seeds", "n_valid"}


def test_benchmark_summary_n_seeds():
    raw = _make_raw_df(n_seeds=5)
    summary = make_benchmark_summary(raw)
    assert (summary["n_seeds"] == 5).all()


def test_benchmark_summary_mean_correct():
    raw = pd.DataFrame(
        {
            "target_cell_type": ["B", "B", "B"],
            "representation": ["rna_pca"] * 3,
            "fraction": [1.0] * 3,
            "seed": [0, 1, 2],
            "f1": [0.6, 0.8, 1.0],
            "precision": [0.5, 0.7, 0.9],
            "recall": [0.4, 0.6, 0.8],
            "neighborhood_purity": [0.3, 0.5, 0.7],
            "silhouette_target": [0.1, 0.2, 0.3],
        }
    )
    summary = make_benchmark_summary(raw)
    f1_row = summary[(summary["metric"] == "f1") & (summary["representation"] == "rna_pca")]
    assert abs(float(f1_row["mean"].iloc[0]) - 0.8) < 1e-9
    assert int(f1_row["n_valid"].iloc[0]) == 3


def test_benchmark_summary_nan_handled():
    raw = pd.DataFrame(
        {
            "target_cell_type": ["B", "B"],
            "representation": ["rna_pca"] * 2,
            "fraction": [1.0] * 2,
            "seed": [0, 1],
            "f1": [0.8, float("nan")],
            "precision": [0.8, float("nan")],
            "recall": [0.8, float("nan")],
            "neighborhood_purity": [0.8, float("nan")],
            "silhouette_target": [0.8, float("nan")],
        }
    )
    summary = make_benchmark_summary(raw)
    f1_row = summary[summary["metric"] == "f1"]
    assert int(f1_row["n_valid"].iloc[0]) == 1
    assert int(f1_row["n_seeds"].iloc[0]) == 2


def test_benchmark_summary_empty_input():
    summary = make_benchmark_summary(pd.DataFrame())
    assert summary.empty
    assert "metric" in summary.columns


def test_best_method_columns():
    raw = _make_raw_df()
    summary = make_benchmark_summary(raw)
    best = make_best_method_summary(summary)
    required = {
        "target_cell_type", "fraction", "metric",
        "best_representation", "best_mean",
        "second_best_representation", "second_best_mean", "delta",
    }
    assert required.issubset(best.columns)


def test_best_method_picks_highest_mean():
    summary = pd.DataFrame(
        {
            "target_cell_type": ["B", "B"],
            "representation": ["rna_pca", "protein_pca"],
            "fraction": [1.0, 1.0],
            "metric": ["f1", "f1"],
            "mean": [0.9, 0.6],
            "std": [0.0, 0.0],
            "n_seeds": [5, 5],
            "n_valid": [5, 5],
        }
    )
    best = make_best_method_summary(summary)
    f1_row = best[best["metric"] == "f1"].iloc[0]
    assert f1_row["best_representation"] == "rna_pca"
    assert abs(float(f1_row["best_mean"]) - 0.9) < 1e-9
    assert f1_row["second_best_representation"] == "protein_pca"
    assert abs(float(f1_row["delta"]) - 0.3) < 1e-9


def test_abundant_control_summary_schema():
    raw = _make_raw_df(reps=("rna_pca",))
    raw["target_cell_type"] = "NK"
    control_summary = make_abundant_control_summary(raw)
    assert set(control_summary.columns) >= {"target_cell_type", "representation", "fraction", "metric", "mean",
                                            "n_seeds"}


def test_abundant_control_summary_empty():
    result = make_abundant_control_summary(pd.DataFrame())
    assert result.empty
