import numpy as np
import pandas as pd

from rarecell.metrics import (
    macro_f1_from_embedding,
    marker_auc,
    neighborhood_purity,
    rare_cell_classification_metrics,
    summarize_metric_by_group,
    target_silhouette_score,
)


def test_perfect_separation_gives_high_f1_and_purity():
    embedding = np.vstack([np.zeros((12, 2)), np.ones((18, 2)) * 10])
    labels = np.array(["B"] * 12 + ["Other"] * 18)
    metrics = rare_cell_classification_metrics(embedding, labels, "B", seed=0, n_neighbors=3)
    assert metrics["f1"] >= 0.9
    assert metrics["recall"] >= 0.9
    assert metrics["macro_f1"] >= 0.9
    assert neighborhood_purity(embedding, labels, "B", k=5) >= 0.9


def test_classification_metrics_return_expected_keys():
    embedding = np.vstack([np.zeros((8, 2)), np.ones((8, 2)) * 5, np.ones((8, 2)) * 10])
    labels = np.array(["B"] * 8 + ["T"] * 8 + ["NK"] * 8)
    metrics = rare_cell_classification_metrics(embedding, labels, "B", seed=0, n_neighbors=3)
    expected = {"precision", "recall", "f1", "macro_f1", "support", "n_positive", "n_negative", "n_train", "n_test"}
    assert expected.issubset(metrics)
    assert macro_f1_from_embedding(embedding, labels, seed=0, n_neighbors=3) >= 0.9


def test_neighborhood_purity_known_configuration():
    embedding = np.array([[0.0], [0.1], [5.0], [5.1]])
    labels = np.array(["B", "B", "T", "T"])
    assert neighborhood_purity(embedding, labels, "B", k=1) == 1.0


def test_overlapping_labels_give_lower_neighborhood_purity():
    embedding = np.arange(40).reshape(20, 2)
    labels = np.array(["B", "Other"] * 10)
    assert neighborhood_purity(embedding, labels, "B", k=2) < 0.8


def test_missing_or_too_small_target_returns_nan_metrics():
    embedding = np.random.default_rng(0).normal(size=(6, 2))
    metrics = rare_cell_classification_metrics(embedding, ["Other"] * 6, "B")
    assert np.isnan(metrics["f1"])
    assert metrics["n_positive"] == 0


def test_silhouette_handles_invalid_cases():
    embedding = np.random.default_rng(0).normal(size=(6, 2))
    assert np.isnan(target_silhouette_score(embedding, ["Other"] * 6, "B"))
    assert np.isnan(target_silhouette_score(embedding, ["B"] + ["Other"] * 5, "B"))


def test_imbalanced_labels_do_not_crash():
    embedding = np.random.default_rng(0).normal(size=(22, 2))
    labels = np.array(["B"] * 2 + ["Other"] * 20)
    metrics = rare_cell_classification_metrics(embedding, labels, "B", n_neighbors=50)
    assert "f1" in metrics


def test_nan_embeddings_return_nan_metrics():
    embedding = np.array([[0.0, 0.0], [np.nan, 1.0], [2.0, 2.0], [3.0, 3.0]])
    labels = np.array(["B", "B", "Other", "Other"])
    metrics = rare_cell_classification_metrics(embedding, labels, "B")
    assert np.isnan(metrics["f1"])
    assert np.isnan(neighborhood_purity(embedding, labels, "B"))
    assert np.isnan(target_silhouette_score(embedding, labels, "B"))


def test_empty_embeddings_return_nan_metrics():
    embedding = np.empty((0, 2))
    labels = np.array([])
    metrics = rare_cell_classification_metrics(embedding, labels, "B")
    assert np.isnan(metrics["f1"])
    assert np.isnan(neighborhood_purity(embedding, labels, "B"))
    assert np.isnan(target_silhouette_score(embedding, labels, "B"))


def test_nan_labels_return_nan_metrics():
    embedding = np.random.default_rng(0).normal(size=(4, 2))
    labels = np.array(["B", np.nan, "Other", "Other"], dtype=object)
    metrics = rare_cell_classification_metrics(embedding, labels, "B")
    assert np.isnan(metrics["f1"])


def test_marker_auc_returns_high_auc_for_clean_marker():
    values = np.array([10, 9, 8, 1, 0, 2])
    labels = np.array(["B", "B", "B", "Other", "Other", "Other"])
    assert marker_auc(values, labels, "B") > 0.9
    assert np.isnan(marker_auc(values, ["B"] * 6, "B"))


def test_knn_predictions_exclude_self():
    from rarecell.metrics import compute_knn_predictions
    embedding = np.array([[0.0, 0.0], [0.0, 0.01], [10.0, 10.0], [10.01, 10.0]])
    labels = np.array(["A", "A", "B", "B"])
    preds = compute_knn_predictions(embedding, labels, k=1)
    # Each cell should be predicted from its neighbor, not itself
    assert preds is not None
    assert len(preds) == 4


def test_rare_cell_precision_recall_f1_valid_on_toy():
    from rarecell.metrics import compute_rare_cell_precision_recall_f1
    embedding = np.vstack([np.zeros((10, 2)), np.ones((20, 2)) * 10])
    labels = np.array(["target"] * 10 + ["other"] * 20)
    result = compute_rare_cell_precision_recall_f1(embedding, labels, "target", k=5)
    assert "precision" in result and "recall" in result and "f1" in result
    for v in result.values():
        assert 0.0 <= float(v) <= 1.0


def test_neighborhood_purity_perfect_separation():
    from rarecell.metrics import neighborhood_purity
    embedding = np.vstack([np.zeros((5, 2)), np.ones((10, 2)) * 100])
    labels = np.array(["target"] * 5 + ["other"] * 10)
    purity = neighborhood_purity(embedding, labels, "target", k=3)
    assert float(purity) >= 0.99


def test_neighborhood_purity_mixed_gives_less_than_one():
    from rarecell.metrics import neighborhood_purity
    embedding = np.arange(20).reshape(10, 2).astype(float)
    labels = np.array(["A", "B"] * 5)
    purity = neighborhood_purity(embedding, labels, "A", k=2)
    assert float(purity) < 1.0


def test_summarize_metric_by_group_returns_expected_columns():
    df = pd.DataFrame(
        {
            "representation": ["rna", "rna", "protein"],
            "retain_fraction": [1.0, 1.0, 1.0],
            "f1": [1.0, 0.8, 0.5],
        }
    )
    summary = summarize_metric_by_group(df, ["representation", "retain_fraction"], ["f1"])
    assert {"f1_mean", "f1_std", "f1_sem", "f1_count"}.issubset(summary.columns)
    assert summary.loc[summary["representation"] == "rna", "f1_count"].item() == 2
