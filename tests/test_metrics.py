import numpy as np

from rarecell.metrics import (
    compute_all_metrics,
    compute_knn_predictions,
    compute_rare_cell_precision_recall_f1,
    macro_f1_from_embedding,
    neighborhood_purity,
    rare_cell_classification_metrics,
    target_silhouette_score,
)


def test_perfect_separation_gives_high_recovery_metrics():
    embedding = np.vstack([np.zeros((12, 2)), np.ones((18, 2)) * 10])
    labels = np.array(["B"] * 12 + ["Other"] * 18)
    metrics = rare_cell_classification_metrics(embedding, labels, "B", seed=0, n_neighbors=3)
    assert metrics["f1"] >= 0.9
    assert metrics["recall"] >= 0.9
    assert macro_f1_from_embedding(embedding, labels, seed=0, n_neighbors=3) >= 0.9
    assert neighborhood_purity(embedding, labels, "B", k=5) >= 0.9


def test_knn_predictions_exclude_self_and_recover_target():
    embedding = np.array([[0.0, 0.0], [0.0, 0.01], [10.0, 10.0], [10.01, 10.0]])
    labels = np.array(["A", "A", "B", "B"])
    preds = compute_knn_predictions(embedding, labels, k=1)
    assert len(preds) == 4
    assert set(preds) == {"A", "B"}
    result = compute_rare_cell_precision_recall_f1(embedding, labels, "A", k=1)
    assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_metric_edge_cases_return_nan_instead_of_crashing():
    embedding = np.random.default_rng(0).normal(size=(6, 2))
    metrics = rare_cell_classification_metrics(embedding, ["Other"] * 6, "B")
    assert np.isnan(metrics["f1"])
    assert metrics["n_positive"] == 0
    assert np.isnan(target_silhouette_score(embedding, ["Other"] * 6, "B"))

    bad_embedding = np.array([[0.0, 0.0], [np.nan, 1.0], [2.0, 2.0], [3.0, 3.0]])
    bad_labels = np.array(["B", "B", "Other", "Other"])
    all_metrics = compute_all_metrics(bad_embedding, bad_labels, "B", seed=0, n_neighbors=2)
    assert np.isnan(all_metrics["f1"])
    assert np.isnan(all_metrics["neighborhood_purity"])


def test_compute_all_metrics_includes_counts_and_core_metrics():
    embedding = np.vstack([np.zeros((6, 2)), np.ones((10, 2)) * 5])
    labels = np.array(["target"] * 6 + ["other"] * 10)
    result = compute_all_metrics(embedding, labels, "target", seed=0, n_neighbors=3)
    expected = {
        "precision",
        "recall",
        "f1",
        "neighborhood_purity",
        "target_silhouette",
        "n_cells",
        "n_target",
        "n_other",
        "target_fraction",
    }
    assert expected.issubset(result)
    assert result["n_cells"] == 16
    assert result["n_target"] == 6
