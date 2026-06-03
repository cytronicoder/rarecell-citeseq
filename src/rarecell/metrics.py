"""Benchmark metrics for rare-cell preservation experiments."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from .utils import to_dense_array


def _as_2d_array(embedding: Any) -> np.ndarray:
    values = to_dense_array(embedding).astype(float)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D embedding, got shape {values.shape}.")
    return values


def _labels_have_nan(labels: Any) -> bool:
    return bool(pd.isna(pd.Series(labels)).any())


def _invalid_embedding_or_labels(values: np.ndarray, labels: Any) -> bool:
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        return True
    if not np.isfinite(values).all():
        return True
    if len(labels) == 0 or _labels_have_nan(labels):
        return True
    return False


def _nan_classification_result(n_positive: int, n_negative: int) -> dict[str, float | int]:
    return {
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "macro_f1": np.nan,
        "support": np.nan,
        "n_positive": int(n_positive),
        "n_negative": int(n_negative),
        "n_train": 0,
        "n_test": 0,
    }


def rare_cell_classification_metrics(
        embedding,
        labels,
        target_label: str,
        test_size: float = 0.3,
        seed: int = 0,
        n_neighbors: int = 15,
) -> dict:
    """Evaluate target-vs-other KNN classification from an embedding."""
    values = _as_2d_array(embedding)
    if values.shape[0] != len(labels):
        raise ValueError("Embedding rows and labels must have the same length.")
    if _invalid_embedding_or_labels(values, labels):
        return _nan_classification_result(0, 0)
    label_values = np.asarray(labels).astype(str)
    if values.shape[0] != label_values.shape[0]:
        raise ValueError("Embedding rows and labels must have the same length.")

    y = (label_values == str(target_label)).astype(int)
    n_positive = int(y.sum())
    n_negative = int(y.shape[0] - n_positive)
    if n_positive < 2 or n_negative < 2 or len(np.unique(label_values)) < 2:
        return _nan_classification_result(n_positive, n_negative)

    stratify = y if min(n_positive, n_negative) >= 2 else None
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            values,
            y,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        x_train, x_test, y_train, y_test = train_test_split(
            values,
            y,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )

    if x_train.shape[0] < 2 or x_test.shape[0] < 1 or len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        result = _nan_classification_result(n_positive, n_negative)
        result["n_train"] = int(y_train.shape[0])
        result["n_test"] = int(y_test.shape[0])
        result["support"] = int(y_test.sum())
        return result

    k = min(max(1, int(n_neighbors)), int(x_train.shape[0]))
    classifier = KNeighborsClassifier(n_neighbors=k)
    try:
        classifier.fit(x_train, y_train)
        predicted = classifier.predict(x_test)
        macro = macro_f1_from_embedding(
            values,
            label_values,
            test_size=test_size,
            seed=seed,
            n_neighbors=n_neighbors,
        )
    except ValueError:
        result = _nan_classification_result(n_positive, n_negative)
        result["n_train"] = int(y_train.shape[0])
        result["n_test"] = int(y_test.shape[0])
        result["support"] = int(y_test.sum())
        return result
    return {
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "macro_f1": float(macro),
        "support": int(y_test.sum()),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_train": int(y_train.shape[0]),
        "n_test": int(y_test.shape[0]),
    }


def macro_f1_from_embedding(
        embedding,
        labels,
        test_size: float = 0.3,
        seed: int = 0,
        n_neighbors: int = 15,
) -> float:
    """Return multiclass macro-F1 from a KNN classifier on the embedding."""
    values = _as_2d_array(embedding)
    if values.shape[0] != len(labels):
        raise ValueError("Embedding rows and labels must have the same length.")
    if _invalid_embedding_or_labels(values, labels):
        return float(np.nan)
    label_values = np.asarray(labels).astype(str)
    classes, counts = np.unique(label_values, return_counts=True)
    if len(classes) < 2 or int(counts.min()) < 2 or values.shape[0] < 3:
        return float(np.nan)
    stratify = label_values if int(counts.min()) >= 2 else None
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            values,
            label_values,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        return float(np.nan)
    if x_train.shape[0] < 1 or x_test.shape[0] < 1 or len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return float(np.nan)
    k = min(max(1, int(n_neighbors)), int(x_train.shape[0]))
    try:
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(x_train, y_train)
        predicted = classifier.predict(x_test)
        return float(f1_score(y_test, predicted, average="macro", zero_division=0))
    except ValueError:
        return float(np.nan)


def neighborhood_purity(
        embedding,
        labels,
        target_label: str,
        k: int = 15,
) -> float:
    """Return mean target-label fraction among target-cell nearest neighbors."""
    values = _as_2d_array(embedding)
    if values.shape[0] != len(labels):
        raise ValueError("Embedding rows and labels must have the same length.")
    if _invalid_embedding_or_labels(values, labels):
        return float(np.nan)
    label_values = np.asarray(labels).astype(str)
    if values.shape[0] != label_values.shape[0]:
        raise ValueError("Embedding rows and labels must have the same length.")
    target_mask = label_values == str(target_label)
    if int(k) < 1 or int(target_mask.sum()) < 2 or values.shape[0] < 2:
        return float(np.nan)

    n_neighbors = min(max(2, int(k) + 1), values.shape[0])
    try:
        nearest = NearestNeighbors(n_neighbors=n_neighbors)
        nearest.fit(values)
        _, indices = nearest.kneighbors(values[target_mask])
        neighbor_indices = indices[:, 1:]
        if neighbor_indices.shape[1] < 1:
            return float(np.nan)
        purities = (label_values[neighbor_indices] == str(target_label)).mean(axis=1)
        return float(np.mean(purities))
    except ValueError:
        return float(np.nan)


def target_silhouette_score(
        embedding,
        labels,
        target_label: str,
) -> float:
    """Return binary target-vs-other silhouette score."""
    values = _as_2d_array(embedding)
    if values.shape[0] != len(labels):
        raise ValueError("Embedding rows and labels must have the same length.")
    if _invalid_embedding_or_labels(values, labels):
        return float(np.nan)
    label_values = np.asarray(labels).astype(str)
    if values.shape[0] != label_values.shape[0]:
        raise ValueError("Embedding rows and labels must have the same length.")
    y = (label_values == str(target_label)).astype(int)
    n_positive = int(y.sum())
    n_negative = int(y.shape[0] - n_positive)
    if len(np.unique(y)) < 2 or n_positive < 2 or n_negative < 2 or values.shape[0] < 3:
        return float(np.nan)
    try:
        return float(silhouette_score(values, y))
    except ValueError:
        return float(np.nan)


def compute_knn_predictions(embedding, labels, k: int = 15) -> np.ndarray:
    """Predict each cell label from its nearest neighbors in the embedding."""
    values = _as_2d_array(embedding)
    label_values = np.asarray(labels).astype(str)
    if values.shape[0] != label_values.shape[0]:
        raise ValueError("Embedding rows and labels must have the same length.")
    if _invalid_embedding_or_labels(values, label_values) or values.shape[0] < 2:
        warnings.warn("kNN predictions are undefined for this embedding/label input.")
        return np.full(label_values.shape, None, dtype=object)

    n_neighbors = min(max(2, int(k) + 1), values.shape[0])
    nearest = NearestNeighbors(n_neighbors=n_neighbors)
    nearest.fit(values)
    _, indices = nearest.kneighbors(values)
    neighbor_indices = indices[:, 1:]
    predictions: list[str | None] = []
    for row in neighbor_indices:
        if row.size == 0:
            predictions.append(None)
            continue
        counts = pd.Series(label_values[row]).value_counts()
        predictions.append(str(counts.index[0]))
    return np.asarray(predictions, dtype=object)


def compute_rare_cell_precision_recall_f1(
        embedding,
        labels,
        target_label: str,
        k: int = 15,
) -> dict[str, float]:
    """Compute target-cell precision, recall, and F1 from kNN label predictions."""
    label_values = np.asarray(labels).astype(str)
    if label_values.size == 0 or int((label_values == str(target_label)).sum()) < 1:
        warnings.warn("Rare-cell precision/recall/F1 are undefined without target cells.")
        return {"precision": float(np.nan), "recall": float(np.nan), "f1": float(np.nan)}
    predictions = compute_knn_predictions(embedding, label_values, k=k)
    if pd.isna(pd.Series(predictions)).any():
        return {"precision": float(np.nan), "recall": float(np.nan), "f1": float(np.nan)}
    y_true = label_values == str(target_label)
    y_pred = predictions.astype(str) == str(target_label)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def class_balance_metrics(labels, target_label: str) -> dict[str, float | int]:
    """Return n_cells, n_target, n_other, and target_fraction for target_label."""
    if labels is None:
        return {
            "n_cells": 0,
            "n_target": 0,
            "n_other": 0,
            "target_fraction": float(np.nan),
        }
    series = pd.Series(labels)
    if series.empty or _labels_have_nan(series):
        return {
            "n_cells": 0,
            "n_target": 0,
            "n_other": 0,
            "target_fraction": float(np.nan),
        }
    label_values = series.astype(str).to_numpy()
    n_cells = int(label_values.shape[0])
    n_target = int((label_values == str(target_label)).sum())
    n_other = int(n_cells - n_target)
    return {
        "n_cells": n_cells,
        "n_target": n_target,
        "n_other": n_other,
        "target_fraction": float(n_target / n_cells) if n_cells else float(np.nan),
    }


def compute_all_metrics(
        embedding,
        labels,
        target_label: str,
        seed: int,
        n_neighbors: int = 15,
) -> dict[str, float | int]:
    """Compute benchmark metrics for one embedding."""
    split_result = rare_cell_classification_metrics(
        embedding,
        labels,
        target_label,
        seed=seed,
        n_neighbors=n_neighbors,
    )
    result = compute_rare_cell_precision_recall_f1(
        embedding,
        labels,
        target_label,
        k=n_neighbors,
    )
    result["macro_f1"] = split_result.get("macro_f1", float(np.nan))
    result["support"] = split_result.get("support", float(np.nan))
    result["n_train"] = split_result.get("n_train", 0)
    result["n_test"] = split_result.get("n_test", 0)
    result["n_positive"] = split_result.get("n_positive", 0)
    result["n_negative"] = split_result.get("n_negative", 0)
    result["neighborhood_purity"] = neighborhood_purity(
        embedding,
        labels,
        target_label,
        k=n_neighbors,
    )
    result["target_silhouette"] = target_silhouette_score(embedding, labels, target_label)
    result["silhouette_target"] = result["target_silhouette"]
    result.update(class_balance_metrics(labels, target_label))
    return result
