"""Small shared utilities for matrix handling and outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

LIKELY_LABEL_COLUMNS = (
    "cell_type",
    "celltype",
    "cell.types",
    "annotation",
    "label",
    "predicted.celltype",
    "cell_type_l1",
    "cell_type_l2",
    "leiden",
    "louvain",
)


def ensure_directory(path: str | Path) -> Path:
    """Create and return a directory path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def to_dense_array(matrix: Any) -> np.ndarray:
    """Convert sparse, pandas, or array-like matrices to a dense NumPy array."""
    if isinstance(matrix, pd.DataFrame):
        return matrix.to_numpy()
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def matrix_to_dataframe(
        matrix: Any,
        index: pd.Index | list[str] | None = None,
        columns: pd.Index | list[str] | None = None,
        prefix: str = "feature",
) -> pd.DataFrame:
    """Convert a matrix-like object to a DataFrame with stable names."""
    if isinstance(matrix, pd.DataFrame):
        return matrix.copy()
    values = to_dense_array(matrix)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {values.shape}.")
    if index is None:
        index = [f"cell_{i}" for i in range(values.shape[0])]
    if columns is None:
        columns = [f"{prefix}_{i + 1}" for i in range(values.shape[1])]
    return pd.DataFrame(values, index=pd.Index(index), columns=pd.Index(columns))


def safe_n_components(requested: int, n_obs: int, n_features: int) -> int:
    """Return a PCA component count that is valid for small matrices."""
    max_components = max(1, min(n_obs, n_features) - 1)
    return max(1, min(int(requested), max_components))


def component_names(prefix: str, n_components: int) -> list[str]:
    """Create stable embedding component names."""
    return [f"{prefix}{i + 1}" for i in range(n_components)]


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    """Write JSON with deterministic indentation."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)
    return out


def infer_label_column(adata: Any) -> str | None:
    """Return the best available cell-label column from ``adata.obs``.

    Known biological annotation names are preferred over clustering columns.
    Returns ``None`` when no plausible label column is present.
    """
    if not hasattr(adata, "obs"):
        raise ValueError("Expected an AnnData-like object with an .obs table.")
    obs = adata.obs
    lower_to_column = {str(column).lower(): str(column) for column in obs.columns}
    for candidate in LIKELY_LABEL_COLUMNS:
        match = lower_to_column.get(candidate.lower())
        if match is not None:
            return match

    for column in obs.columns:
        series = obs[column]
        if pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
            n_unique = series.nunique(dropna=True)
            if 1 < n_unique <= min(100, max(2, len(series) // 2)):
                return str(column)
    return None
