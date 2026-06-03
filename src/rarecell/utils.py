"""Shared utilities: matrix helpers, naming, script setup, and validation."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from .config import ANNDATA_KEY_TO_REPRESENTATION, REPRESENTATION_ALIASES, REPRESENTATION_KEY_MAP

# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------

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
    """Return the best available cell-label column from ``adata.obs``."""
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


# ---------------------------------------------------------------------------
# Output naming helpers (from naming.py)
# ---------------------------------------------------------------------------


def slugify_label(label: str) -> str:
    """Convert cell-type labels to safe lowercase snake_case filename slugs."""
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(label))
    return text.strip("_").lower()


def make_output_prefix(dataset_name: str, target_label: str) -> str:
    """Return a canonical output prefix: {dataset}_{target_slug}."""
    return f"{slugify_label(dataset_name)}_{slugify_label(target_label)}"


# ---------------------------------------------------------------------------
# Script setup helpers (from script_utils.py)
# ---------------------------------------------------------------------------


def setup_file_logger(log_path: str | Path) -> Path:
    """Configure logging to stdout and a script-specific file."""
    out = Path(log_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(out, mode="w"), logging.StreamHandler()],
        force=True,
    )
    logging.info("Started at %s", datetime.now().isoformat(timespec="seconds"))
    logging.info("Log file: %s", out)
    return out


def standard_representation_name(value: str) -> str:
    """Return the user-facing representation name for an alias or AnnData key."""
    raw = str(value)
    key = REPRESENTATION_ALIASES.get(raw, raw)
    return ANNDATA_KEY_TO_REPRESENTATION.get(key, raw)


def resolve_representation_key(adata: Any, representation: str) -> str:
    """Resolve a user-facing representation name to an existing AnnData obsm key."""
    canonical_key = REPRESENTATION_ALIASES.get(str(representation), str(representation))
    if canonical_key in adata.obsm:
        return canonical_key
    if str(representation) == "joint_pca":
        for legacy_key in ("X_joint_simple", "X_joint"):
            if legacy_key in adata.obsm:
                return legacy_key
    raise KeyError(f"Representation '{representation}' is missing from adata.obsm as '{canonical_key}'.")


def resolve_representations(adata: Any, representations: Iterable[str] | None = None) -> list[tuple[str, str]]:
    """Return (standard_name, obsm_key) pairs for requested or available representations."""
    requested = list(representations) if representations else list(REPRESENTATION_KEY_MAP)
    resolved: list[tuple[str, str]] = []
    missing: list[str] = []
    for name in requested:
        try:
            key = resolve_representation_key(adata, name)
            resolved.append((standard_representation_name(name), key))
        except KeyError:
            missing.append(str(name))
    if representations and missing:
        raise KeyError(f"Representation keys are missing from adata.obsm: {missing}")
    if not resolved:
        raise KeyError("No benchmark representations found in adata.obsm.")
    return resolved


# ---------------------------------------------------------------------------
# AnnData and results validation (from validation.py)
# ---------------------------------------------------------------------------

DEFAULT_LABEL_KEY_CANDIDATES = (
    "cell_type_simple",
    "input_label",
    "cell_type",
    "celltype",
    "cell_type_label",
    "annotation",
    "label",
    "leiden",
)


def find_label_key(adata: Any, candidates: Iterable[str]) -> str:
    """Return first candidate label key found in adata.obs."""
    candidate_list = list(candidates)
    for key in candidate_list:
        if key in adata.obs:
            return key
    raise ValueError(
        f"No label key found. Tried: {candidate_list}. "
        f"Available columns: {list(adata.obs.columns)}."
    )


def resolve_label_key(
        adata: Any,
        preferred: str | None = None,
        candidates: Iterable[str] = DEFAULT_LABEL_KEY_CANDIDATES,
) -> str:
    """Return preferred label key when present, otherwise the first available candidate."""
    ordered: list[str] = []
    for key in [preferred, *list(candidates)]:
        if key and key not in ordered:
            ordered.append(key)
    return find_label_key(adata, ordered)


def resolve_target_label(adata: Any, label_key: str, preferred: str | None = None) -> str:
    """Return preferred target label when present, otherwise the smallest available label class."""
    if label_key not in adata.obs:
        raise KeyError(f"Label key '{label_key}' is not present in adata.obs.")

    labels = adata.obs[label_key].astype(str)
    if preferred is not None and str(preferred) in set(labels):
        return str(preferred)

    counts = labels.value_counts(dropna=False)
    if counts.empty:
        raise ValueError(f"Cannot resolve target label because adata.obs['{label_key}'] is empty.")

    count_table = counts.rename_axis("label").reset_index(name="n_cells")
    count_table["label"] = count_table["label"].astype(str)
    count_table = count_table.sort_values(["n_cells", "label"], kind="mergesort")
    return str(count_table.iloc[0]["label"])
