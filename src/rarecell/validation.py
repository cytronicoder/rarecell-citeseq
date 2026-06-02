"""AnnData and results-table validation helpers for the benchmark pipeline."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

_RESULT_COLUMNS = [
    "target_label",
    "label_key",
    "retain_fraction",
    "seed",
    "representation",
    "n_neighbors",
    "n_cells",
    "n_target",
    "n_other",
    "target_fraction",
    "precision",
    "recall",
    "f1",
    "neighborhood_purity",
    "target_silhouette",
]

_NUMERIC_COLUMNS = [
    "retain_fraction",
    "seed",
    "n_neighbors",
    "n_cells",
    "n_target",
    "n_other",
    "target_fraction",
    "precision",
    "recall",
    "f1",
    "neighborhood_purity",
    "target_silhouette",
]

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


def validate_adata_fields(
        adata,
        label_key: str | None = None,
        representation_keys: list[str] | None = None,
) -> None:
    """Validate expected AnnData fields and raise clear errors."""
    if not hasattr(adata, "obs"):
        raise ValueError("Expected an AnnData-like object with an .obs table.")
    if label_key is not None and label_key not in adata.obs:
        available = list(adata.obs.columns)
        raise KeyError(
            f"Label key '{label_key}' is not present in adata.obs. "
            f"Available columns: {available}."
        )
    if representation_keys:
        available_obsm = list(getattr(adata, "obsm", {}).keys())
        missing = [k for k in representation_keys if k not in available_obsm]
        if missing:
            raise ValueError(
                f"Missing required adata.obsm representations: {missing}. "
                f"Available: {available_obsm or '<none>'}."
            )


def find_label_key(adata, candidates: Iterable[str]) -> str:
    """Return first candidate label key found in adata.obs.

    Raise ValueError if none found.
    """
    candidate_list = list(candidates)
    for key in candidate_list:
        if key in adata.obs:
            return key
    raise ValueError(
        f"No label key found. Tried: {candidate_list}. "
        f"Available columns: {list(adata.obs.columns)}."
    )


def resolve_label_key(
        adata,
        preferred: str | None = None,
        candidates: Iterable[str] = DEFAULT_LABEL_KEY_CANDIDATES,
) -> str:
    """Return preferred label key when present, otherwise the first available candidate."""
    ordered: list[str] = []
    for key in [preferred, *list(candidates)]:
        if key and key not in ordered:
            ordered.append(key)
    return find_label_key(adata, ordered)


def resolve_target_label(adata, label_key: str, preferred: str | None = None) -> str:
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


def resolve_representation_keys(adata, requested_keys: list[str]) -> list[str]:
    """Validate that all requested keys exist in adata.obsm."""
    resolved: list[str] = []
    for key in requested_keys:
        if key in adata.obsm:
            resolved.append(key)
        else:
            raise KeyError(
                f"Representation '{key}' is missing from adata.obsm. "
                f"Available: {list(adata.obsm.keys())}."
            )
    return resolved


def validate_results_table(
        results_df: pd.DataFrame,
        representation_keys: list[str],
        retain_fractions: list[float],
        seeds: list[int],
) -> None:
    """Confirm benchmark results contain expected rows and required metric columns."""
    assert not results_df.empty, "Benchmark results table is empty."

    missing = [c for c in _RESULT_COLUMNS if c not in results_df.columns]
    assert not missing, f"Results table missing columns: {missing}."

    obs_reps = set(results_df["representation"].astype(str))
    exp_reps = set(map(str, representation_keys))
    assert exp_reps.issubset(obs_reps), (
        f"Missing representation rows. Expected {sorted(exp_reps)}, "
        f"observed {sorted(obs_reps)}."
    )

    obs_fracs = set(results_df["retain_fraction"].astype(float))
    exp_fracs = set(map(float, retain_fractions))
    assert exp_fracs.issubset(obs_fracs), (
        f"Missing retain_fraction rows. Expected {sorted(exp_fracs)}, "
        f"observed {sorted(obs_fracs)}."
    )

    obs_seeds = set(results_df["seed"].astype(int))
    exp_seeds = set(map(int, seeds))
    assert exp_seeds.issubset(obs_seeds), (
        f"Missing seed rows. Expected {sorted(exp_seeds)}, "
        f"observed {sorted(obs_seeds)}."
    )

    non_numeric = [
        c
        for c in _NUMERIC_COLUMNS
        if c in results_df and not is_numeric_dtype(results_df[c])
    ]
    assert not non_numeric, (
        f"Expected numeric metric/count columns, found non-numeric: {non_numeric}."
    )

    ordered = (
        results_df.groupby(["representation", "seed", "retain_fraction"], dropna=False)["n_target"]
        .max()
        .reset_index()
    )
    for (rep, seed), group in ordered.groupby(["representation", "seed"], dropna=False):
        group = group.sort_values("retain_fraction")
        counts = group["n_target"].to_numpy(dtype=float)
        if counts.size > 1:
            assert bool(np.all(np.diff(counts) >= 0)), (
                f"n_target should not increase as retain_fraction decreases "
                f"for representation={rep}, seed={seed}."
            )
