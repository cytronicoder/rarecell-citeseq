"""Deterministic target-cell downsampling utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def validate_target_label(adata: Any, label_key: str, target_label: str) -> None:
    """Validate that label_key exists and contains target_label."""
    if not hasattr(adata, "obs"):
        raise ValueError("Expected an AnnData-like object with an .obs table.")
    if label_key not in adata.obs:
        available = ", ".join(map(str, adata.obs.columns))
        raise KeyError(
            f"Label key '{label_key}' is not present in adata.obs. "
            f"Available columns: {available or '<none>'}."
        )
    labels = adata.obs[label_key].astype(str)
    if str(target_label) not in set(labels):
        counts = labels.value_counts(dropna=False).head(20).to_dict()
        raise ValueError(
            f"Target label '{target_label}' is not present in adata.obs['{label_key}']. "
            f"Observed labels include: {counts}."
        )


def _retained_target_count(n_target: int, retain_fraction: float, min_target_cells: int) -> int:
    if n_target <= 0:
        return 0
    if retain_fraction >= 1:
        return int(n_target)
    n_retain = int(np.floor(n_target * retain_fraction))
    return int(min(n_target, max(min_target_cells, n_retain)))


def downsample_target_cells(
        adata: Any,
        label_key: str | None = None,
        target_label: str | None = None,
        retain_fraction: float = 1.0,
        seed: int = 0,
        min_target_cells: int = 1,
        *,
        label_column: str | None = None,
        target_cell_type: str | None = None,
):
    """Return a copied AnnData object with only a fraction of target cells retained."""
    if label_key is None:
        label_key = label_column
    if target_label is None:
        target_label = target_cell_type
    if label_key is None:
        raise ValueError("A label_key or label_column must be provided.")
    if target_label is None:
        raise ValueError("A target_label or target_cell_type must be provided.")
    if not np.isfinite(retain_fraction) or retain_fraction <= 0 or retain_fraction > 1:
        raise ValueError("retain_fraction must be > 0 and <= 1.")
    if min_target_cells < 1:
        raise ValueError("min_target_cells must be at least 1.")
    validate_target_label(adata, label_key, target_label)

    labels = adata.obs[label_key].astype(str)
    target = str(target_label)
    target_mask = labels == target

    target_indices = np.flatnonzero(target_mask.to_numpy())
    n_target_before = int(target_indices.size)
    n_non_target = int(adata.n_obs - n_target_before)
    if n_target_before < min_target_cells:
        raise ValueError(
            f"Target population '{target_label}' has {n_target_before} cells, "
            f"fewer than min_target_cells={min_target_cells}."
        )
    if min_target_cells > n_target_before:
        raise ValueError(
            f"min_target_cells={min_target_cells} exceeds available target cells={n_target_before}."
        )

    if retain_fraction == 1:
        retained_targets = target_indices
    else:
        n_retain = _retained_target_count(n_target_before, retain_fraction, min_target_cells)
        rng = np.random.default_rng(seed)
        retained_targets = np.sort(rng.choice(target_indices, size=n_retain, replace=False))

    keep_mask = np.ones(adata.n_obs, dtype=bool)
    keep_mask[target_indices] = False
    keep_mask[retained_targets] = True
    adata_sub = adata[keep_mask, :].copy()
    retained_labels = adata_sub.obs[label_key].astype(str)
    is_target_cell = retained_labels == target
    if int(is_target_cell.sum()) < 1:
        raise ValueError("Downsampling removed all target cells.")

    adata_sub.obs["downsample_target"] = target
    adata_sub.obs["retain_fraction"] = float(retain_fraction)
    adata_sub.obs["downsample_seed"] = int(seed)
    adata_sub.obs["is_target_cell"] = is_target_cell.to_numpy(dtype=bool)

    metadata = dict(adata_sub.uns.get("downsampling", {}))
    metadata.update({
        "label_key": label_key,
        "target_label": target_label,
        "retain_fraction": float(retain_fraction),
        "seed": int(seed),
        "n_target_before": n_target_before,
        "n_target_after": int(retained_targets.size),
        "n_non_target": n_non_target,
        "n_cells_before": int(adata.n_obs),
        "n_cells_after": int(adata_sub.n_obs),
        "min_target_cells": int(min_target_cells),
    })
    adata_sub.uns["downsampling"] = metadata
    return adata_sub


def get_target_counts(adata: Any, label_key: str) -> pd.DataFrame:
    """Count labels in an AnnData `.obs` column."""
    if label_key not in adata.obs:
        raise KeyError(f"Label key '{label_key}' is not present in adata.obs.")
    counts = adata.obs[label_key].astype(str).value_counts(dropna=False)
    total = max(1, int(counts.sum()))
    return pd.DataFrame(
        {
            "label": counts.index.astype(str),
            "n_cells": counts.to_numpy(dtype=int),
            "fraction": counts.to_numpy(dtype=float) / total,
        }
    )


def summarize_downsampling_grid(
        adata: Any,
        label_key: str,
        target_label: str,
        retain_fractions: list[float],
        seeds: list[int],
        min_target_cells: int = 1,
) -> pd.DataFrame:
    """Return expected target-cell counts for each retain_fraction and seed."""
    validate_target_label(adata, label_key, target_label)
    if min_target_cells < 1:
        raise ValueError("min_target_cells must be at least 1.")
    labels = adata.obs[label_key].astype(str)
    n_target = int((labels == str(target_label)).sum())
    n_other = int(len(labels) - n_target)

    rows = []
    for retain_fraction in retain_fractions:
        if not np.isfinite(retain_fraction) or retain_fraction <= 0 or retain_fraction > 1:
            raise ValueError("retain_fractions must contain values > 0 and <= 1.")
        retained = _retained_target_count(n_target, float(retain_fraction), min_target_cells)
        for seed in seeds:
            rows.append(
                {
                    "label_key": str(label_key),
                    "target_label": str(target_label),
                    "retain_fraction": float(retain_fraction),
                    "seed": int(seed),
                    "original_n_cells": int(len(labels)),
                    "original_n_target": n_target,
                    "expected_retained_n_target": retained,
                    "expected_n_other": n_other,
                    "expected_n_cells": int(n_other + retained),
                }
            )
    return pd.DataFrame(rows)
