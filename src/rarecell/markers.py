"""Marker lookup and marker-AUC helpers for toy CITE-seq benchmarks."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .metrics import marker_auc
from .utils import to_dense_array

DEFAULT_MARKERS = {
    "B cell": {
        "rna": ["MS4A1", "CD79A", "CD79B", "BANK1", "CD74"],
        "protein": ["CD19", "CD20", "CD45RA", "HLA-DR"],
    },
    "NK cell": {
        "rna": ["NKG7", "GNLY", "KLRD1", "GZMB", "PRF1"],
        "protein": ["CD56", "CD16", "CD335"],
    },
    "CD4 T cell": {
        "rna": ["CD3D", "CD3E", "CD4", "IL7R", "CCR7", "LTB"],
        "protein": ["CD3", "CD4", "CD45RA", "CD45RO", "CD27"],
    },
    "CD8 T cell": {
        "rna": ["CD3D", "CD3E", "CD8A", "CD8B", "NKG7", "GZMK"],
        "protein": ["CD3", "CD8", "CD45RA", "CD45RO"],
    },
    "T cell": {
        "rna": ["CD3D", "CD3E", "TRAC", "LTB"],
        "protein": ["CD3"],
    },
    "CD14 monocyte": {
        "rna": ["LYZ", "S100A8", "S100A9", "FCN1", "LST1", "CTSS"],
        "protein": ["CD14", "CD11b", "HLA-DR"],
    },
    "FCGR3A monocyte": {
        "rna": ["FCGR3A", "MS4A7", "LST1", "IFITM3"],
        "protein": ["CD16", "CD11b", "HLA-DR"],
    },
    "Dendritic cell": {
        "rna": ["FCER1A", "CST3", "CLEC10A", "LILRA4"],
        "protein": ["CD11c", "HLA-DR", "CD123"],
    },
    "pDC": {
        "rna": ["GZMB", "IRF7", "LILRA4", "TCF4", "IL3RA"],
        "protein": ["CD123", "HLA-DR"],
    },
    "Platelet": {
        "rna": ["PPBP", "PF4", "NRGN"],
        "protein": ["CD41", "CD61"],
    },
}

TARGET_MARKER_ALIASES = {
    "B": "B cell",
    "B CELL": "B cell",
    "B CELLS": "B cell",
    "NK": "NK cell",
    "NK CELL": "NK cell",
    "NK CELLS": "NK cell",
    "CD4 T": "CD4 T cell",
    "CD4 T CELL": "CD4 T cell",
    "CD4 T CELLS": "CD4 T cell",
    "CD8 T": "CD8 T cell",
    "CD8 T CELL": "CD8 T cell",
    "CD8 T CELLS": "CD8 T cell",
    "T CELL": "T cell",
    "T CELLS": "T cell",
    "MONOCYTE": "CD14 monocyte",
    "MONOCYTES": "CD14 monocyte",
    "CD14 MONOCYTE": "CD14 monocyte",
    "CD14 MONOCYTES": "CD14 monocyte",
    "FCGR3A MONOCYTE": "FCGR3A monocyte",
    "CD16 MONOCYTE": "FCGR3A monocyte",
    "DENDRITIC CELL": "Dendritic cell",
    "DENDRITIC CELLS": "Dendritic cell",
    "DC": "Dendritic cell",
    "CDC": "Dendritic cell",
    "PDC": "pDC",
    "PLASMACYTOID DENDRITIC CELL": "pDC",
    "PLATELET": "Platelet",
    "PLATELETS": "Platelet",
}

MARKER_AUC_COLUMNS = [
    "target_label",
    "label_key",
    "modality",
    "marker",
    "matched_feature_name",
    "auc",
    "n_target",
    "n_other",
    "marker_source",
    "match_type",
    "match_reason",
]

MARKER_MATCHING_DIAGNOSTIC_COLUMNS = [
    "target_label",
    "marker_source",
    "modality",
    "requested_marker",
    "matched_feature_name",
    "match_type",
    "status",
    "reason",
]


def normalize_feature_name(name: str) -> str:
    """Normalize feature names for robust RNA/protein marker matching."""
    text = str(name).strip().upper()
    text = re.sub(r"^(ADT|PROT|PROTEIN)[_\-\s]+", "", text)
    text = re.sub(r"^(TOTALSEQ[A-Z]*|TOTALSEQ[A-Z]*)[_\-\s]+", "", text)
    text = re.sub(r"^(ANTI[\-\s]+HUMAN|ANTI[\-\s]+MOUSE|ANTI)[_\-\s]+", "", text)
    text = re.sub(r"[_\-\s]+(TOTALSEQ[A-Z]*|TOTALSEQ[A-Z]*|ANTIBODY|PROT|PROTEIN|ADT)$", "", text)
    text = re.sub(r"[\(\)\[\]\{\}:;/,]+", " ", text)
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _compact_normalized_name(name: str) -> str:
    return normalize_feature_name(name).replace(" ", "")


def _protein_tokens(name: str) -> set[str]:
    return set(normalize_feature_name(name).split())


def match_marker_to_features(marker: str, feature_names: list[str], modality: str) -> dict | None:
    """Match a requested marker to available feature names."""
    marker_text = str(marker)
    if not feature_names:
        return None

    for feature_name in feature_names:
        if str(feature_name) == marker_text:
            return {
                "marker": marker_text,
                "matched_feature_name": str(feature_name),
                "match_type": "exact",
                "score": 1.0,
                "reason": "Exact case-sensitive feature-name match.",
            }

    marker_upper = marker_text.upper()
    for feature_name in feature_names:
        if str(feature_name).upper() == marker_upper:
            return {
                "marker": marker_text,
                "matched_feature_name": str(feature_name),
                "match_type": "case_insensitive_exact",
                "score": 0.95,
                "reason": "Exact feature-name match after case normalization.",
            }

    marker_norm = _compact_normalized_name(marker_text)
    for feature_name in feature_names:
        if _compact_normalized_name(str(feature_name)) == marker_norm:
            return {
                "marker": marker_text,
                "matched_feature_name": str(feature_name),
                "match_type": "normalized_exact",
                "score": 0.9,
                "reason": "Exact match after feature-name normalization.",
            }

    if modality.lower() != "protein":
        return None

    marker_tokens = _protein_tokens(marker_text)
    for feature_name in feature_names:
        if marker_tokens and marker_tokens.issubset(_protein_tokens(str(feature_name))):
            return {
                "marker": marker_text,
                "matched_feature_name": str(feature_name),
                "match_type": "protein_token",
                "score": 0.8,
                "reason": "Protein marker matched tokenized antibody feature name.",
            }

    for feature_name in feature_names:
        feature_norm = _compact_normalized_name(str(feature_name))
        if marker_norm and (marker_norm in feature_norm or feature_norm in marker_norm):
            return {
                "marker": marker_text,
                "matched_feature_name": str(feature_name),
                "match_type": "protein_contains",
                "score": 0.65,
                "reason": "Protein marker matched by permissive normalized substring search.",
            }

    return None


def canonical_target_label(target_label: str, marker_dict: dict[str, Any] = DEFAULT_MARKERS) -> str | None:
    """Map target label aliases to canonical biological marker dictionary keys."""
    raw = str(target_label).strip()
    if raw in marker_dict:
        return raw
    normalized = re.sub(r"[_\-]+", " ", raw).upper().strip()
    normalized = re.sub(r"^CLUSTER\s+", "", normalized)
    if raw.lower().startswith("cluster") or normalized.isdigit():
        return None
    return TARGET_MARKER_ALIASES.get(normalized)


def _markers_for_target(target_label: str, marker_dict: dict[str, Any]) -> tuple[str | None, dict[str, list[str]]]:
    canonical = canonical_target_label(target_label, marker_dict)
    if canonical is None:
        return None, {"rna": [], "protein": []}
    return canonical, marker_dict.get(canonical, {"rna": [], "protein": []})


def get_rna_feature_names(adata) -> list[str]:
    """Return RNA feature names from an AnnData object."""
    return [str(name) for name in adata.var_names]


def get_protein_feature_names(adata) -> list[str]:
    """Return protein feature names from known AnnData storage locations."""
    for key in ("protein_names", "adt_names", "protein_features"):
        if key in adata.uns:
            return [str(value) for value in adata.uns[key]]
    for key in ("protein_counts", "protein_expression", "adt"):
        if key in adata.obsm and isinstance(adata.obsm[key], pd.DataFrame):
            return [str(value) for value in adata.obsm[key].columns]
    for key in ("protein_counts", "protein_expression", "adt"):
        if key in adata.obsm:
            n_features = int(adata.obsm[key].shape[1]) if len(adata.obsm[key].shape) == 2 else 0
            return [f"protein_{index}" for index in range(n_features)]
    return []


def protein_name_source(adata) -> str:
    """Return where protein names were read from, or why they are unavailable."""
    for key in ("protein_names", "adt_names", "protein_features"):
        if key in adata.uns:
            return f"adata.uns['{key}']"
    for key in ("protein_counts", "protein_expression", "adt"):
        if key in adata.obsm and isinstance(adata.obsm[key], pd.DataFrame):
            return f"adata.obsm['{key}'].columns"
    for key in ("protein_counts", "protein_expression", "adt"):
        if key in adata.obsm:
            return f"generated names for adata.obsm['{key}']"
    return "not_available"


def _feature_lookup(adata: Any) -> list[str]:
    names = get_rna_feature_names(adata)
    if "original_feature_name" in adata.var:
        names.extend(str(value) for value in adata.var["original_feature_name"].astype(str))
    return list(dict.fromkeys(names))


def _rna_feature_to_var_name(adata: Any, matched_feature_name: str) -> str:
    if matched_feature_name in set(map(str, adata.var_names)):
        return matched_feature_name
    if "original_feature_name" in adata.var:
        matches = adata.var.index[adata.var["original_feature_name"].astype(str) == str(matched_feature_name)]
        if len(matches) > 0:
            return str(matches[0])
    return matched_feature_name


def _protein_matrix_and_names(adata: Any) -> tuple[Any | None, list[str], str]:
    protein_keys = ("protein_counts", "protein_expression", "adt")
    for key in protein_keys:
        if key not in adata.obsm:
            continue
        matrix = adata.obsm[key]
        names = get_protein_feature_names(adata)
        if isinstance(matrix, pd.DataFrame):
            if not names:
                names = [str(value) for value in matrix.columns]
            return matrix, names, protein_name_source(adata)
        if not names:
            return matrix, [], "not_available"
        if len(names) != matrix.shape[1]:
            return matrix, [], f"{protein_name_source(adata)} length mismatch"
        return matrix, names, protein_name_source(adata)
    return None, [], "not_available"


def _one_feature_vector(matrix: Any, column_index: int | None = None) -> np.ndarray:
    if isinstance(matrix, pd.DataFrame):
        values = matrix.to_numpy() if column_index is None else matrix.iloc[:, column_index].to_numpy()
        return np.asarray(values).reshape(-1)
    if column_index is None:
        values = to_dense_array(matrix)
    else:
        values = to_dense_array(matrix[:, column_index])
    return np.asarray(values).reshape(-1)


def _mean_by_mask(matrix: Any, mask: np.ndarray) -> np.ndarray:
    subset = matrix[mask]
    if sparse.issparse(subset):
        return np.asarray(subset.mean(axis=0)).reshape(-1)
    return np.asarray(subset).mean(axis=0).reshape(-1)


def _diagnostic_row(
        target_label: str,
        marker_source: str,
        modality: str,
        requested_marker: str,
        matched_feature_name: str,
        match_type: str,
        status: str,
        reason: str,
) -> dict:
    return {
        "target_label": str(target_label),
        "marker_source": marker_source,
        "modality": modality,
        "requested_marker": str(requested_marker),
        "matched_feature_name": str(matched_feature_name),
        "match_type": match_type,
        "status": status,
        "reason": reason,
    }


def find_available_rna_markers(adata, target_label: str, marker_dict=DEFAULT_MARKERS) -> list[dict]:
    """Find requested RNA markers present in `adata.var_names` or original feature names."""
    canonical, marker_set = _markers_for_target(target_label, marker_dict)
    requested = marker_set.get("rna", [])
    features = _feature_lookup(adata)
    rows = []
    for marker in requested:
        match = match_marker_to_features(marker, features, "rna")
        if match is not None:
            match["matched_feature_name"] = _rna_feature_to_var_name(adata, match["matched_feature_name"])
            match["marker_source"] = "predefined_biological_marker"
            match["canonical_target_label"] = canonical
            rows.append(match)
    return rows


def find_available_protein_markers(adata, target_label: str, marker_dict=DEFAULT_MARKERS) -> list[dict]:
    """Find requested protein markers with robust CITE-seq feature-name matching."""
    canonical, marker_set = _markers_for_target(target_label, marker_dict)
    requested = marker_set.get("protein", [])
    _, names, _ = _protein_matrix_and_names(adata)
    if not names or all(name.startswith("protein_") for name in names):
        return []

    rows = []
    for marker in requested:
        match = match_marker_to_features(marker, names, "protein")
        if match is not None:
            match["marker_source"] = "predefined_biological_marker"
            match["canonical_target_label"] = canonical
            rows.append(match)
    return rows


def marker_matching_diagnostics(
        adata,
        target_label: str,
        marker_dict=DEFAULT_MARKERS,
        include_data_driven_note: bool = True,
) -> pd.DataFrame:
    """Return diagnostic rows explaining predefined marker matching outcomes."""
    canonical, marker_set = _markers_for_target(target_label, marker_dict)
    rows = []
    rna_features = _feature_lookup(adata)
    protein_features = get_protein_feature_names(adata)
    protein_names_are_generated = bool(protein_features) and all(
        name.startswith("protein_") for name in protein_features)

    if canonical is None:
        rows.append(
            _diagnostic_row(
                target_label,
                "predefined_biological_marker",
                "all",
                "",
                "",
                "none",
                "skipped",
                "Target label did not map to a predefined biological marker dictionary entry.",
            )
        )
    for modality, features in (("rna", rna_features), ("protein", protein_features)):
        requested = marker_set.get(modality, [])
        if modality == "protein" and protein_names_are_generated:
            rows.append(
                _diagnostic_row(
                    target_label,
                    "predefined_biological_marker",
                    "protein",
                    "",
                    "",
                    "none",
                    "skipped",
                    "Protein counts exist but interpretable protein feature names are unavailable.",
                )
            )
            continue
        for marker in requested:
            match = match_marker_to_features(marker, features, modality)
            if match is None:
                rows.append(
                    _diagnostic_row(
                        target_label,
                        "predefined_biological_marker",
                        modality,
                        marker,
                        "",
                        "none",
                        "missing",
                        f"Requested {modality.upper()} marker was not found in available feature names.",
                    )
                )
            else:
                rows.append(
                    _diagnostic_row(
                        target_label,
                        "predefined_biological_marker",
                        modality,
                        marker,
                        match["matched_feature_name"],
                        match["match_type"],
                        "matched",
                        match["reason"],
                    )
                )

    if include_data_driven_note and canonical is None:
        rows.append(
            _diagnostic_row(
                target_label,
                "data_driven_cluster_marker",
                "rna",
                "",
                "",
                "none",
                "available",
                "Target appears to require data-driven RNA marker discovery versus all other cells.",
            )
        )
    return pd.DataFrame(rows, columns=MARKER_MATCHING_DIAGNOSTIC_COLUMNS)


def compute_cluster_rna_marker_candidates(
        adata,
        label_key: str,
        target_label: str,
        n_top: int = 10,
        method: str = "wilcoxon",
) -> pd.DataFrame:
    """Compute data-driven RNA marker candidates for one cluster/cell group."""
    del method
    if label_key not in adata.obs:
        raise KeyError(f"Label key '{label_key}' is not present in adata.obs.")
    labels = adata.obs[label_key].astype(str).to_numpy()
    target_mask = labels == str(target_label)
    if int(target_mask.sum()) == 0:
        return pd.DataFrame(
            columns=[
                "target_label",
                "modality",
                "marker",
                "matched_feature_name",
                "score",
                "logfoldchange",
                "pval_adj",
                "marker_source",
            ]
        )
    other_mask = ~target_mask
    target_mean = _mean_by_mask(adata.X, target_mask)
    other_mean = _mean_by_mask(adata.X, other_mask)
    pseudocount = 1e-6
    if np.nanmin(target_mean) >= 0 and np.nanmin(other_mean) >= 0:
        logfoldchange = np.log2((target_mean + pseudocount) / (other_mean + pseudocount))
        score = logfoldchange * np.log1p(target_mean)
    else:
        logfoldchange = np.full_like(target_mean, np.nan, dtype=float)
        score = target_mean - other_mean
    score = np.nan_to_num(score, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    ranked = np.argsort(score)[::-1]
    rows = []
    for index in ranked:
        if len(rows) >= n_top:
            break
        if score[index] <= 0:
            continue
        gene = str(adata.var_names[index])
        rows.append(
            {
                "target_label": str(target_label),
                "modality": "rna",
                "marker": gene,
                "matched_feature_name": gene,
                "score": float(score[index]),
                "logfoldchange": float(logfoldchange[index]),
                "pval_adj": pd.NA,
                "marker_source": "data_driven_cluster_marker",
            }
        )
    return pd.DataFrame(rows)


def compute_marker_auc_table(
        adata,
        label_key: str,
        target_label: str,
        marker_dict=DEFAULT_MARKERS,
        allow_data_driven_cluster_markers: bool = True,
        n_data_driven_markers: int = 10,
) -> pd.DataFrame:
    """Compute available RNA/protein marker AUCs for a target label."""
    if label_key not in adata.obs:
        raise KeyError(f"Label key '{label_key}' is not present in adata.obs.")

    labels = adata.obs[label_key].astype(str).to_numpy()
    target_mask = labels == str(target_label)
    n_target = int(target_mask.sum())
    n_other = int(labels.shape[0] - n_target)
    rows = []

    for match in find_available_rna_markers(adata, target_label, marker_dict=marker_dict):
        gene = match["matched_feature_name"]
        values = _one_feature_vector(adata[:, gene].X)
        rows.append(
            {
                "target_label": str(target_label),
                "label_key": str(label_key),
                "modality": "rna",
                "marker": match["marker"],
                "matched_feature_name": gene,
                "auc": marker_auc(values, labels, str(target_label)),
                "n_target": n_target,
                "n_other": n_other,
                "marker_source": match["marker_source"],
                "match_type": match["match_type"],
                "match_reason": match["reason"],
            }
        )

    protein, names, _ = _protein_matrix_and_names(adata)
    if protein is not None and names and not all(name.startswith("protein_") for name in names):
        name_to_index = {name: index for index, name in enumerate(names)}
        for match in find_available_protein_markers(adata, target_label, marker_dict=marker_dict):
            feature = match["matched_feature_name"]
            values = _one_feature_vector(protein, name_to_index[feature])
            rows.append(
                {
                    "target_label": str(target_label),
                    "label_key": str(label_key),
                    "modality": "protein",
                    "marker": match["marker"],
                    "matched_feature_name": feature,
                    "auc": marker_auc(values, labels, str(target_label)),
                    "n_target": n_target,
                    "n_other": n_other,
                    "marker_source": match["marker_source"],
                    "match_type": match["match_type"],
                    "match_reason": match["reason"],
                }
            )

    if rows:
        return pd.DataFrame(rows, columns=MARKER_AUC_COLUMNS)

    if allow_data_driven_cluster_markers:
        candidates = compute_cluster_rna_marker_candidates(
            adata,
            label_key,
            str(target_label),
            n_top=n_data_driven_markers,
        )
        for candidate in candidates.itertuples(index=False):
            values = _one_feature_vector(adata[:, candidate.matched_feature_name].X)
            rows.append(
                {
                    "target_label": str(target_label),
                    "label_key": str(label_key),
                    "modality": "rna",
                    "marker": str(candidate.marker),
                    "matched_feature_name": str(candidate.matched_feature_name),
                    "auc": marker_auc(values, labels, str(target_label)),
                    "n_target": n_target,
                    "n_other": n_other,
                    "marker_source": "data_driven_cluster_marker",
                    "match_type": "data_driven",
                    "match_reason": "Selected as a top RNA marker for target cells versus all other cells.",
                }
            )

    return pd.DataFrame(rows, columns=MARKER_AUC_COLUMNS)
