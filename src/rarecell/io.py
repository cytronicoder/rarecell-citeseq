"""Input/output helpers for common CITE-seq object layouts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd

from .config import DATA_DIR
from .utils import matrix_to_dataframe

PROTEIN_OBSM_KEYS = ("protein_expression", "protein_counts", "adt")
PROTEIN_MODALITY_NAMES = ("prot", "protein", "adt", "modality2")
RNA_MODALITY_NAMES = ("rna", "gex", "gene_expression", "Gene Expression")
PROTEIN_FEATURE_TYPES = ("Antibody Capture", "ADT", "Protein", "protein")
RNA_FEATURE_TYPES = ("Gene Expression", "GEX", "RNA", "rna")
DEFAULT_LABEL_KEYS = (
    "cell_type",
    "celltype",
    "celltypes",
    "cell_type_label",
    "celltype_label",
    "annotation",
    "annotations",
    "label",
    "labels",
    "louvain",
    "leiden",
    "highlevel",
)

SCVI_10X_PREFIXES = ("scvi:", "10x:")
SCVI_10X_ALIASES: dict[str, str] = {
    "pbmc5k": "5k_pbmc_protein_v3_nextgem",
    "pbmc10k": "pbmc_10k_protein_v3",
}


def _is_scvi_10x_spec(value: str) -> bool:
    lower = value.lower()
    return any(lower.startswith(prefix) for prefix in SCVI_10X_PREFIXES)


def _canonical_scvi_10x_dataset_name(dataset_name: str) -> str:
    value = dataset_name.strip()
    if not value:
        raise ValueError("Expected a dataset spec like 'scvi:5k_pbmc_protein_v3_nextgem'.")
    return SCVI_10X_ALIASES.get(value.lower(), value)


def _resolve_scvi_10x_h5_path(dataset_name: str) -> Path | None:
    candidate = DATA_DIR / dataset_name / "filtered_feature_bc_matrix.h5"
    return candidate if candidate.exists() else None


def _has_file_with_optional_gzip(path: Path, basename: str) -> bool:
    return (path / basename).exists() or (path / f"{basename}.gz").exists()


def _is_10x_mex_dir(path: Path) -> bool:
    return (
            path.is_dir()
            and _has_file_with_optional_gzip(path, "matrix.mtx")
            and _has_file_with_optional_gzip(path, "barcodes.tsv")
            and (
                    _has_file_with_optional_gzip(path, "features.tsv")
                    or _has_file_with_optional_gzip(path, "genes.tsv")
            )
    )


def _load_10x_mex_dir(path: Path) -> Any:
    try:
        import muon as mu
    except ImportError as exc:
        raise ImportError("Loading local 10x MEX directories requires muon.") from exc

    obj = mu.read_10x_mtx(path)
    raw_candidate = path.parent / "raw_feature_bc_matrix"
    if hasattr(obj, "uns"):
        obj.uns["rarecell_input_path"] = str(path)
        obj.uns["rarecell_raw_feature_bc_matrix_exists"] = bool(_is_10x_mex_dir(raw_candidate))
    return obj


def _load_10x_h5(path: Path) -> ad.AnnData:
    import scanpy as sc
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Variable names are not unique.*",
            category=UserWarning,
        )
        adata = sc.read_10x_h5(path, gex_only=False)

    if "feature_types" not in adata.var:
        raise ValueError(
            f"10x HDF5 input '{path}' is missing adata.var['feature_types']; "
            "cannot distinguish RNA from antibody capture features."
        )
    observed = set(adata.var["feature_types"].astype(str))
    if "Gene Expression" not in observed:
        raise ValueError(f"10x HDF5 input '{path}' does not contain Gene Expression features.")
    if "Antibody Capture" not in observed:
        raise ValueError(f"10x HDF5 input '{path}' does not contain Antibody Capture protein/ADT features.")

    if "original_feature_name" not in adata.var:
        if "gene_symbols" in adata.var:
            adata.var["original_feature_name"] = adata.var["gene_symbols"].astype(str)
        elif "feature_name" in adata.var:
            adata.var["original_feature_name"] = adata.var["feature_name"].astype(str)
        else:
            adata.var["original_feature_name"] = adata.var_names
    _make_var_names_unique(adata)
    return adata


def _load_scvi_10x_dataset(spec: str) -> ad.AnnData:
    dataset_name = _canonical_scvi_10x_dataset_name(spec.split(":", maxsplit=1)[1])
    try:
        from scvi.data import dataset_10x
    except ImportError:
        local_h5 = _resolve_scvi_10x_h5_path(dataset_name)
        if local_h5 is not None:
            return _load_10x_h5(local_h5)
        raise ImportError("Loading scvi 10x datasets requires scvi-tools.")

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Variable names are not unique.*",
            category=UserWarning,
        )
        return dataset_10x(dataset_name, save_path=str(DATA_DIR), gex_only=False)


def _is_mudata(obj: Any) -> bool:
    mod = getattr(obj, "mod", None)
    return isinstance(mod, Mapping) or (hasattr(mod, "keys") and hasattr(mod, "__getitem__"))


def _find_modality_key(keys: list[Any], accepted_names: tuple[str, ...]) -> Any | None:
    accepted = {name.lower() for name in accepted_names}
    for key in keys:
        if str(key).lower() in accepted:
            return key
    return None


def _obs_frame(obj: Any) -> pd.DataFrame:
    if _is_mudata(obj):
        return obj.obs
    if isinstance(obj, ad.AnnData):
        return obj.obs
    raise TypeError(f"Unsupported object type: {type(obj)!r}")


def _label_sources(obj: Any) -> list[tuple[str | None, pd.DataFrame]]:
    if _is_mudata(obj):
        sources: list[tuple[str | None, pd.DataFrame]] = [(None, obj.obs)]
        for modality, adata in obj.mod.items():
            if hasattr(adata, "obs"):
                sources.append((str(modality), adata.obs))
        return sources
    return [(None, _obs_frame(obj))]


def _feature_type_mask(adata: ad.AnnData, accepted: tuple[str, ...]) -> pd.Series | None:
    if "feature_types" not in adata.var:
        return None
    values = adata.var["feature_types"].astype(str)
    return values.isin(accepted)


def load_anndata(path: str | Path) -> ad.AnnData:
    """Load an AnnData `.h5ad` object."""
    return ad.read_h5ad(Path(path))


def load_mudata(path: str | Path) -> Any:
    """Load a MuData `.h5mu` object."""
    path = Path(path)
    try:
        import mudata as md

        if hasattr(md, "read_h5mu"):
            return md.read_h5mu(path)
    except ImportError:
        pass

    try:
        import muon as mu

        if hasattr(mu, "read_h5mu"):
            return mu.read_h5mu(path)
        return mu.read(path)
    except ImportError as exc:
        raise ImportError("Loading .h5mu files requires mudata or muon.") from exc


def _make_var_names_unique(obj: Any) -> None:
    """Ensure var names are unique across modalities to avoid AnnData warnings."""
    if _is_mudata(obj):
        for _, adata in obj.mod.items():
            if hasattr(adata, "var_names") and "original_feature_name" not in adata.var:
                adata.var["original_feature_name"] = adata.var_names
            if hasattr(adata, "var_names_make_unique"):
                adata.var_names_make_unique()
        return
    if isinstance(obj, ad.AnnData):
        if "original_feature_name" not in obj.var:
            obj.var["original_feature_name"] = obj.var_names
        obj.var_names_make_unique()


def load_citeseq(path: str | Path) -> Any:
    """Load a CITE-seq object from AnnData, MuData, 10x HDF5/MEX, or scvi specs."""
    raw_value = str(path)
    if _is_scvi_10x_spec(raw_value):
        obj = _load_scvi_10x_dataset(raw_value)
        _make_var_names_unique(obj)
        return obj
    path = Path(path)
    if path.is_dir():
        if _is_10x_mex_dir(path):
            obj = _load_10x_mex_dir(path)
            _make_var_names_unique(obj)
            return obj
        raise ValueError(
            f"Unsupported directory input '{path}'. Expected a 10x MEX directory "
            "with matrix.mtx(.gz), barcodes.tsv(.gz), and features.tsv(.gz)."
        )
    suffix = path.suffix.lower()
    if suffix == ".h5ad":
        obj = load_anndata(path)
        _make_var_names_unique(obj)
        return obj
    if suffix == ".h5mu":
        obj = load_mudata(path)
        _make_var_names_unique(obj)
        return obj
    if suffix == ".h5":
        obj = _load_10x_h5(path)
        _make_var_names_unique(obj)
        return obj
    raise ValueError(
        f"Unsupported input suffix '{suffix}'. Expected .h5ad, .h5mu, .h5, "
        "a local 10x MEX directory, or a scvi/10x dataset spec."
    )


def save_citeseq_object(obj: Any, path: str | Path) -> Path:
    """Save an AnnData or MuData object using the matching on-disk format."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    if isinstance(obj, ad.AnnData):
        if suffix != ".h5ad":
            raise ValueError("AnnData objects must be saved with a .h5ad suffix.")
        obj.write_h5ad(out)
        return out
    if _is_mudata(obj):
        if suffix != ".h5mu":
            raise ValueError("MuData objects must be saved with a .h5mu suffix.")
        if hasattr(obj, "write_h5mu"):
            obj.write_h5mu(out)
            return out
        if hasattr(obj, "write"):
            obj.write(out)
            return out
        raise TypeError("MuData-like object does not provide write_h5mu() or write().")
    raise TypeError(f"Unsupported object type for saving: {type(obj)!r}")


def to_internal_anndata(obj: Any, dataset: str | None = None) -> ad.AnnData:
    """Convert AnnData/MuData CITE-seq input to the project AnnData layout.

    The returned object uses RNA features in ``adata.X``, protein/ADT counts in
    ``adata.obsm["protein_counts"]``, and protein feature names in
    ``adata.uns["protein_names"]``.
    """
    rna = get_rna_adata(obj).copy()
    protein = get_protein_matrix(obj)

    shared = pd.Index(rna.obs_names).intersection(pd.Index(protein.index))
    if len(shared) == 0:
        if rna.n_obs != protein.shape[0]:
            raise ValueError(
                "RNA and protein cell barcodes do not overlap, and the matrices "
                "have different numbers of cells."
            )
        protein = protein.copy()
        protein.index = rna.obs_names
    else:
        rna = rna[shared, :].copy()
        protein = protein.loc[shared].copy()

    if "counts" not in rna.layers:
        rna.layers["counts"] = rna.X.copy()
    rna.obsm["protein_counts"] = protein.reindex(rna.obs_names).to_numpy()
    rna.uns["protein_names"] = list(protein.columns.astype(str))
    if dataset is not None:
        rna.uns["dataset"] = str(dataset)
    return rna


def detect_modalities(obj: Any) -> dict[str, Any]:
    """Detect RNA and protein/ADT locations in an AnnData or MuData object."""
    if _is_mudata(obj):
        keys = list(obj.mod.keys())
        rna_key = _find_modality_key(keys, RNA_MODALITY_NAMES)
        protein_key = _find_modality_key(keys, PROTEIN_MODALITY_NAMES)
        return {
            "object_type": "MuData",
            "modalities": keys,
            "rna_key": rna_key,
            "protein_key": protein_key,
            "protein_location": "mod" if protein_key else None,
        }

    if not isinstance(obj, ad.AnnData):
        raise TypeError(f"Unsupported object type: {type(obj)!r}")

    protein_obsm_key = next((key for key in PROTEIN_OBSM_KEYS if key in obj.obsm), None)
    protein_feature_mask = _feature_type_mask(obj, PROTEIN_FEATURE_TYPES)
    rna_feature_mask = _feature_type_mask(obj, RNA_FEATURE_TYPES)
    has_protein_features = bool(protein_feature_mask is not None and protein_feature_mask.any())
    has_rna_features = bool(rna_feature_mask is None or rna_feature_mask.any())

    return {
        "object_type": "AnnData",
        "modalities": ["rna"] + (["protein"] if protein_obsm_key or has_protein_features else []),
        "rna_key": "X" if has_rna_features else None,
        "protein_key": protein_obsm_key or ("feature_types" if has_protein_features else None),
        "protein_location": "obsm" if protein_obsm_key else ("var" if has_protein_features else None),
    }


def get_rna_adata(obj: Any) -> ad.AnnData:
    """Return the RNA AnnData view/copy from AnnData or MuData input."""
    if _is_mudata(obj):
        info = detect_modalities(obj)
        if info["rna_key"] is None:
            raise ValueError(
                f"No RNA modality found. Expected one of {RNA_MODALITY_NAMES}; "
                f"available modalities are {info['modalities']}."
            )
        return obj.mod[info["rna_key"]]

    if not isinstance(obj, ad.AnnData):
        raise TypeError(f"Unsupported object type: {type(obj)!r}")

    rna_mask = _feature_type_mask(obj, RNA_FEATURE_TYPES)
    protein_mask = _feature_type_mask(obj, PROTEIN_FEATURE_TYPES)
    if rna_mask is not None and rna_mask.any():
        return obj[:, rna_mask.to_numpy()].copy()
    if protein_mask is not None and protein_mask.any() and (~protein_mask).any():
        return obj[:, (~protein_mask).to_numpy()].copy()
    return obj


def get_protein_matrix(obj: Any) -> pd.DataFrame:
    """Extract the protein/ADT matrix as a cell-by-protein DataFrame."""
    if _is_mudata(obj):
        info = detect_modalities(obj)
        protein_key = info["protein_key"]
        if protein_key is None:
            raise ValueError(
                f"No protein modality found. Expected one of {PROTEIN_MODALITY_NAMES}; "
                f"available modalities are {info['modalities']}."
            )
        prot = obj.mod[protein_key]
        return matrix_to_dataframe(
            prot.X,
            index=prot.obs_names,
            columns=prot.var_names,
            prefix="protein",
        )

    if not isinstance(obj, ad.AnnData):
        raise TypeError(f"Unsupported object type: {type(obj)!r}")

    for key in PROTEIN_OBSM_KEYS:
        if key in obj.obsm:
            value = obj.obsm[key]
            columns = None
            if isinstance(value, pd.DataFrame):
                return value.copy()
            if f"{key}_names" in obj.uns:
                columns = list(obj.uns[f"{key}_names"])
            return matrix_to_dataframe(value, index=obj.obs_names, columns=columns, prefix="protein")

    protein_mask = _feature_type_mask(obj, PROTEIN_FEATURE_TYPES)
    if protein_mask is not None and protein_mask.any():
        prot = obj[:, protein_mask.to_numpy()].copy()
        return matrix_to_dataframe(
            prot.X,
            index=prot.obs_names,
            columns=prot.var_names,
            prefix="protein",
        )

    raise ValueError(
        "No protein/ADT data found. Expected AnnData `.obsm` key "
        f"one of {PROTEIN_OBSM_KEYS}, combined 10x features with "
        "`adata.var['feature_types'] == 'Antibody Capture'`, or a MuData "
        f"modality named one of {PROTEIN_MODALITY_NAMES}."
    )


def candidate_label_columns(obj: Any) -> list[str]:
    """Return plausible cell-label columns from `.obs`."""
    candidates: list[str] = []
    for modality, obs in _label_sources(obj):
        lower_to_key = {key.lower(): key for key in obs.columns}
        for key in DEFAULT_LABEL_KEYS:
            if key.lower() in lower_to_key:
                candidate = lower_to_key[key.lower()]
                name = f"{modality}:{candidate}" if modality else candidate
                if name not in candidates:
                    candidates.append(name)

        for key in obs.columns:
            name = f"{modality}:{key}" if modality else key
            if name in candidates:
                continue
            series = obs[key]
            if pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
                n_unique = series.nunique(dropna=True)
                if 1 < n_unique <= min(100, max(2, len(series) // 2)):
                    candidates.append(name)
    return candidates


def get_cell_labels(
        obj: Any, preferred_keys: str | list[str] | tuple[str, ...] | None = None
) -> pd.Series | None:
    """Return the first available cell-label column, or None if absent."""
    if isinstance(preferred_keys, str):
        keys = [preferred_keys]
    else:
        keys = list(preferred_keys or DEFAULT_LABEL_KEYS)
    candidates = candidate_label_columns(obj)
    for key in keys + candidates:
        if ":" in key:
            modality, column = key.split(":", maxsplit=1)
            if _is_mudata(obj) and modality in obj.mod and column in obj.mod[modality].obs:
                labels = obj.mod[modality].obs[column].copy()
                labels.name = key
                return labels
        else:
            for modality, obs in _label_sources(obj):
                if key in obs:
                    labels = obs[key].copy()
                    labels.name = f"{modality}:{key}" if modality else key
                    return labels
    return None


def validate_citeseq_object(obj: Any) -> dict[str, Any]:
    """Validate basic CITE-seq structure and return a summary dictionary."""
    info = detect_modalities(obj)
    rna = get_rna_adata(obj)
    labels = get_cell_labels(obj)
    label_candidates = candidate_label_columns(obj)

    protein_error = None
    protein = None
    try:
        protein = get_protein_matrix(obj)
    except ValueError as exc:
        protein_error = str(exc)

    barcodes_align = None
    if protein is not None:
        if len(protein.index) == rna.n_obs:
            barcodes_align = bool(pd.Index(rna.obs_names).equals(pd.Index(protein.index)))
        else:
            barcodes_align = False

    return {
        "object_type": info["object_type"],
        "modalities": info["modalities"],
        "rna_key": info["rna_key"],
        "protein_key": info["protein_key"],
        "protein_location": info["protein_location"],
        "n_cells": int(rna.n_obs),
        "n_genes": int(rna.n_vars),
        "protein_modality_exists": protein is not None,
        "n_protein_features": int(protein.shape[1]) if protein is not None else 0,
        "protein_error": protein_error,
        "cell_type_labels_exist": labels is not None,
        "cell_type_label_key": labels.name if labels is not None else None,
        "candidate_label_columns": label_candidates,
        "barcodes_align": barcodes_align,
    }
