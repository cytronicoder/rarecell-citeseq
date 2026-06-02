"""Shared helpers for reproducible command-line scripts."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import ANNDATA_KEY_TO_REPRESENTATION, REPRESENTATION_ALIASES, REPRESENTATION_KEY_MAP


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


def resolve_representation_key(adata, representation: str) -> str:
    """Resolve a user-facing representation name to an existing AnnData obsm key."""
    canonical_key = REPRESENTATION_ALIASES.get(str(representation), str(representation))
    if canonical_key in adata.obsm:
        return canonical_key
    raise KeyError(f"Representation '{representation}' is missing from adata.obsm as '{canonical_key}'.")


def resolve_representations(adata, representations: Iterable[str] | None = None) -> list[tuple[str, str]]:
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


def read_count_table(path: Path) -> pd.DataFrame | None:
    """Read a known count summary table if present."""
    if not path.exists():
        return None
    table = pd.read_csv(path)
    return table if not table.empty else None
