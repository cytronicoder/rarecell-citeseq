"""Input path resolution helpers for benchmark scripts."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _display_path(path: str | Path) -> str:
    value = Path(path)
    resolved = _resolve_path(value)
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def resolve_input_file(
        requested_path: str | Path | None,
        default_candidates: list[str | Path],
) -> Path:
    """
    Resolve an input AnnData path.

    Behavior:
    - If requested_path is provided and exists, return it.
    - If requested_path is provided and does not exist: raise FileNotFoundError.
    - If requested_path is None: try default_candidates in order.
    - If no file is found: raise FileNotFoundError listing all tried paths.
    """
    tried: list[str] = []

    if requested_path is not None:
        requested = _resolve_path(requested_path)
        tried.append(_display_path(requested_path))
        if requested.exists():
            return requested
        raise FileNotFoundError(
            "Requested input file was not found:\n"
            f"{_display_path(requested_path)}\n"
            "Tried paths:\n"
            + "\n".join(f"- {path}" for path in tried)
        )

    for candidate in default_candidates:
        candidate_path = _resolve_path(candidate)
        display = _display_path(candidate)
        if display not in tried:
            tried.append(display)
        if candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(
        "No input AnnData file was found. Tried paths:\n"
        + "\n".join(f"- {path}" for path in tried)
    )
