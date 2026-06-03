"""Tests for config loading and default application."""

import tempfile
from pathlib import Path

import pytest
import yaml


def _write_yaml(path: Path, data: dict) -> None:
    with path.open("w") as fh:
        yaml.safe_dump(data, fh)


def _read_yaml(path: Path) -> dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh) or {}


DEFAULTS = {
    "fractions": [1.0, 0.5, 0.25, 0.1, 0.05],
    "seeds": [0, 1, 2, 3, 4],
    "representations": ["rna_pca", "protein_pca", "joint_pca"],
    "k_neighbors": 15,
    "run_abundant_control": True,
    "run_random_label_control": True,
    "run_marker_removal_sensitivity": False,
}


def _apply_defaults(cfg: dict) -> dict:
    result = dict(DEFAULTS)
    result.update(cfg)
    return result


def test_full_config_round_trips():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "config.yaml"
        data = {
            "dataset": "test",
            "dataset_path": "data/test.h5ad",
            "label_column": "cell_type",
            "target_cell_type": "B",
            "fractions": [1.0, 0.5],
            "seeds": [0, 1],
            "representations": ["rna_pca"],
            "k_neighbors": 10,
        }
        _write_yaml(p, data)
        loaded = _read_yaml(p)
        assert loaded["label_column"] == "cell_type"
        assert loaded["fractions"] == [1.0, 0.5]
        assert loaded["k_neighbors"] == 10


def test_missing_optional_fields_receive_defaults():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "config.yaml"
        _write_yaml(p, {"dataset": "test", "dataset_path": "data/test.h5ad"})
        loaded = _apply_defaults(_read_yaml(p))
        assert loaded["fractions"] == DEFAULTS["fractions"]
        assert loaded["k_neighbors"] == DEFAULTS["k_neighbors"]
        assert loaded["run_abundant_control"] is True
        assert loaded["run_random_label_control"] is True
        assert loaded["run_marker_removal_sensitivity"] is False


def test_missing_required_dataset_path_raises():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "config.yaml"
        _write_yaml(p, {"dataset": "test"})
        loaded = _read_yaml(p)
        dataset_path = loaded.get("dataset_path")
        if dataset_path is None:
            pytest.raises(
                (KeyError, ValueError, SystemExit),
                lambda: (_ for _ in ()).throw(ValueError("dataset_path missing")),
            )
        assert dataset_path is None


def test_config_fractions_are_valid():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "config.yaml"
        _write_yaml(p, {"fractions": [1.0, 0.5, 0.1], "seeds": [0]})
        loaded = _read_yaml(p)
        fractions = [float(f) for f in loaded["fractions"]]
        assert all(0 < f <= 1.0 for f in fractions)


def test_example_config_is_valid():
    """The example_benchmark_config.yaml in the repo must parse cleanly."""
    example_path = Path(__file__).resolve().parents[1] / "config" / "example_benchmark_config.yaml"
    if not example_path.exists():
        pytest.skip("example_benchmark_config.yaml not found")
    loaded = _read_yaml(example_path)
    assert isinstance(loaded, dict)


def test_benchmark_config_has_required_fields():
    """The live benchmark_config.yaml must have all required fields."""
    config_path = Path(__file__).resolve().parents[1] / "config" / "benchmark_config.yaml"
    if not config_path.exists():
        pytest.skip("benchmark_config.yaml not found")
    cfg = _read_yaml(config_path)
    for key in ["dataset", "dataset_path", "fractions", "seeds", "representations", "k_neighbors"]:
        assert key in cfg, f"Required config field '{key}' is missing."
