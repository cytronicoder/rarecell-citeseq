import pytest

from rarecell.naming import resolve_existing_output_prefix


def test_resolve_existing_output_prefix_prefers_existing_requested_prefix(tmp_path):
    (tmp_path / "requested__benchmark_results.csv").write_text("x\n", encoding="utf-8")
    (tmp_path / "other__benchmark_results.csv").write_text("x\n", encoding="utf-8")

    assert resolve_existing_output_prefix(tmp_path, "requested") == "requested"


def test_resolve_existing_output_prefix_falls_back_to_matching_table(tmp_path):
    (tmp_path / "pbmc5k_15__benchmark_results.csv").write_text("x\n", encoding="utf-8")

    assert resolve_existing_output_prefix(tmp_path, "pbmc5k_b_cell") == "pbmc5k_15"


def test_resolve_existing_output_prefix_raises_when_no_results_exist(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run notebooks/rare_cell_downsampling_benchmark"):
        resolve_existing_output_prefix(tmp_path, "pbmc5k_b_cell")
