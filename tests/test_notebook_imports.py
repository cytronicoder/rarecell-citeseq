import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _setup_cell_source(notebook_path: Path) -> str:
    notebook = json.loads(notebook_path.read_text())
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and "PROJECT_ROOT = Path.cwd().resolve()" in source:
            return source
    raise AssertionError(f"No setup cell found in {notebook_path}")


def test_notebook_setup_cells_import_from_repo_root(monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT)
    for notebook_path in sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb")):
        namespace = {"__name__": "__main__"}
        exec(_setup_cell_source(notebook_path), namespace)


def test_notebook_setup_cells_import_from_notebooks_dir(monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT / "notebooks")
    for notebook_path in sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb")):
        namespace = {"__name__": "__main__"}
        exec(_setup_cell_source(notebook_path), namespace)
