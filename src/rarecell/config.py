"""Project paths and representation-name contracts."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
LOGS_DIR = RESULTS_DIR / "logs"
METRICS_DIR = RESULTS_DIR / "metrics"
INTERMEDIATE_DIR = RESULTS_DIR / "intermediate"
REPORTS_DIR = RESULTS_DIR / "reports"

REPRESENTATION_KEY_MAP = {
    "rna_pca": "X_rna_pca",
    "protein_pca": "X_protein_pca",
    "joint_pca": "X_joint_pca",
}

REPRESENTATION_ALIASES = {
    **REPRESENTATION_KEY_MAP,
    "X_rna_pca": "X_rna_pca",
    "X_protein_pca": "X_protein_pca",
    "X_joint_pca": "X_joint_pca",
    "X_joint_simple": "X_joint_simple",
    "X_joint": "X_joint_simple",
}

ANNDATA_KEY_TO_REPRESENTATION = {value: key for key, value in REPRESENTATION_KEY_MAP.items()}
ANNDATA_KEY_TO_REPRESENTATION["X_joint_simple"] = "joint_pca"
ANNDATA_KEY_TO_REPRESENTATION["X_joint"] = "joint_pca"


def ensure_project_dirs() -> None:
    """Create project data/result directories if they are missing."""
    for path in [
        DATA_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        LOGS_DIR,
        METRICS_DIR,
        INTERMEDIATE_DIR,
        REPORTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
