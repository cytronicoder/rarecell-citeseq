import numpy as np
import pandas as pd
import pytest

ad = pytest.importorskip("anndata")

from rarecell.validation import resolve_label_key, resolve_target_label


def test_resolve_label_key_prefers_available_annotation():
    adata = ad.AnnData(
        np.ones((3, 2)),
        obs=pd.DataFrame(
            {"cell_type_simple": ["B", "T", "T"], "leiden": ["0", "1", "1"]},
            index=["cell0", "cell1", "cell2"],
        ),
    )
    assert resolve_label_key(adata, preferred="cell_type_simple") == "cell_type_simple"


def test_resolve_label_key_falls_back_to_leiden():
    adata = ad.AnnData(
        np.ones((3, 2)),
        obs=pd.DataFrame({"leiden": ["0", "1", "1"]}, index=["cell0", "cell1", "cell2"]),
    )
    assert resolve_label_key(adata, preferred="cell_type_simple") == "leiden"


def test_resolve_target_label_uses_preferred_when_present():
    adata = ad.AnnData(
        np.ones((3, 2)),
        obs=pd.DataFrame({"cell_type": ["B", "T", "T"]}, index=["cell0", "cell1", "cell2"]),
    )
    assert resolve_target_label(adata, "cell_type", preferred="B") == "B"


def test_resolve_target_label_falls_back_to_smallest_class():
    adata = ad.AnnData(
        np.ones((5, 2)),
        obs=pd.DataFrame(
            {"leiden": ["0", "0", "1", "1", "2"]},
            index=["cell0", "cell1", "cell2", "cell3", "cell4"],
        ),
    )
    assert resolve_target_label(adata, "leiden", preferred="B cell") == "2"
