import pandas as pd

from rarecell.plotting import (
    format_category_label,
    format_fraction_label,
    format_metric_label,
    format_representation_label,
    plot_marker_auc_results,
)


def test_format_representation_label_known_values():
    assert format_representation_label("rna_pca") == "RNA-only PCA"
    assert format_representation_label("protein_pca") == "Protein-only PCA"
    assert format_representation_label("joint_pca") == "RNA + protein PCA"


def test_format_metric_label_known_values():
    assert format_metric_label("f1") == "Rare-cell F1 score"
    assert format_metric_label("neighborhood_purity") == "Target-cell neighborhood purity"


def test_format_fraction_label_as_percentage():
    assert format_fraction_label(1.0) == "100%"
    assert format_fraction_label(0.1) == "10%"


def test_unknown_labels_fall_back_with_readable_spacing():
    assert format_representation_label("custom_representation") == "custom representation"
    assert format_metric_label("custom_metric") == "custom metric"


def test_format_category_label_for_clusters():
    assert format_category_label("cluster_3") == "Cluster 3"
    assert format_category_label("memory_B_cell") == "memory B cell"


def test_empty_marker_auc_plot_creates_placeholder(tmp_path):
    out = tmp_path / "empty_marker_auc.png"
    plot_marker_auc_results(pd.DataFrame(), out, target_label="cluster 3")
    assert out.exists()
    assert out.stat().st_size > 0


def test_non_empty_marker_auc_plot_creates_barplot(tmp_path):
    out = tmp_path / "marker_auc.png"
    table = pd.DataFrame(
        {
            "target_label": ["3"],
            "label_key": ["cluster"],
            "modality": ["rna"],
            "marker": ["GENE_HIGH"],
            "matched_feature_name": ["GENE_HIGH"],
            "auc": [1.0],
            "n_target": [3],
            "n_other": [3],
            "marker_source": ["data_driven_cluster_marker"],
            "match_type": ["data_driven"],
            "match_reason": ["test"],
        }
    )
    plot_marker_auc_results(table, out, target_label="cluster 3")
    assert out.exists()
    assert out.stat().st_size > 0
