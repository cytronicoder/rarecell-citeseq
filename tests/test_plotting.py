import pandas as pd

from rarecell.plotting import (
    format_category_label,
    format_fraction_label,
    format_fraction_labels,
    format_metric_label,
    format_representation_label,
    get_method_colors,
    order_methods,
    plot_cross_target_method_ranking,
    plot_final_figure_2_main_benchmark,
    plot_multi_target_metric_curve,
    plot_study_design,
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
    assert format_fraction_labels([1.0, 0.5, 0.05]) == ["100%", "50%", "5%"]


def test_method_order_and_colors_are_stable():
    methods = ["X_joint_simple", "X_protein_pca", "X_rna_pca"]
    assert order_methods(methods) == ["X_rna_pca", "X_protein_pca", "X_joint_simple"]
    colors = get_method_colors(methods)
    assert colors["X_rna_pca"] == colors["RNA-only PCA"]
    assert colors["X_protein_pca"] == colors["Protein-only PCA"]
    assert colors["X_joint_simple"] == colors["RNA + protein PCA"]


def test_unknown_labels_fall_back_with_readable_spacing():
    assert format_representation_label("custom_representation") == "custom representation"
    assert format_metric_label("custom_metric") == "custom metric"


def test_format_category_label_for_clusters():
    assert format_category_label("cluster_3") == "Cluster 3"
    assert format_category_label("memory_B_cell") == "memory B cell"


def test_final_plotting_functions_create_files(tmp_path):
    raw = pd.DataFrame(
        {
            "target_cell_type": ["B cell", "B cell", "NK cell", "NK cell"],
            "representation": ["rna_pca", "joint_pca", "rna_pca", "joint_pca"],
            "fraction": [1.0, 1.0, 0.5, 0.5],
            "seed": [0, 0, 0, 0],
            "f1": [0.8, 0.9, 0.5, 0.6],
            "neighborhood_purity": [0.7, 0.8, 0.4, 0.5],
        }
    )
    ranking = pd.DataFrame(
        {
            "representation": ["rna_pca", "joint_pca"],
            "metric": ["f1", "f1"],
            "mean_rank": [2.0, 1.0],
            "best_count": [0, 2],
            "worst_count": [2, 0],
            "notes": ["", ""],
        }
    )
    outputs = [
        plot_multi_target_metric_curve(raw, "f1", tmp_path / "multi_f1.png"),
        plot_cross_target_method_ranking(ranking, tmp_path / "ranking.png"),
        plot_study_design(tmp_path / "study_design.png", {"target_cell_types": ["B cell"], "fractions": [1.0]}),
        plot_final_figure_2_main_benchmark(raw, tmp_path / "main.png"),
    ]
    for output in outputs:
        assert output.exists()
        assert output.stat().st_size > 0
