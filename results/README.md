### Results

Generated benchmark outputs live here. Only `.gitkeep` placeholders are tracked inside output subdirectories.

```text
results/
  figures/        benchmark PNG/PDF figures
  intermediate/   cell-count grids and other small intermediate tables
  logs/           script logs and run summaries
  metrics/        raw per-condition benchmark/control metrics
  reports/        generated Markdown reports
  tables/         benchmark summaries and figure index tables
```

Canonical generated files use `{output_prefix}__{descriptor}` names:

- `tables/{output_prefix}__benchmark_results.csv`
- `metrics/{output_prefix}__benchmark_raw.csv`
- `tables/{output_prefix}__metric_summary.csv`
- `intermediate/{output_prefix}__cell_counts_by_fraction.csv`
- `tables/{output_prefix}__random_label_control.csv`
- `metrics/{output_prefix}__abundant_control_raw.csv`
- `figures/{output_prefix}__rare_cell_f1_curve.{png,pdf}`
- `figures/{output_prefix}__rare_cell_recall_curve.{png,pdf}`
- `figures/{output_prefix}__rare_cell_precision_curve.{png,pdf}`
- `figures/{output_prefix}__neighborhood_purity_curve.{png,pdf}`
- `figures/{output_prefix}__target_cell_counts.{png,pdf}`
- `figures/{output_prefix}__*_heatmap.{png,pdf}`
- `reports/{output_prefix}__benchmark_report.md`

Regenerate outputs from the project root:

```bash
python scripts/run_benchmark.py
python scripts/generate_figures.py
python scripts/write_report.py
```
