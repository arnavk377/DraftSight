# Final Report Modeling Outputs

This folder contains report-ready tables and figures generated from the final
walk-forward model run.

## Reproduce

Run the full model evaluation:

```bash
python -m src.model.train_n_evaluate
```

Then regenerate the report tables and figures:

```bash
python -m src.model.generate_report_outputs
```

The model script writes raw model outputs to `results/`. The report script reads
those outputs and writes the cleaned deliverables to `reports/final_report_modeling/latest/`.

Use `latest/figures_vector/` for final report insertion when possible. Those
SVG/PDF files stay sharp when resized. Use `latest/figures/` if the destination
app requires PNG; those PNGs are exported at high resolution.

For Google Docs specifically, use `latest/figures_google_doc/`. Those PNGs are
page-sized with larger labels, so they stay clearer after Docs rescales them.
This folder includes Google Docs-friendly versions of the pred-vs-actual grid,
walk-forward metrics, and the single XGBoost 2024 holdout plot.

## Modeling Notes

- College performance features are enabled by default.
- To reproduce a no-college sensitivity run, set `DRAFTSIGHT_EXCLUDE_COLLEGE_STATS=1`.
- FT-Transformer uses a smaller report-friendly training budget by default so the
  full walk-forward run is feasible on a laptop.
- The spline model is intentionally kept as a simple baseline; it is not expected
  to beat the tree models.

## Key Tables

- `latest/tables/figure_captions.csv`
- `latest/tables/overall_model_performance.csv`
- `latest/tables/overall_requested_model_performance.csv`
- `latest/tables/year_by_year_model_metrics.csv`
- `latest/tables/selected_years_model_metrics.csv`
- `latest/tables/pick_bin_performance_by_model.csv`
- `latest/tables/pick_value_curves_oos_by_model.csv`
- `latest/tables/all_out_of_sample_predictions_long.csv`

## Key Figures

- `latest/figures/pred_vs_actual_grid_requested_models_2024.png`
- `latest/figures/pred_vs_actual_grid_requested_models_all_years.png`
- `latest/figures/walkforward_metrics_requested_models.png`
- `latest/figures/walkforward_rmse_teaser_requested_models.png`
- `latest/figures/overall_model_performance_grid.png`
- `latest/figures/pick_value_curves_oos_by_model.png`
- `latest/figures/pick_value_curves_exact_pick_no_bins.png`
- `latest/figures/pick_value_curves_model_facets.png`
- `latest/figures/pick_value_curve_monotonic_report.png`
- `latest/figures/mae_by_pick_bin_requested_models.png`
- `latest/figures/calibration_by_prediction_decile_requested_models.png`
- `latest/figures/residual_distributions_requested_models.png`
- `latest/figures/feature_importance_xgb_top20.png`
- `latest/figures/feature_importance_rf_top20.png`
- `latest/figures/tree_feature_importance_top5.png`

## Captions

Use `latest/figure_captions.md` or `latest/tables/figure_captions.csv` as the
source text for report figure explanations. Each generated figure has a short
caption describing what is plotted and how to interpret it.
