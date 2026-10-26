# Comparative Decision Report

## Executive summary

Experiment 1 forecasts total simulated demand for capacity and staffing planning. Experiment 2 forecasts an environmentally linked encounter rate per 1,000 registered residents for environmental monitoring and surge preparedness. They answer different questions and their RMSE/MAE values are in different units; compare R² cautiously and do not treat either experiment as a causal estimate.

## Best-model test performance

| experiment | model | target | test_r2 | test_rmse | test_mae |
| --- | --- | --- | --- | --- | --- |
| Experiment 1: macro volume | Ridge (tuned) | total_visits_demand | 0.9957414902933 | 252.0378963910478 | 161.3809396227993 |
| Experiment 2: environmental rate | LightGBM (tuned) | environmental_visits_per_1000_residents | 0.8011249030770438 | 86.74852890365413 | 51.63285246062297 |

Experiment 1 is the macro-volume baseline from its preserved artifact set; Experiment 2 selects the best tuned/baseline model by the predeclared temporal holdout RMSE.

## Feature-driver shift

- Experiment 1 top-ten features: `lag1_residents_registered`, `lag1_households`, `lag1_p2_bins_installed`, `lag1_p2_waste_generated_tons`, `lag1_p2_recyclable_tons`, `lag1_p2_organic_tons`, `district_key`, `lag1_pw_water_age_hours`, `lag1_p3_no2_ugm3`, `lag1_p2_collection_efficiency_pct`. Demographic volume proxies (especially lagged registered residents and households) dominate its ranking.
- Experiment 2 top-ten features: `district_key`, `lag1_children_pct`, `lag1_elderly_pct`, `lag1_pw_chlorine_residual_mg_l`, `lag1_p2_collection_efficiency_pct`, `lag1_p2_odor_pest_risk_index`, `lag1_p2_bin_hours_ge85_pct`, `month_cos`, `lag1_p3_max_aqi`, `month_index`.
- Experiment 2 top-ten environmental features: `lag1_pw_chlorine_residual_mg_l`, `lag1_p3_max_aqi`; explicit PM2.5, AQI, temperature, and water-pollution ranks follow, with full results in `exp2_environmental_rate/results/feature_importance_all.csv`.

Requested environmental driver ranks (from held-out permutation importance):

| driver | rank | importance_mean | importance_std |
| --- | --- | --- | --- |
| PM2.5 | 42 | -0.24352033708752643 | 0.04488066958304308 |
| Average AQI | 40 | -0.003365009115297113 | 0.007773960956366244 |
| Temperature | 25 | 0.0 | 0.0 |
| Water pollution index | 19 | 0.0 | 0.0 |

The named air, temperature, and water-pollution features rank below district and demographic controls; near-zero or negative held-out permutation importance means they added little predictive signal in this benchmark. This is not evidence of no causal environmental effect.

Experiment 2 excludes raw population counts from predictors, but registered population is its requested denominator. Districts 1009–1015 have marked resident-versus-exposed-population discrepancies in the source panel; the registered-resident rate is therefore a specified operational index, not a validated epidemiological incidence rate. Its large values can reflect denominator mismatch.

## Dual-model deployment recommendation

1. **Capacity and staffing budgets:** use Experiment 1's total-volume forecast as a planning input for facility capacity, workforce, and budget scenarios. Review forecast errors by district and month; retain operational buffers because the target is simulator-derived.
2. **Environmental early warning and surge response:** use Experiment 2 as a monitoring signal alongside direct AQI, water-quality, and temperature thresholds. Trigger human review and preparedness actions rather than automatic resource allocation from model scores alone. Recalibrate only after validating encounter labels and the registered-population denominator against observed data.
3. **Do not infer causal sensitivity from these benchmarks.** Both outcomes and some environmental-health components were generated in a synthetic linkage pipeline. Permutation importance and predictive performance measure associations within this dataset, not causal environmental effects.

## Reproducibility

- Seed: 42; training: 2025-02–2026-03 (210 district-months); test: 2026-04–2026-09 (90 district-months).
- Experiment 2 winner: `LightGBM (tuned)`; tuning: `{"lightgbm": {"model__max_depth": 2, "model__min_child_samples": 10}, "ridge": {"model__alpha": 1.0}}`; LightGBM device: `cpu`.
- Experiment 1 original artifacts are preserved under `experiments/exp1_macro_volume/original_artifacts/`; reproducible reruns are written separately.
