# Triple-Benchmark Synthesis Report

## Executive summary

The three benchmarks form a planning ladder, not a causal identification design. Experiment 1 predicts overall simulated visit volume; Experiment 2 predicts the specified environmental-encounter rate while retaining district and demographic controls; Experiment 3 removes spatial and demographic shortcuts and tests prediction using lagged physical environmental/infrastructure stressors alone.

The target is synthetic in the available source pipeline, and the registered-resident denominator differs substantially from `p3_population_exposed` in some districts. Treat Experiments 2–3 rates as the requested analytical index pending label and denominator validation.

## Unified held-out performance

| experiment | cv_selected_model | target | test_rmse | test_mae | test_r2 | metric_units |
| --- | --- | --- | --- | --- | --- | --- |
| Exp 1: Macro volume | Ridge (tuned) | Total simulated visits | 252.0378963910478 | 161.3809396227993 | 0.9957414902933 | visits / district-month |
| Exp 2: Environmental rate | LightGBM (tuned) | Environmental visits / 1,000 registered residents | 86.74852890365413 | 51.63285246062297 | 0.8011249030770438 | rate points / 1,000 registered residents |
| Exp 3: Pure environmental ablation | LightGBM (tuned) | Environmental visits / 1,000 registered residents | 158.68892549538072 | 77.64817589397884 | 0.334497937281125 | rate points / 1,000 registered residents |

Model rows are selected using expanding-window CV on the training period; the test split is reserved for final reporting. RMSE and MAE for Exp 1 are in absolute visits and must not be compared numerically to rate-point errors in Exp 2–3. R² is also affected by the differing target construction and predictor restrictions; compare it as within-experiment performance, not as a causal sensitivity score.

## Feature-importance evolution

- **Exp 1 — demographic volume:** top features were `lag1_residents_registered`, `lag1_households`, `lag1_p2_bins_installed`, `lag1_p2_waste_generated_tons`, `lag1_p2_recyclable_tons`, `lag1_p2_organic_tons`, `district_key`, `lag1_pw_water_age_hours`, `lag1_p3_no2_ugm3`, `lag1_p2_collection_efficiency_pct`. Lagged registered residents and households led, reflecting volume scale.
- **Exp 2 — spatial/age controls:** top features were `district_key`, `lag1_children_pct`, `lag1_elderly_pct`, `lag1_pw_chlorine_residual_mg_l`, `lag1_p2_collection_efficiency_pct`, `lag1_p2_odor_pest_risk_index`, `lag1_p2_bin_hours_ge85_pct`, `month_cos`, `lag1_p3_max_aqi`, `month_index`. District identity and age composition led; environmental measures added comparatively little importance.
- **Exp 3 — pure environmental ablation:** top features were `lag1_pw_water_age_hours`, `lag1_p3_no2_ugm3`, `lag1_pw_leak_incidents`, `lag1_p2_collection_efficiency_pct`, `lag1_p2_organic_tons`, `lag1_p3_avg_aqi`, `lag1_p2_avg_bin_fill_pct`, `lag1_pw_water_supplied_m3`, `lag1_p3_pm25_ugm3`, `lag1_p2_uncollected_tons`. Only lagged physical conditions and event indicators are eligible; neither district nor demographic predictors can act as shortcuts.

Requested physical driver ranks in Experiment 3:

| feature | rank | importance_mean |
| --- | --- | --- |
| lag1_p3_no2_ugm3 | 2 | 13.451277661078638 |
| lag1_pw_water_age_hours | 1 | 55.38749583711162 |
| lag1_p3_pm25_ugm3 | 9 | 0.21647548186112003 |
| lag1_p3_avg_aqi | 6 | 0.466151310282099 |
| lag1_avg_temp_c | 31 | 0.0 |
| lag1_pw_water_pollution_index | 45 | -0.0010677586442568554 |

A high rank indicates predictive contribution in this held-out sample, not causation. Correlated, seasonal, and generated features can obscure or redistribute permutation importance. Negative or near-zero importance means the feature did not improve held-out prediction under this model.

## Three-tier urban governance use

1. **Tier 1 — long-term facility budgets:** use Exp 1 total-volume forecasts to scenario-plan clinic capacity, staffing, and budget envelopes. Validate with observed facility utilization and retain uncertainty buffers.
2. **Tier 2 — structural district vulnerability ranking:** use Exp 2's district/age-controlled rate benchmark as a screening and prioritization aid, not an individual or causal risk score. Confirm population denominators and clinical labels before policy allocation.
3. **Tier 3 — environmental surge response:** use Exp 3 as a physical-stressor monitoring layer alongside direct air, water, heat, and event thresholds. Where the model detects useful signal, trigger human review, sampling, inspections, and preparedness—not autonomous clinical or punitive decisions.

Exp 3 is the strongest ablation of population/spatial shortcut pathways available in this panel, but it does not estimate a causal effect: outcomes and encounter components are simulated, and a 15-district/21-month panel cannot by itself identify causal environmental sensitivity.

## Reproducibility and preservation

- Experiment 3 seed: 42; training rows: 210 (2025-02–2026-03); test rows: 90 (2026-04–2026-09).
- Exp 3 CV winner: `LightGBM (tuned)`; LightGBM device: `cpu`.
- Experiment 1 and 2 artifact directories were read-only during Experiment 3. The source panel was also read-only; run manifests record the input checksum.
- Reproduce Experiment 3 with `python run_experiments.py --experiment 3`; run all sequentially with `python run_experiments.py --experiment all`.
- Detailed target lineage, feature sets/exclusions, CV/tuning settings, model-by-model results, limitations, and the artifact map are documented in the [Experiment Methodology and Findings](./EXPERIMENT_METHODOLOGY_AND_FINDINGS.md).
