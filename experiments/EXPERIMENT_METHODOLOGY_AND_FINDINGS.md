# Experiment Methodology and Findings

This document records the reproducible design and principal findings for all three municipal-panel experiments. It complements the [triple-benchmark synthesis](./TRIPLE_BENCHMARK_SYNTHESIS_REPORT.md) and the individual reports under each experiment directory.

## 1. Data, scope, and integrity

- Input: `Data/Processed/master_panel.csv`, read-only throughout.
- Panel size: 315 district-month rows, 15 districts, 21 monthly timestamps, January 2025–September 2026.
- Panel key: (`district_key`, `month`); the source panel has one row per district-month.
- Source checksum recorded in the experiment manifests: `cff9180167a4f1981dae4afe0fecfa19e94c17dab99d3277ac6a8f303f910a43`.
- Deterministic seed: 42. The Python and NumPy RNGs and model estimator seeds are fixed; CPU fitting uses at most 8 threads.
- LightGBM attempted GPU execution. This installed build did not enable CUDA tree learning and had no OpenCL device; the experiments therefore ran on CPU.
- No random K-fold splitting was used.

The first month, January 2025, supplies the within-district t−1 feature history only. The common model partitions are:

| Partition | Months | District-month rows |
|---|---|---:|
| Train | February 2025–March 2026 | 210 |
| Test | April–September 2026 | 90 |

Model selection and tuning use three expanding-window `TimeSeriesSplit` folds. Each validation fold covers 45 rows (three complete months × 15 districts), keeping each timestamp's districts together. The held-out six-month test period is reserved for final metrics and permutation importance, not hyperparameter selection.

## 2. Outcome lineage and interpretation

The repository's linking code constructs `total_visits_demand` from `nonresp_visits` plus waterborne, respiratory, and heat-related encounters. `nonresp_visits` is a modeled general-demand baseline, not an observed environmental encounter. The encounter totals used in Experiments 2 and 3 are also generated in the source pipeline.

Experiments 2 and 3 use the requested rate:

```text
1000 * (
    pw_waterborne_visits_total
    + p3_respiratory_visits_total
    + p3_heat_related_visits
) / residents_registered
```

`nonresp_visits` is excluded from this numerator. The requested `residents_registered` denominator is used exactly, but its semantics should be validated before interpreting the result as an epidemiological rate. The panel shows marked differences between registered and exposed populations, including district 1015 (43–89 registered residents versus 1,271–1,584 exposed people) and districts 1009–1014. Do not interpret predictive scores, feature importance, or the model's performance as causal environmental sensitivity.

## 3. Experiment designs

### Experiment 1 — Macro-volume demand

- **Target:** raw `total_visits_demand`.
- **Feature design:** lagged environmental/service indicators, lagged `residents_registered` and `households`, district categorical indicators, and calendar terms. Within-district lag warm-up removes January 2025 from fitting.
- **Exclusions:** target label and additive components (`nonresp_visits`, waterborne, respiratory, heat-related visits); contemporaneous direct inputs to generated baseline demand (`p3_population_exposed`, `children_pct`, `elderly_pct`); target descendants (`pc_visits_demand`, `cpr`, `wait_days`, `unmet_visits`, `unmet_share`, `service_quality`, `need_score`, `attractiveness_index_v2`). Zero-variance columns include `wait_days=0.4`, `unmet_visits=0`, `unmet_share=0`, and `service_quality=0.96`.
- **Candidates:** mean `DummyRegressor`; standardized `Ridge`; shallow `RandomForestRegressor` (200 trees, max depth 3, leaf size 8); LightGBM (120 trees, max depth 3, at most 7 leaves, minimum 20 child samples).
- **Selection/tuning:** expanding-window CV selected Ridge; its alpha search was `[0.1, 1, 10, 100]`, with `alpha=1`.
- **Test result:** RMSE 252.04 visits, MAE 161.38 visits, R² 0.9957.
- **Leading features:** lagged registered residents and households, followed by lagged waste scale/quantity features. This is the macro-volume model and demographic volume proxies predict its scale.

### Experiment 2 — Environmental encounter rate with controls

- **Target:** the environmental encounter rate per 1,000 registered residents above.
- **Feature design:** one-month lags of the specified air-quality, water/network, waste/sanitation, and climate measurements; binary indicators parsed from `event_flags` and lagged one month; lagged `children_pct` and `elderly_pct` controls; district categorical and calendar terms.
- **Exclusions:** current-month environmental encounter components and their clinic/ER subcomponents; baseline and total volume outcomes; target/capacity identities (`pc_visits_demand`, `cpr`, `unmet_visits`, `unmet_share`); invariant `wait_days` and `service_quality`; raw population counts; composite indices; raw event text.
- **Candidates:** mean dummy baseline; standardized Ridge (alpha grid `[0.01, 0.1, 1, 10, 100]`); 100-tree Random Forest (max depth 3, minimum leaf 8); LightGBM (100 trees, max depth 3, at most 7 leaves, minimum 10 child samples).
- **Tuning:** Ridge and LightGBM were tuned using expanding-window CV. LightGBM grid: max depth `[2, 3]`, minimum child samples `[10, 20]`. CV selected LightGBM with max depth 2 and 10 child samples.
- **Test result:** RMSE 86.75 rate points, MAE 51.63 rate points, R² 0.8011.
- **Leading features:** district identity and age-share controls led. The named environmental drivers ranked much lower: water pollution index 19th, temperature 25th, average AQI 40th, PM2.5 42nd. The ranking does not establish that environmental exposure has no causal effect.

### Experiment 3 — Pure environmental ablation

- **Target:** exactly the Experiment 2 rate formula.
- **Feature design:** 33 requested physical environmental/infrastructure/weather measurements, each lagged one month, plus 18 binary `event_flags` indicators lagged one month (51 features total). No district ID, calendar features, demographics, population counts, service complaint measures, or composite index is in the predictor matrix.
- **Exclusions:** current-month target numerator components and all `pw_waterborne_*` / `p3_respiratory_*` clinical subcomponents; baseline and total visits; target/capacity identities; zero-variance outcomes; all specified population-volume indicators; district and demographics; `need_score`, both attractiveness indices, `p4_service_need_index`, `p4_complaints_per_1000`; and all non-approved fields.
- **Candidates:** mean dummy baseline; standardized Ridge (alpha grid `[0.01, 0.1, 1, 10, 100]`); 100-tree Random Forest (max depth 3, minimum leaf 8); tuned LightGBM (100 trees, shallow depth, at most 7 leaves, minimum child samples 10).
- **Tuning:** Ridge and LightGBM used expanding-window CV. LightGBM grid: max depth `[2, 3]`, minimum child samples `[10, 20]`. CV selected LightGBM with max depth 3 and 10 child samples. Random Forest has slightly lower test RMSE, but it was not selected based on the test set.
- **CV-selected-model test result:** RMSE 158.69 rate points, MAE 77.65 rate points, R² 0.3345.
- **Lowest-RMSE test candidate:** Random Forest, RMSE 146.63, MAE 80.86, R² 0.4318. This is recorded for transparency; the CV-selected LightGBM remains the designated model to avoid test-set selection.
- **Leading features:** lagged water age ranked first (permutation RMSE increase 55.39); NO₂ ranked second (13.45); leak incidents ranked third (8.15). PM2.5 ranked 9th (0.22), average AQI 6th (0.47), temperature 31st (0), and water pollution index 45th (approximately −0.001). Importance is model- and holdout-specific predictive association, not causal attribution.

## 4. Held-out metrics and units

| Experiment/model selected by CV | Test RMSE | Test MAE | Test R² | Metric units |
|---|---:|---:|---:|---|
| Exp 1 — Ridge (`alpha=1`) | 252.04 | 161.38 | 0.9957 | Visits per district-month |
| Exp 2 — LightGBM (depth 2, child samples 10) | 86.75 | 51.63 | 0.8011 | Rate points per 1,000 registered residents |
| Exp 3 — LightGBM (depth 3, child samples 10) | 158.69 | 77.65 | 0.3345 | Rate points per 1,000 registered residents |

Exp 1 absolute-volume RMSE/MAE cannot be directly compared to the normalized-rate RMSE/MAE. R² differs with target variance and feature restrictions and is not a causal effect measure.

## 5. Key findings and governance implications

1. Removing population counts and district/demographic shortcuts reduces Exp 3's test R² relative to the controlled Exp 2 benchmark (0.3345 versus 0.8011). Environmental-only lagged features carry some predictive signal, led by water age and NO₂, but are substantially weaker than district/age controls in this panel.
2. The top Exp 3 drivers do not mean water age or NO₂ caused the generated visits. The sample is only 15 districts over 21 months, measured encounters are simulator-generated, environmental predictors are correlated, and the rate denominator has unresolved population-definition discrepancies.
3. **Tier 1:** use Exp 1 for long-term facility-capacity and staffing scenarios, with operational buffers and validation against observed utilization.
4. **Tier 2:** use Exp 2 as a structural district vulnerability screening aid only after validating the denominator and labels; do not treat it as individual risk.
5. **Tier 3:** use Exp 3 alongside direct air/water/heat monitoring as a cautious surge-preparedness signal. Route flags to human review, sampling, inspections, and interventions rather than autonomous clinical or punitive actions.

## 6. Reproduction and artifact map

Install the pinned requirements and run:

```bash
python run_experiments.py --experiment 1
python run_experiments.py --experiment 2
python run_experiments.py --experiment 3
python run_experiments.py --experiment all
```

`--all` runs sequentially. To preserve existing experiment outputs, Experiment 1 and 2 reruns are written under `experiments/reruns/`; Experiment 3 writes only under `experiments/exp3_pure_environmental/`. The existing Experiment 1 and 2 experiment directories are not run destinations.

- Experiment 1 preserved snapshot: `experiments/exp1_macro_volume/original_artifacts/`.
- Experiment 2 preserved benchmark: `experiments/exp2_environmental_rate/`.
- Experiment 3 partitions, exclusions, serialized models, results, manifest, and logs: `experiments/exp3_pure_environmental/`.
- Unified executive synthesis: `experiments/TRIPLE_BENCHMARK_SYNTHESIS_REPORT.md`.
- The runner prints SHA-256 checksums for generated deliverables. Repeat runs were checked byte-for-byte; both protected experiment directories and the source panel were verified unchanged.
