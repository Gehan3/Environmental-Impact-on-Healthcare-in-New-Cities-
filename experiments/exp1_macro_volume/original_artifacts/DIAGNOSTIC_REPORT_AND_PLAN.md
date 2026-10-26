# Municipal Panel Diagnostic Audit and Experiment Plan

**Status:** Phase 1 reviewed and approved; Phase 2 executed using the approved raw-volume fallback.  
**Audited input:** `Data/Processed/master_panel.csv` (read-only)

## 1. Data inventory and integrity

| Check | Result |
|---|---|
| Rows / columns | 315 / 85 |
| Panel | 15 districts × 21 monthly timestamps |
| Period | January 2025–September 2026 |
| District-month key | `district_key`, `month`; 0 duplicate keys |
| Missing values | 118 blank `event_flags`; no blanks in the other columns |
| Original-file handling | Read only; no data, code, or model artifacts were generated during this audit |

The dataset contains a complete balanced district-month panel. `event_flags` is text and should not enter the initial numeric feature matrix without an explicit encoding decision.

## 2. Zero-variance and invariant features

These four columns have one unique value across all 315 rows and must be removed from model inputs. They carry no learnable signal; three are also downstream target metrics.

| Column | Observed value | Reason |
|---|---:|---|
| `wait_days` | 0.4 | Zero variance; generated from the capacity ratio |
| `unmet_visits` | 0.0 | Zero variance; generated from demand and capacity |
| `unmet_share` | 0.0 | Zero variance; generated from unmet visits |
| `service_quality` | 0.96 | Zero variance; generated from wait and unmet share |

No other column is invariant. The lowest non-constant standard deviation is `elderly_pct` (about 0.0112); it is not invariant and should not be removed merely on that basis.

## 3. Target construction and leakage audit

The repository's `link_all_data.py` constructs the target as:

```text
nonresp_visits = round(p3_population_exposed * 0.2125
                       * (1 + 1.2 * (children_pct - 0.28)
                            + 2.5 * (elderly_pct - 0.04)))
total_visits_demand = nonresp_visits
                    + p3_respiratory_visits_total
                    + pw_waterborne_visits_total
                    + p3_heat_related_visits
pc_visits_demand = round(0.70 * total_visits_demand)
cpr = pc_visits_demand / pc_effective_capacity
```

The target identity holds exactly in **315/315 rows**. `pc_visits_demand` equals rounded `0.70 * total_visits_demand` in **315/315 rows**. The stored `cpr` agrees with its generating ratio within 0.0005, consistent with output rounding. Thus evaluation against this label measures how well a model reconstructs a simulated formula, not prediction of independently observed clinic demand.

The feature exclusion list for a same-month prediction is:

| Column(s) | Observed dtype | Technical reason |
|---|---|---|
| `total_visits_demand` | float64 | Target label; never an input |
| `nonresp_visits` | float64 | Additive target component; included in the exact target sum |
| `p3_respiratory_visits_total`, `pw_waterborne_visits_total`, `p3_heat_related_visits` | int64 | Additive target components; their sum with `nonresp_visits` reproduces the label exactly |
| `p3_population_exposed` | int64 | Direct input to the generated `nonresp_visits` formula; contemporaneous value can reconstruct a target component |
| `children_pct`, `elderly_pct` | float64 | Direct inputs to the generated `nonresp_visits` formula; contemporaneous values can reconstruct a target component |
| `pc_visits_demand` | float64 | Rounded 70%-of-target proxy |
| `cpr` | float64 | Directly derived from target proxy and capacity |
| `wait_days`, `service_quality` | float64 | Target/capacity descendants; additionally invariant in this extract |
| `unmet_visits`, `unmet_share` | float64 | Target/capacity descendants; additionally invariant in this extract |
| `need_score` | float64 | Includes `cpr` in its formula |
| `attractiveness_index_v2` | float64 | Downstream score includes `service_quality`; exclude to avoid target-descendant lineage |

Only those listed columns and the label itself are proposed for exclusion on target-lineage grounds; unrelated environmental measurements remain candidates. All environmental/weather predictors would use **t−1 values**, never contemporaneous values, and no visit-count component or lagged visit-count feature is proposed for the first benchmark. This blocks direct same-month construction paths while preserving a one-month-ahead environmental forecasting question. It does not make a synthetic label equivalent to observed health outcomes.

## 4. Population scale and target choice

Absolute target volume is strongly associated with population scale:

| Pair | Pearson correlation |
|---|---:|
| `residents_registered` vs. `total_visits_demand` | 0.9699 |
| `households` vs. `total_visits_demand` | 0.9699 |

However, the population denominator needs validation before a residents-normalized target is used. For example, district `1015` has 43–89 registered residents but 1,271–1,584 exposed people; districts `1009`–`1014` also show sizable resident/exposed-population differences. These may reflect different denominator definitions, but the current panel does not document the distinction. Normalizing by registered residents produces a target rate from 257 to 7,372 visits per 1,000 residents and a correlation of −0.447 with registered population, indicating that this denominator is not presently safe to interpret as a per-capita rate.

**Decision after review:** the repository does not contain the source that creates `p3_population_exposed` or documentation of its population/units. The linking code's use of the field confirms intended computational lineage, but does not validate its semantics. The pragmatic Phase 2 choice is therefore to use raw `total_visits_demand` as the primary operational-volume target and defer per-exposed rates until source definitions are supplied and validated. The proposed rate `1,000 * total_visits_demand / p3_population_exposed` yields 171–303 visits per 1,000 exposed people and has a raw-volume correlation of 0.9938, but is not interpreted or benchmarked as a validated rate. Raw target results remain evaluations of a simulator-generated outcome, not independent observed clinic demand.

## 5. Temporal validation design

- Sort by `month`, then `district_key`; create all lag features within each district.
- Hold out the final six timestamps: **April–September 2026** (90 district-month rows).
- Train only on **January 2025–March 2026** (225 rows before lag warm-up). January 2025 is retained only as lag history; rows with unavailable t−1 predictors are excluded from fitting.
- Within the training period, use expanding-window validation on whole timestamps, with no random/K-fold splitting. A three-fold `TimeSeriesSplit` over the ordered training months is a practical tuning strategy; all districts for a month stay in the same fold.
- Fit all transforms on each training fold only. Evaluate the locked test set once after selecting parameters.

Weather is repeated across districts for each month, so the panel has 315 rows but only 21 distinct monthly weather states. Results should be interpreted as a small-sample temporal benchmark, not as 315 independent environmental replications.

## 6. Proposed benchmark

1. `DummyRegressor` — constant reference baseline.
2. `Ridge` — standardized continuous predictors plus one-hot district and calendar features.
3. `RandomForestRegressor` — shallow trees (`max_depth=3`, `min_samples_leaf=8–15`, bounded tree count).
4. `LightGBM` — shallow trees (`max_depth=2–3`, `num_leaves<=7`, `min_child_samples=20–30`, regularization, bounded estimators).

Compare models on the same temporal folds and report test RMSE, MAE, and R². Tune only the CV winner within the training period, then report the held-out result once. Use temporal permutation importance on the held-out data for the top 10 features, with a warning that correlated environmental measures can divide or obscure importance; report whether environmental features rank highly rather than requiring that they do.

## 7. Phase 2 execution and results

The user approved Phase 2 with per-exposed rates conditional on validating the source definition. That source and documentation are absent, so the approved pragmatic fallback was applied: raw total visits is the primary target; no per-exposed rate is presented as validated.

The reproducible pipeline is [`run_experiments.py`](../run_experiments.py). It writes split files to `experiments/data/` and reports, predictions, exclusions, importance, and a run manifest to `experiments/results/`. The original input remained unchanged; its SHA-256 is `cff9180167a4f1981dae4afe0fecfa19e94c17dab99d3277ac6a8f303f910a43`.

### Held-out benchmark

Training used February 2025–March 2026 (210 rows after t−1 warm-up); testing used April–September 2026 (90 rows). The winning family was selected by three expanding-window folds on the training period; the test set was not used for tuning.

| Model | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|
| Tuned Ridge (`alpha=1.0`) | 252.04 | 161.38 | 0.9957 |
| Ridge (initial) | 313.66 | 197.89 | 0.9934 |
| Random Forest | 1,060.84 | 518.47 | 0.9246 |
| LightGBM | 1,156.80 | 525.62 | 0.9103 |
| Dummy mean | 3,960.34 | 2,792.88 | −0.0515 |

Metrics are in visits per district-month and describe reconstruction of the simulator-generated volume, not independent observed clinical demand. The high scores should not be interpreted as evidence of causal environmental effects.

### Top 10 tuned-Ridge permutation importance

| Rank | Feature | RMSE increase |
|---:|---|---:|
| 1 | `lag1_residents_registered` | 1,033.29 |
| 2 | `lag1_households` | 1,032.69 |
| 3 | `lag1_p2_bins_installed` | 860.30 |
| 4 | `lag1_p2_waste_generated_tons` | 663.41 |
| 5 | `lag1_p2_recyclable_tons` | 540.94 |
| 6 | `lag1_p2_organic_tons` | 62.10 |
| 7 | `district_key` | 51.37 |
| 8 | `lag1_pw_water_age_hours` | 34.30 |
| 9 | `lag1_p3_no2_ugm3` | 27.24 |
| 10 | `lag1_p2_collection_efficiency_pct` | 19.54 |

Environmental features appear among the top ten, but population proxies rank first and second. Correlation between predictors makes permutation scores non-additive; importance is predictive, not causal.

### Reproduction and artifacts

Install dependencies with `pip install -r requirements.txt`, then run `python run_experiments.py`. The script fixes seeds at 42, performs a strict chronological split and training-only expanding-window tuning, and records versions, split dates, parameters, device, and the input checksum. It attempted LightGBM CUDA and GPU backends; this LightGBM build had no enabled CUDA tree learner and no OpenCL device, so execution explicitly fell back to 8-thread CPU.

- Split datasets: [`experiments/data/`](./data/)
- Benchmark table, feature exclusion table, top-10 importance, predictions, and manifest: [`experiments/results/`](./results/)
- Reproduction instructions: [`README.md`](../README.md)
- Source denominator remains unvalidated; add the `p3_population_exposed` source definition before benchmarking normalized rates.
