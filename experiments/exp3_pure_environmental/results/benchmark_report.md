# Experiment 3: Pure Environmental Ablation

- Target: `environmental_visits_per_1000_residents` = 1,000 × (waterborne + respiratory + heat-related visits) / registered residents.
- `nonresp_visits` is excluded from the target; its value is not an input.
- Temporal split: train 2025-02–2026-03 (210 rows); test 2026-04–2026-09 (90 rows).
- Seed: 42; CV-selected model: `LightGBM (tuned)`; LightGBM device: `cpu`.
- Predictor matrix: 51 lagged physical/environmental and event features only; no district, demographic, target-component, population-scale, or composite-index predictors.
- Lowest held-out RMSE was `Random forest`; the reported feature importance and synthesis use the training-CV-selected model `LightGBM (tuned)` to avoid selecting a winner on the test set.

The rate denominator is used exactly as specified but was not validated against `p3_population_exposed`; several districts have large denominator discrepancies. Clinical encounters are simulator-generated in the available pipeline. This is a predictive feature ablation and not a causal sensitivity estimate.

## Held-out benchmark

| model | test_rmse | test_mae | test_r2 |
| --- | --- | --- | --- |
| Random forest | 146.62558159946227 | 80.86322035285608 | 0.4318334489036947 |
| LightGBM (tuned) | 158.68892549538072 | 77.64817589397884 | 0.334497937281125 |
| Ridge (tuned) | 170.90429145786672 | 124.64091225399923 | 0.22809811198475805 |
| Dummy mean | 199.13400982067574 | 171.9171648286277 | -0.04796564299852135 |

## Top 10 pure-environmental permutation importance

| rank | feature | importance_mean | importance_std |
| --- | --- | --- | --- |
| 1 | lag1_pw_water_age_hours | 55.38749583711162 | 11.812650857381561 |
| 2 | lag1_p3_no2_ugm3 | 13.451277661078638 | 12.712840950309372 |
| 3 | lag1_pw_leak_incidents | 8.147580403810435 | 2.6877948168066643 |
| 4 | lag1_p2_collection_efficiency_pct | 0.7036631505577204 | 1.2461977517652592 |
| 5 | lag1_p2_organic_tons | 0.6347914621621384 | 0.7578309313209095 |
| 6 | lag1_p3_avg_aqi | 0.466151310282099 | 0.5261578739102364 |
| 7 | lag1_p2_avg_bin_fill_pct | 0.3098451377430521 | 0.1643898340776323 |
| 8 | lag1_pw_water_supplied_m3 | 0.22599713382709297 | 0.1026550869444351 |
| 9 | lag1_p3_pm25_ugm3 | 0.21647548186112003 | 0.04721086385477325 |
| 10 | lag1_p2_uncollected_tons | 0.14179357135849813 | 0.2334054765506869 |

## Requested physical drivers

| rank | feature | importance_mean | importance_std |
| --- | --- | --- | --- |
| 1 | lag1_pw_water_age_hours | 55.38749583711162 | 11.812650857381561 |
| 2 | lag1_p3_no2_ugm3 | 13.451277661078638 | 12.712840950309372 |
| 6 | lag1_p3_avg_aqi | 0.466151310282099 | 0.5261578739102364 |
| 9 | lag1_p3_pm25_ugm3 | 0.21647548186112003 | 0.04721086385477325 |
| 31 | lag1_avg_temp_c | 0.0 | 0.0 |
| 45 | lag1_pw_water_pollution_index | -0.0010677586442568554 | 0.25977137459477134 |

Importance identifies held-out predictive contribution under this ablation; it is not causal attribution.

## Feature exclusions

| column | dtype | justification |
| --- | --- | --- |
| district_key | int64 | Spatial identity shortcut excluded for pure environmental ablation |
| month | datetime64[us] | Used only to sort and form fixed chronological partitions |
| residents_registered | int64 | Population denominator only; excluded from predictor matrix |
| households | int64 | Raw population-volume confounder excluded from predictor matrix |
| children_pct | float64 | Demographic shortcut excluded for pure environmental ablation |
| elderly_pct | float64 | Demographic shortcut excluded for pure environmental ablation |
| daytime_workers_inflow | int64 | Population-volume confounder excluded from predictor matrix |
| students_day | int64 | Population-volume confounder excluded from predictor matrix |
| daytime_population | int64 | Raw population-volume confounder excluded from predictor matrix |
| density_per_km2 | float64 | Spatial/population-scale shortcut excluded from predictor matrix |
| attractiveness_index_old | float64 | Composite attractiveness index excluded |
| avg_temp_c | float64 | Current-month physical value withheld; only the t-1 lag is modeled |
| heat_days_ge38c | int64 | Current-month physical value withheld; only the t-1 lag is modeled |
| pw_metered_consumption_m3 | int64 | Current-month physical value withheld; only the t-1 lag is modeled |
| pw_water_supplied_m3 | int64 | Current-month physical value withheld; only the t-1 lag is modeled |
| pw_nrw_pct | float64 | Current-month physical value withheld; only the t-1 lag is modeled |
| pw_night_flow_ratio | float64 | Current-month physical value withheld; only the t-1 lag is modeled |
| pw_leak_incidents | int64 | Current-month physical value withheld; only the t-1 lag is modeled |
| pw_supply_interruption_hours | float64 | Current-month physical value withheld; only the t-1 lag is modeled |
| pw_water_age_hours | float64 | Current-month physical value withheld; only the t-1 lag is modeled |
| pw_chlorine_residual_mg_l | float64 | Current-month physical value withheld; only the t-1 lag is modeled |
| pw_turbidity_ntu | float64 | Current-month physical value withheld; only the t-1 lag is modeled |
| pw_ecoli_positive_pct | float64 | Current-month physical value withheld; only the t-1 lag is modeled |
| pw_sewage_overflow_incidents | int64 | Current-month physical value withheld; only the t-1 lag is modeled |
| pw_water_pollution_index | float64 | Current-month physical value withheld; only the t-1 lag is modeled |
| pw_waterborne_clinic_visits | int64 | Direct current-month environmental target component; target lineage leakage |
| pw_waterborne_er_visits | int64 | Direct current-month environmental target component; target lineage leakage |
| pw_waterborne_visits_total | int64 | Direct current-month environmental target component; target lineage leakage |
| p2_bins_installed | int64 | Current-month physical value withheld; only the t-1 lag is modeled |
| p2_avg_bin_fill_pct | float64 | Current-month physical value withheld; only the t-1 lag is modeled |
| p2_bin_hours_ge85_pct | float64 | Current-month physical value withheld; only the t-1 lag is modeled |
| p2_overflow_incidents | int64 | Current-month physical value withheld; only the t-1 lag is modeled |
| p2_collection_efficiency_pct | float64 | Current-month physical value withheld; only the t-1 lag is modeled |
| p2_collection_disruption_pp | float64 | Current-month physical value withheld; only the t-1 lag is modeled |
| p2_waste_generated_tons | float64 | Current-month physical value withheld; only the t-1 lag is modeled |
| p2_uncollected_tons | float64 | Current-month physical value withheld; only the t-1 lag is modeled |
| p2_organic_tons | float64 | Current-month physical value withheld; only the t-1 lag is modeled |
| p2_recyclable_tons | float64 | Current-month physical value withheld; only the t-1 lag is modeled |
| p2_recycling_diversion_pct | float64 | Current-month physical value withheld; only the t-1 lag is modeled |
| p2_odor_pest_risk_index | float64 | Current-month physical value withheld; only the t-1 lag is modeled |
| p3_population_exposed | int64 | Raw population-volume confounder and outcome-generation input |
| p3_avg_aqi | float64 | Current-month physical value withheld; only the t-1 lag is modeled |
| p3_max_aqi | int64 | Current-month physical value withheld; only the t-1 lag is modeled |
| p3_days_aqi_gt150 | int64 | Current-month physical value withheld; only the t-1 lag is modeled |
| p3_pm25_ugm3 | float64 | Current-month physical value withheld; only the t-1 lag is modeled |
| p3_pm10_ugm3 | float64 | Current-month physical value withheld; only the t-1 lag is modeled |
| p3_no2_ugm3 | float64 | Current-month physical value withheld; only the t-1 lag is modeled |
| p3_o3_ugm3 | float64 | Current-month physical value withheld; only the t-1 lag is modeled |
| p3_respiratory_clinic_visits | int64 | Direct current-month environmental target component; target lineage leakage |
| p3_respiratory_er_visits | int64 | Direct current-month environmental target component; target lineage leakage |
| p3_respiratory_visits_total | int64 | Direct current-month environmental target component; target lineage leakage |
| p3_heat_related_visits | int64 | Direct current-month environmental target component; target lineage leakage |
| p3_env_attractiveness_score | float64 | Not selected; outside the explicitly approved physical feature set |
| p4_total_complaints | int64 | Not selected; outside the explicitly approved physical feature set |
| p4_water_complaints | int64 | Not selected; outside the explicitly approved physical feature set |
| p4_waste_complaints | int64 | Not selected; outside the explicitly approved physical feature set |
| p4_pollution_complaints | int64 | Not selected; outside the explicitly approved physical feature set |
| p4_infrastructure_complaints | int64 | Not selected; outside the explicitly approved physical feature set |
| p4_complaints_per_1000 | float64 | Service/complaint proxy excluded from pure environmental feature space |
| p4_avg_resolution_hours | float64 | Not selected; outside the explicitly approved physical feature set |
| p4_sla_breach_pct | float64 | Not selected; outside the explicitly approved physical feature set |
| p4_open_requests_end_month | int64 | Not selected; outside the explicitly approved physical feature set |
| p4_citizen_satisfaction_pct | float64 | Not selected; outside the explicitly approved physical feature set |
| p4_service_need_index | float64 | Composite health/service index excluded |
| event_flags | str | Raw text parsed to binary event flags; only t-1 flags are modeled |
| pc_centers | int64 | Not selected; outside the explicitly approved physical feature set |
| pc_capacity_visits | int64 | Not selected; outside the explicitly approved physical feature set |
| hospital_centers | float64 | Not selected; outside the explicitly approved physical feature set |
| hospital_capacity_visits | float64 | Not selected; outside the explicitly approved physical feature set |
| pc_effective_capacity | float64 | Not selected; outside the explicitly approved physical feature set |
| nonresp_visits | float64 | Non-environmental baseline excluded from the target |
| total_visits_demand | float64 | Total-volume identity includes non-environmental baseline |
| pc_visits_demand | float64 | Derived 70% share of total demand |
| cpr | float64 | Derived from target-related demand and capacity |
| wait_days | float64 | Zero-variance invariant across all panel rows |
| unmet_visits | float64 | Derived from demand and capacity |
| unmet_share | float64 | Derived from demand and capacity |
| service_quality | float64 | Zero-variance invariant across all panel rows |
| esi_heat | float64 | Not selected; outside the explicitly approved physical feature set |
| esi_air | float64 | Not selected; outside the explicitly approved physical feature set |
| esi_water | float64 | Not selected; outside the explicitly approved physical feature set |
| esi_waste | float64 | Not selected; outside the explicitly approved physical feature set |
| environmental_stress_index | float64 | Not selected; outside the explicitly approved physical feature set |
| need_score | float64 | Composite decision index excluded |
| attractiveness_index_v2 | float64 | Composite downstream index excluded |
| environmental_visits_per_1000_residents | float64 | Experiment 3 target; never included as a predictor |
