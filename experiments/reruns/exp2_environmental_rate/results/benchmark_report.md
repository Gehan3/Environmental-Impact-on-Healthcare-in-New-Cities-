# Experiment 2: Environmental Health Intensity

- Target: `environmental_visits_per_1000_residents` = 1,000 × (environment-linked encounters) / `residents_registered`.
- Numerator: `pw_waterborne_visits_total + p3_respiratory_visits_total + p3_heat_related_visits`; `nonresp_visits` is excluded.
- Train: 2025-02–2026-03 (210 rows); test: 2026-04–2026-09 (90 rows).
- Fixed seed: 42; LightGBM device: `cpu`; CPU fallback: 8 threads.
- Winner selected on training-only expanding-window CV: `LightGBM (tuned)`.
- Ridge tuning: `{"model__alpha": 1.0}`; LightGBM tuning: `{"model__max_depth": 2, "model__min_child_samples": 10}`.

The requested registered-resident denominator is used exactly, but has known discrepancies from `p3_population_exposed` in several districts. Treat the result as a specified rate index, not validated epidemiological incidence. Since the encounter components are synthetic in the source pipeline, results and feature importance are predictive associations, not causal effects.

## Held-out benchmark

| model | test_rmse | test_mae | test_r2 |
| --- | --- | --- | --- |
| LightGBM (tuned) | 86.74852890365413 | 51.63285246062297 | 0.8011249030770438 |
| Random forest | 94.46736589551105 | 46.87366833742045 | 0.7641587443254103 |
| Ridge (tuned) | 95.24547988309462 | 65.90540248150879 | 0.7602575630772812 |
| Dummy mean | 199.13400982067574 | 171.9171648286277 | -0.04796564299852135 |

## Top 10 permutation importance

| feature | importance_mean | importance_std |
| --- | --- | --- |
| district_key | 108.50664613300862 | 11.955938783732424 |
| lag1_children_pct | 56.78458398807479 | 6.807948220447827 |
| lag1_elderly_pct | 9.379164399697782 | 2.0273421551123554 |
| lag1_pw_chlorine_residual_mg_l | 0.5674210756357511 | 0.26997558783103676 |
| lag1_p2_collection_efficiency_pct | 0.2679025671570187 | 0.20778287095485293 |
| lag1_p2_odor_pest_risk_index | 0.04874691078240332 | 0.022468069789273862 |
| lag1_p2_bin_hours_ge85_pct | 0.016174235160112717 | 0.06726006658809121 |
| month_cos | 0.0 | 0.0 |
| lag1_p3_max_aqi | 0.0 | 0.0 |
| month_index | 0.0 | 0.0 |

## Requested environmental drivers

| feature | rank | importance_mean |
| --- | --- | --- |
| lag1_p3_pm25_ugm3 | 42 | -0.24352033708752643 |
| lag1_p3_avg_aqi | 40 | -0.003365009115297113 |
| lag1_avg_temp_c | 25 | 0.0 |
| lag1_pw_water_pollution_index | 19 | 0.0 |

Full permutation rankings are saved to `feature_importance_all.csv`.

The named air, temperature, and water-pollution variables add little held-out predictive signal relative to district and age-share controls in this benchmark. This does not establish that environmental factors have no causal effect.

All environmental features are lagged one month; current-month encounter components, baseline demand, target descendants, raw population counts, and raw event text are excluded from predictors.

## Feature exclusions

| column | dtype | justification |
| --- | --- | --- |
| month | datetime64[us] | Used for chronological split and calendar features |
| residents_registered | int64 | Population denominator only; raw scale excluded from predictors |
| households | int64 | Raw population-scale confounder excluded from predictors |
| children_pct | float64 | Current-month value withheld; only the t-1 lag is modeled |
| elderly_pct | float64 | Current-month value withheld; only the t-1 lag is modeled |
| daytime_workers_inflow | int64 | Not selected; outside approved environmental and ratio-control feature scope |
| students_day | int64 | Not selected; outside approved environmental and ratio-control feature scope |
| daytime_population | int64 | Raw population-scale confounder excluded from predictors |
| density_per_km2 | float64 | Not selected; outside approved environmental and ratio-control feature scope |
| attractiveness_index_old | float64 | Not selected; outside approved environmental and ratio-control feature scope |
| avg_temp_c | float64 | Current-month value withheld; only the t-1 lag is modeled |
| heat_days_ge38c | int64 | Current-month value withheld; only the t-1 lag is modeled |
| pw_metered_consumption_m3 | int64 | Not selected; outside approved environmental and ratio-control feature scope |
| pw_water_supplied_m3 | int64 | Not selected; outside approved environmental and ratio-control feature scope |
| pw_nrw_pct | float64 | Not selected; outside approved environmental and ratio-control feature scope |
| pw_night_flow_ratio | float64 | Not selected; outside approved environmental and ratio-control feature scope |
| pw_leak_incidents | int64 | Not selected; outside approved environmental and ratio-control feature scope |
| pw_supply_interruption_hours | float64 | Current-month value withheld; only the t-1 lag is modeled |
| pw_water_age_hours | float64 | Not selected; outside approved environmental and ratio-control feature scope |
| pw_chlorine_residual_mg_l | float64 | Current-month value withheld; only the t-1 lag is modeled |
| pw_turbidity_ntu | float64 | Current-month value withheld; only the t-1 lag is modeled |
| pw_ecoli_positive_pct | float64 | Current-month value withheld; only the t-1 lag is modeled |
| pw_sewage_overflow_incidents | int64 | Current-month value withheld; only the t-1 lag is modeled |
| pw_water_pollution_index | float64 | Current-month value withheld; only the t-1 lag is modeled |
| pw_waterborne_clinic_visits | int64 | Direct current-month target component or clinical subcomponent; excluded from predictors |
| pw_waterborne_er_visits | int64 | Direct current-month target component or clinical subcomponent; excluded from predictors |
| pw_waterborne_visits_total | int64 | Direct current-month target component or clinical subcomponent; excluded from predictors |
| p2_bins_installed | int64 | Not selected; outside approved environmental and ratio-control feature scope |
| p2_avg_bin_fill_pct | float64 | Not selected; outside approved environmental and ratio-control feature scope |
| p2_bin_hours_ge85_pct | float64 | Current-month value withheld; only the t-1 lag is modeled |
| p2_overflow_incidents | int64 | Current-month value withheld; only the t-1 lag is modeled |
| p2_collection_efficiency_pct | float64 | Current-month value withheld; only the t-1 lag is modeled |
| p2_collection_disruption_pp | float64 | Not selected; outside approved environmental and ratio-control feature scope |
| p2_waste_generated_tons | float64 | Not selected; outside approved environmental and ratio-control feature scope |
| p2_uncollected_tons | float64 | Not selected; outside approved environmental and ratio-control feature scope |
| p2_organic_tons | float64 | Not selected; outside approved environmental and ratio-control feature scope |
| p2_recyclable_tons | float64 | Not selected; outside approved environmental and ratio-control feature scope |
| p2_recycling_diversion_pct | float64 | Not selected; outside approved environmental and ratio-control feature scope |
| p2_odor_pest_risk_index | float64 | Current-month value withheld; only the t-1 lag is modeled |
| p3_population_exposed | int64 | Raw population-scale confounder and target-generation input |
| p3_avg_aqi | float64 | Current-month value withheld; only the t-1 lag is modeled |
| p3_max_aqi | int64 | Current-month value withheld; only the t-1 lag is modeled |
| p3_days_aqi_gt150 | int64 | Current-month value withheld; only the t-1 lag is modeled |
| p3_pm25_ugm3 | float64 | Current-month value withheld; only the t-1 lag is modeled |
| p3_pm10_ugm3 | float64 | Current-month value withheld; only the t-1 lag is modeled |
| p3_no2_ugm3 | float64 | Current-month value withheld; only the t-1 lag is modeled |
| p3_o3_ugm3 | float64 | Current-month value withheld; only the t-1 lag is modeled |
| p3_respiratory_clinic_visits | int64 | Direct current-month target component or clinical subcomponent; excluded from predictors |
| p3_respiratory_er_visits | int64 | Direct current-month target component or clinical subcomponent; excluded from predictors |
| p3_respiratory_visits_total | int64 | Direct current-month target component or clinical subcomponent; excluded from predictors |
| p3_heat_related_visits | int64 | Direct current-month target component or clinical subcomponent; excluded from predictors |
| p3_env_attractiveness_score | float64 | Not selected; outside approved environmental and ratio-control feature scope |
| p4_total_complaints | int64 | Not selected; outside approved environmental and ratio-control feature scope |
| p4_water_complaints | int64 | Not selected; outside approved environmental and ratio-control feature scope |
| p4_waste_complaints | int64 | Not selected; outside approved environmental and ratio-control feature scope |
| p4_pollution_complaints | int64 | Not selected; outside approved environmental and ratio-control feature scope |
| p4_infrastructure_complaints | int64 | Not selected; outside approved environmental and ratio-control feature scope |
| p4_complaints_per_1000 | float64 | Not selected; outside approved environmental and ratio-control feature scope |
| p4_avg_resolution_hours | float64 | Not selected; outside approved environmental and ratio-control feature scope |
| p4_sla_breach_pct | float64 | Not selected; outside approved environmental and ratio-control feature scope |
| p4_open_requests_end_month | int64 | Not selected; outside approved environmental and ratio-control feature scope |
| p4_citizen_satisfaction_pct | float64 | Not selected; outside approved environmental and ratio-control feature scope |
| p4_service_need_index | float64 | Not selected; outside approved environmental and ratio-control feature scope |
| event_flags | str | Parsed to binary flags and lagged one month; raw text excluded |
| pc_centers | int64 | Not selected; outside approved environmental and ratio-control feature scope |
| pc_capacity_visits | int64 | Not selected; outside approved environmental and ratio-control feature scope |
| hospital_centers | float64 | Not selected; outside approved environmental and ratio-control feature scope |
| hospital_capacity_visits | float64 | Not selected; outside approved environmental and ratio-control feature scope |
| pc_effective_capacity | float64 | Not selected; outside approved environmental and ratio-control feature scope |
| nonresp_visits | float64 | Non-environmental baseline demand; excluded from target |
| total_visits_demand | float64 | Contains non-environmental baseline and target components |
| pc_visits_demand | float64 | Derived 70% share of total demand |
| cpr | float64 | Derived from demand and capacity |
| wait_days | float64 | Zero-variance invariant; derived from demand and capacity |
| unmet_visits | float64 | Zero-variance invariant; derived from demand and capacity |
| unmet_share | float64 | Zero-variance invariant; derived from demand and capacity |
| service_quality | float64 | Zero-variance invariant; derived from demand and capacity |
| esi_heat | float64 | Not selected; outside approved environmental and ratio-control feature scope |
| esi_air | float64 | Not selected; outside approved environmental and ratio-control feature scope |
| esi_water | float64 | Not selected; outside approved environmental and ratio-control feature scope |
| esi_waste | float64 | Not selected; outside approved environmental and ratio-control feature scope |
| environmental_stress_index | float64 | Not selected; outside approved environmental and ratio-control feature scope |
| need_score | float64 | Not selected; outside approved environmental and ratio-control feature scope |
| attractiveness_index_v2 | float64 | Derived score may contain target descendants |
| environmental_visits_per_1000_residents | float64 | Target variable; not used as a feature |
