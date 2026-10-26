# Benchmark Results

- Target: raw `total_visits_demand` (per-exposed rates deferred pending source validation).
- Temporal test: 2026-04–2026-09, 90 rows.
- Training: 2025-02–2026-03, 210 rows.
- Seeds: 42; LightGBM device: `cpu`; CPU threads: 8.
- Tuned model selected using training-only expanding-window CV: `Ridge`.
- Best tuned parameters: `{"model__alpha": 1.0}`.

## Cross-validation

| model | cv_rmse_mean | cv_rmse_std | cv_mae_mean | cv_r2_mean |
| --- | --- | --- | --- | --- |
| Dummy mean | 3106.344398545157 | 240.16578445547776 | 2244.0144814814817 | -0.028885297450069958 |
| Ridge | 361.6233540995102 | 45.517357835934554 | 262.9794472296167 | 0.9852599973007578 |
| Random forest | 926.8534324370445 | 135.94710229635382 | 538.6601104951097 | 0.9012604658585429 |
| LightGBM | 1021.4965405114873 | 164.46111434941386 | 616.4411302719114 | 0.8788625464832905 |

## Held-out test

| model | cv_rmse_mean | test_rmse | test_mae | test_r2 |
| --- | --- | --- | --- | --- |
| Ridge (tuned) | 300.1959313999186 | 252.0378963910478 | 161.3809396227993 | 0.9957414902933 |
| Ridge | 361.6233540995102 | 313.66184720555685 | 197.88589671037101 | 0.9934044754580081 |
| Random forest | 926.8534324370445 | 1060.8445523869846 | 518.466565719953 | 0.9245551759077119 |
| LightGBM | 1021.4965405114873 | 1156.8008607589566 | 525.6222713318833 | 0.9102895251674564 |
| Dummy mean | 3106.344398545157 | 3960.342458808718 | 2792.8751322751323 | -0.05145679729600472 |

## Top 10 permutation importance

| feature | importance_mean | importance_std |
| --- | --- | --- |
| lag1_residents_registered | 1033.287077017782 | 79.1181201304733 |
| lag1_households | 1032.6934217034682 | 79.08474180515651 |
| lag1_p2_bins_installed | 860.3013898955775 | 74.34871291188709 |
| lag1_p2_waste_generated_tons | 663.4131521741249 | 59.81630148093186 |
| lag1_p2_recyclable_tons | 540.9372759960578 | 52.82845557135118 |
| lag1_p2_organic_tons | 62.09710812091299 | 15.423661544768336 |
| district_key | 51.374495353869825 | 9.752003123244418 |
| lag1_pw_water_age_hours | 34.298831834956616 | 9.83727248638041 |
| lag1_p3_no2_ugm3 | 27.240769008614528 | 8.597918053186719 |
| lag1_p2_collection_efficiency_pct | 19.53610677270815 | 6.397797134295982 |

Permutation importance is computed on the held-out timestamps. Correlated environmental predictors can share or mask importance; this table describes predictive contribution, not causal impact.

## Feature exclusions

| column | dtype | justification |
| --- | --- | --- |
| month | datetime64[us] | Used only to create chronological split and calendar features |
| residents_registered | int64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| households | int64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| children_pct | float64 | Target leakage: direct input to generated nonresp_visits |
| elderly_pct | float64 | Target leakage: direct input to generated nonresp_visits |
| daytime_workers_inflow | int64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| students_day | int64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| daytime_population | int64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| density_per_km2 | float64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| attractiveness_index_old | float64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| avg_temp_c | float64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| heat_days_ge38c | int64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| pw_metered_consumption_m3 | int64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| pw_water_supplied_m3 | int64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| pw_nrw_pct | float64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| pw_night_flow_ratio | float64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| pw_leak_incidents | int64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| pw_supply_interruption_hours | float64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| pw_water_age_hours | float64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| pw_chlorine_residual_mg_l | float64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| pw_turbidity_ntu | float64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| pw_ecoli_positive_pct | float64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| pw_sewage_overflow_incidents | int64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| pw_water_pollution_index | float64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| pw_waterborne_clinic_visits | int64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| pw_waterborne_er_visits | int64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| pw_waterborne_visits_total | int64 | Mathematical redundancy: additive target component |
| p2_bins_installed | int64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| p2_avg_bin_fill_pct | float64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| p2_bin_hours_ge85_pct | float64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| p2_overflow_incidents | int64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| p2_collection_efficiency_pct | float64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| p2_collection_disruption_pp | float64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| p2_waste_generated_tons | float64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| p2_uncollected_tons | float64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| p2_organic_tons | float64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| p2_recyclable_tons | float64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| p2_recycling_diversion_pct | float64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| p2_odor_pest_risk_index | float64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| p3_population_exposed | int64 | Target leakage: direct input to generated nonresp_visits |
| p3_avg_aqi | float64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| p3_max_aqi | int64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| p3_days_aqi_gt150 | int64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| p3_pm25_ugm3 | float64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| p3_pm10_ugm3 | float64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| p3_no2_ugm3 | float64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| p3_o3_ugm3 | float64 | Contemporaneous value withheld; only its t-1 lag is modeled |
| p3_respiratory_clinic_visits | int64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| p3_respiratory_er_visits | int64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| p3_respiratory_visits_total | int64 | Mathematical redundancy: additive target component |
| p3_heat_related_visits | int64 | Mathematical redundancy: additive target component |
| p3_env_attractiveness_score | float64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| p4_total_complaints | int64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| p4_water_complaints | int64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| p4_waste_complaints | int64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| p4_pollution_complaints | int64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| p4_infrastructure_complaints | int64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| p4_complaints_per_1000 | float64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| p4_avg_resolution_hours | float64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| p4_sla_breach_pct | float64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| p4_open_requests_end_month | int64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| p4_citizen_satisfaction_pct | float64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| p4_service_need_index | float64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| event_flags | str | Text event flags excluded from this numeric benchmark |
| pc_centers | int64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| pc_capacity_visits | int64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| hospital_centers | float64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| hospital_capacity_visits | float64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| pc_effective_capacity | float64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| nonresp_visits | float64 | Mathematical redundancy: additive target component |
| total_visits_demand | float64 | Target label; never included in the feature matrix |
| pc_visits_demand | float64 | Mathematical redundancy: rounded 70% of target |
| cpr | float64 | Target leakage: derived from pc_visits_demand and capacity |
| wait_days | float64 | Zero variance and target-derived |
| unmet_visits | float64 | Zero variance and target-derived |
| unmet_share | float64 | Zero variance and target-derived |
| service_quality | float64 | Zero variance and target-derived |
| esi_heat | float64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| esi_air | float64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| esi_water | float64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| esi_waste | float64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| environmental_stress_index | float64 | Not selected: health/capacity/derived measure outside the environmental feature scope |
| need_score | float64 | Target leakage: includes cpr |
| attractiveness_index_v2 | float64 | Target leakage: downstream score includes service_quality |
