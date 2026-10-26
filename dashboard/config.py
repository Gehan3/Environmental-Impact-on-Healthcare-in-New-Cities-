"""Configuration registry for data fields, benchmark artifacts, and policy assumptions."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "Data" / "Processed" / "master_panel.csv"

EXPERIMENTS = {
    "Experiment 1 — Macro volume": {
        "key": "exp1",
        "label": "Macro volume",
        "target": "total_visits_demand",
        "metrics": ROOT
        / "experiments/exp1_macro_volume/original_artifacts/results/benchmark_metrics.csv",
        "predictions": ROOT
        / "experiments/exp1_macro_volume/original_artifacts/results/test_predictions.csv",
        "importance": ROOT
        / "experiments/exp1_macro_volume/original_artifacts/results/top10_feature_importance.csv",
        "exclusions": ROOT
        / "experiments/exp1_macro_volume/original_artifacts/results/feature_exclusions.csv",
        "models": {},
    },
    "Experiment 2 — Controlled environmental rate": {
        "key": "exp2",
        "label": "Controlled rate",
        "target": "environmental_visits_per_1000_residents",
        "metrics": ROOT / "experiments/exp2_environmental_rate/results/benchmark_metrics.csv",
        "predictions": ROOT / "experiments/exp2_environmental_rate/results/test_predictions.csv",
        "importance": ROOT / "experiments/exp2_environmental_rate/results/top10_feature_importance.csv",
        "exclusions": ROOT / "experiments/exp2_environmental_rate/results/feature_exclusions.csv",
        "models": {
            "Dummy mean": ROOT / "experiments/exp2_environmental_rate/models/dummy_mean.joblib",
            "LightGBM (tuned)": ROOT
            / "experiments/exp2_environmental_rate/models/lightgbm_tuned.joblib",
            "Random forest": ROOT
            / "experiments/exp2_environmental_rate/models/random_forest.joblib",
            "Ridge (tuned)": ROOT
            / "experiments/exp2_environmental_rate/models/ridge_tuned.joblib",
        },
    },
    "Experiment 3 — Pure environmental ablation": {
        "key": "exp3",
        "label": "Pure environmental",
        "target": "environmental_visits_per_1000_residents",
        "metrics": ROOT
        / "experiments/exp3_pure_environmental/results/benchmark_metrics.csv",
        "predictions": ROOT
        / "experiments/exp3_pure_environmental/results/test_predictions.csv",
        "importance": ROOT
        / "experiments/exp3_pure_environmental/results/top10_feature_importance.csv",
        "exclusions": ROOT / "experiments/exp3_pure_environmental/feature_exclusions.csv",
        "models": {
            "Dummy mean": ROOT / "experiments/exp3_pure_environmental/models/dummy_mean.joblib",
            "LightGBM (tuned)": ROOT
            / "experiments/exp3_pure_environmental/models/lightgbm_tuned.joblib",
            "Random forest": ROOT
            / "experiments/exp3_pure_environmental/models/random_forest.joblib",
            "Ridge (tuned)": ROOT
            / "experiments/exp3_pure_environmental/models/ridge_tuned.joblib",
        },
    },
}

DISTRICT_ALIAS_PREFIX = "D"
DISTRICT_ALIAS_WIDTH = 2
DISTRICT_COLUMN = "district_key"
MONTH_COLUMN = "month"

KPI_REGISTRY = {
    "Registered residents": ("residents_registered", "sum"),
    "Daytime population": ("daytime_population", "sum"),
    "Active complaints": ("p4_open_requests_end_month", "sum"),
    "Mean AQI": ("p3_avg_aqi", "mean"),
    "Water pollution index": ("pw_water_pollution_index", "mean"),
    "Total clinical visits": ("total_visits_demand", "sum"),
}

TIME_SERIES_ENVIRONMENT_FIELDS = [
    "pw_water_age_hours",
    "p3_no2_ugm3",
    "p3_pm25_ugm3",
    "p2_uncollected_tons",
    "p3_avg_aqi",
    "pw_water_pollution_index",
]
TIME_SERIES_HEALTH_FIELDS = [
    "p3_respiratory_visits_total",
    "pw_waterborne_visits_total",
    "p3_heat_related_visits",
    "total_visits_demand",
]
VULNERABILITY_GROUPS = {
    "Demographics": ["elderly_pct", "children_pct"],
    "Complaints & service": [
        "p4_total_complaints",
        "p4_complaints_per_1000",
        "p4_avg_resolution_hours",
        "p4_sla_breach_pct",
        "p4_open_requests_end_month",
        "p4_service_need_index",
    ],
    "Environmental stress": [
        "esi_heat",
        "esi_air",
        "esi_water",
        "esi_waste",
        "environmental_stress_index",
    ],
}

FINANCIAL_ASSUMPTIONS = {
    "currency": "M EGP",
    "budget_min_m_egp": 5.0,
    "budget_max_m_egp": 100.0,
    "default_budget_m_egp": 45.0,
    "capacity_expansion_min_pct": 0,
    "capacity_expansion_max_pct": 50,
    "default_capacity_expansion_pct": 15,
    "population_growth_min_pct": -10,
    "population_growth_max_pct": 50,
    "default_population_growth_pct": 20,
    "prevention_reach_max_pct": 30,
    "default_prevention_reach_pct": 10,
    "access_improvement_min_minutes": -15,
    "access_improvement_max_minutes": 0,
    "default_access_improvement_minutes": -5,
    "new_center_capacity_pct": 35.0,
    "prevention_efficacy": 0.20,
    "triage_capacity_gain_per_minute": 0.02,
    "calibrated_pressure_min": 0.75,
    "calibrated_pressure_max": 1.35,
    "baseline_window_months": 6,
    "scenario_costs_m_egp": {
        "center_fixed": 24.0,
        "capacity_expansion_at_default": 9.0,
        "access_program_at_default": 16.0,
        "prevention_at_default": 7.5,
    },
}

SCENARIO_PACKAGES = {
    "A — New center + prevention": "center_prevention",
    "B — Expand existing capacity": "capacity",
    "C — Transport + triage access": "access",
    "D — Prevention only": "prevention",
}
