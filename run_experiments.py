"""Reproducible temporal benchmark for the municipal health-demand panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))

import lightgbm as lgb
import joblib
import numpy as np
import pandas as pd
import sklearn
from lightgbm.basic import LightGBMError
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "Data" / "Processed" / "master_panel.csv"
EXP1_DIR = ROOT / "experiments" / "exp1_macro_volume"
EXP1_RUN_DIR = ROOT / "experiments" / "reruns" / "exp1_macro_volume"
EXP1_DATA_DIR = EXP1_RUN_DIR / "data"
EXP1_RESULTS_DIR = EXP1_RUN_DIR / "results"
EXP2_DIR = ROOT / "experiments" / "exp2_environmental_rate"
EXP2_RUN_DIR = ROOT / "experiments" / "reruns" / "exp2_environmental_rate"
EXP2_DATA_DIR = EXP2_RUN_DIR / "data"
EXP2_MODEL_DIR = EXP2_RUN_DIR / "models"
EXP2_RESULTS_DIR = EXP2_RUN_DIR / "results"
EXP3_DIR = ROOT / "experiments" / "exp3_pure_environmental"
EXP3_DATA_DIR = EXP3_DIR / "data"
EXP3_MODEL_DIR = EXP3_DIR / "models"
EXP3_RESULTS_DIR = EXP3_DIR / "results"
EXP3_TARGET = "environmental_visits_per_1000_residents"
TARGET = "total_visits_demand"

ENVIRONMENTAL_COLUMNS = [
    "avg_temp_c",
    "heat_days_ge38c",
    "pw_metered_consumption_m3",
    "pw_water_supplied_m3",
    "pw_nrw_pct",
    "pw_night_flow_ratio",
    "pw_leak_incidents",
    "pw_supply_interruption_hours",
    "pw_water_age_hours",
    "pw_chlorine_residual_mg_l",
    "pw_turbidity_ntu",
    "pw_ecoli_positive_pct",
    "pw_sewage_overflow_incidents",
    "pw_water_pollution_index",
    "p2_bins_installed",
    "p2_avg_bin_fill_pct",
    "p2_bin_hours_ge85_pct",
    "p2_overflow_incidents",
    "p2_collection_efficiency_pct",
    "p2_collection_disruption_pp",
    "p2_waste_generated_tons",
    "p2_uncollected_tons",
    "p2_organic_tons",
    "p2_recyclable_tons",
    "p2_recycling_diversion_pct",
    "p2_odor_pest_risk_index",
    "p3_avg_aqi",
    "p3_max_aqi",
    "p3_days_aqi_gt150",
    "p3_pm25_ugm3",
    "p3_pm10_ugm3",
    "p3_no2_ugm3",
    "p3_o3_ugm3",
]
LAG_COLUMNS = ENVIRONMENTAL_COLUMNS + ["residents_registered", "households"]
LAG_FEATURES = [f"lag1_{column}" for column in LAG_COLUMNS]
CALENDAR_FEATURES = ["month_index", "month_sin", "month_cos"]
EXP2_ENVIRONMENTAL_COLUMNS = [
    "p3_avg_aqi",
    "p3_max_aqi",
    "p3_days_aqi_gt150",
    "p3_pm25_ugm3",
    "p3_pm10_ugm3",
    "p3_no2_ugm3",
    "p3_o3_ugm3",
    "pw_water_pollution_index",
    "pw_turbidity_ntu",
    "pw_chlorine_residual_mg_l",
    "pw_ecoli_positive_pct",
    "pw_sewage_overflow_incidents",
    "pw_supply_interruption_hours",
    "p2_odor_pest_risk_index",
    "p2_bin_hours_ge85_pct",
    "p2_overflow_incidents",
    "p2_collection_efficiency_pct",
    "avg_temp_c",
    "heat_days_ge38c",
]
EXP2_CONTROL_COLUMNS = ["children_pct", "elderly_pct"]
EXP2_LAG_COLUMNS = EXP2_ENVIRONMENTAL_COLUMNS + EXP2_CONTROL_COLUMNS
EXP2_NUMERATOR_COLUMNS = [
    "pw_waterborne_visits_total",
    "p3_respiratory_visits_total",
    "p3_heat_related_visits",
]
EXP2_DERIVED_EXCLUSIONS = {
    "nonresp_visits": "Non-environmental baseline demand; excluded from target",
    "total_visits_demand": "Contains non-environmental baseline and target components",
    "pc_visits_demand": "Derived 70% share of total demand",
    "cpr": "Derived from demand and capacity",
    "residents_registered": "Population denominator only; raw scale excluded from predictors",
    "households": "Raw population-scale confounder excluded from predictors",
    "daytime_population": "Raw population-scale confounder excluded from predictors",
    "p3_population_exposed": "Raw population-scale confounder and target-generation input",
}
EXP3_PHYSICAL_COLUMNS = [
    "p3_avg_aqi",
    "p3_max_aqi",
    "p3_days_aqi_gt150",
    "p3_pm25_ugm3",
    "p3_pm10_ugm3",
    "p3_no2_ugm3",
    "p3_o3_ugm3",
    "pw_water_pollution_index",
    "pw_chlorine_residual_mg_l",
    "pw_turbidity_ntu",
    "pw_ecoli_positive_pct",
    "pw_water_age_hours",
    "pw_leak_incidents",
    "pw_supply_interruption_hours",
    "pw_sewage_overflow_incidents",
    "pw_nrw_pct",
    "pw_night_flow_ratio",
    "pw_metered_consumption_m3",
    "pw_water_supplied_m3",
    "p2_avg_bin_fill_pct",
    "p2_bin_hours_ge85_pct",
    "p2_overflow_incidents",
    "p2_collection_efficiency_pct",
    "p2_collection_disruption_pp",
    "p2_odor_pest_risk_index",
    "p2_bins_installed",
    "p2_waste_generated_tons",
    "p2_uncollected_tons",
    "p2_organic_tons",
    "p2_recyclable_tons",
    "p2_recycling_diversion_pct",
    "avg_temp_c",
    "heat_days_ge38c",
]
EXP3_LAG_FEATURES = [f"lag1_{column}" for column in EXP3_PHYSICAL_COLUMNS]
EXP3_POPULATION_EXCLUSIONS = {
    "residents_registered": "Population denominator only; excluded from predictor matrix",
    "households": "Raw population-volume confounder excluded from predictor matrix",
    "daytime_population": "Raw population-volume confounder excluded from predictor matrix",
    "p3_population_exposed": "Raw population-volume confounder and outcome-generation input",
    "students_day": "Population-volume confounder excluded from predictor matrix",
    "daytime_workers_inflow": "Population-volume confounder excluded from predictor matrix",
    "density_per_km2": "Spatial/population-scale shortcut excluded from predictor matrix",
}
EXP3_SHORTCUT_EXCLUSIONS = {
    "district_key": "Spatial identity shortcut excluded for pure environmental ablation",
    "children_pct": "Demographic shortcut excluded for pure environmental ablation",
    "elderly_pct": "Demographic shortcut excluded for pure environmental ablation",
    "need_score": "Composite decision index excluded",
    "attractiveness_index_old": "Composite attractiveness index excluded",
    "attractiveness_index_v2": "Composite downstream index excluded",
    "p4_service_need_index": "Composite health/service index excluded",
    "p4_complaints_per_1000": "Service/complaint proxy excluded from pure environmental feature space",
}

LEAKAGE_REASONS = {
    "nonresp_visits": "Mathematical redundancy: additive target component",
    "p3_respiratory_visits_total": "Mathematical redundancy: additive target component",
    "pw_waterborne_visits_total": "Mathematical redundancy: additive target component",
    "p3_heat_related_visits": "Mathematical redundancy: additive target component",
    "p3_population_exposed": "Target leakage: direct input to generated nonresp_visits",
    "children_pct": "Target leakage: direct input to generated nonresp_visits",
    "elderly_pct": "Target leakage: direct input to generated nonresp_visits",
    "pc_visits_demand": "Mathematical redundancy: rounded 70% of target",
    "cpr": "Target leakage: derived from pc_visits_demand and capacity",
    "wait_days": "Zero variance and target-derived",
    "unmet_visits": "Zero variance and target-derived",
    "unmet_share": "Zero variance and target-derived",
    "service_quality": "Zero variance and target-derived",
    "need_score": "Target leakage: includes cpr",
    "attractiveness_index_v2": "Target leakage: downstream score includes service_quality",
}
ZERO_VARIANCE = {"wait_days", "unmet_visits", "unmet_share", "service_quality"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cpu_threads() -> int:
    return min(8, max(1, os.cpu_count() or 1))


def detect_nvidia_gpu() -> bool:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return False
    result = subprocess.run(
        [executable, "-L"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return result.returncode == 0 and "GPU" in result.stdout


def make_pipeline(estimator, numeric_features, scale_numeric=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_features),
            ("district", OneHotEncoder(handle_unknown="ignore"), ["district_key"]),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def make_lightgbm(device_type: str, threads: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=120,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=7,
        min_child_samples=20,
        reg_lambda=1.0,
        colsample_bytree=0.8,
        random_state=SEED,
        data_random_seed=SEED,
        feature_fraction_seed=SEED,
        bagging_seed=SEED,
        deterministic=True,
        force_col_wise=True,
        device_type=device_type,
        n_jobs=threads,
        verbosity=-1,
    )


def choose_lightgbm_device(X_train, y_train, numeric_features, threads):
    if not detect_nvidia_gpu():
        print("No NVIDIA GPU detected; LightGBM will use multi-core CPU.")
        return "cpu"

    for device_type in ("cuda", "gpu"):
        probe = make_pipeline(
            make_lightgbm(device_type, threads),
            numeric_features,
        )
        try:
            probe.fit(X_train.iloc[: min(30, len(X_train))], y_train.iloc[: min(30, len(y_train))])
            print(f"LightGBM GPU acceleration enabled with device_type={device_type}.")
            return device_type
        except LightGBMError as error:
            print(f"LightGBM device_type={device_type} unavailable: {error}")

    print("LightGBM GPU backends unavailable; falling back to multi-core CPU.")
    return "cpu"


def build_models(numeric_features, threads, lgbm_device):
    return {
        "Dummy mean": make_pipeline(
            DummyRegressor(strategy="mean"),
            numeric_features,
            scale_numeric=True,
        ),
        "Ridge": make_pipeline(
            Ridge(alpha=10.0),
            numeric_features,
            scale_numeric=True,
        ),
        "Random forest": make_pipeline(
            RandomForestRegressor(
                n_estimators=200,
                max_depth=3,
                min_samples_leaf=8,
                max_features=0.8,
                random_state=SEED,
                n_jobs=1,
            ),
            numeric_features,
        ),
        "LightGBM": make_pipeline(
            make_lightgbm(lgbm_device, threads),
            numeric_features,
        ),
    }


def tuning_grid(name):
    if name == "Ridge":
        return {"model__alpha": [0.1, 1.0, 10.0, 100.0]}
    if name == "Random forest":
        return {
            "model__max_depth": [2, 3],
            "model__min_samples_leaf": [8, 12],
        }
    if name == "LightGBM":
        return {
            "model__max_depth": [2, 3],
            "model__min_child_samples": [20, 30],
        }
    raise ValueError(f"No tunable parameter grid for {name!r}")


def exclusion_reason(column, dtype):
    if column == TARGET:
        return "Target label; never included in the feature matrix"
    if column in LEAKAGE_REASONS:
        return LEAKAGE_REASONS[column]
    if column in LAG_COLUMNS:
        return "Contemporaneous value withheld; only its t-1 lag is modeled"
    if column == "district_key":
        return "Used as a district categorical feature"
    if column == "month":
        return "Used only to create chronological split and calendar features"
    if column == "event_flags":
        return "Text event flags excluded from this numeric benchmark"
    if column in ZERO_VARIANCE:
        return "Zero variance across all rows"
    if dtype == "object":
        return "Not selected: non-numeric text outside the approved feature scope"
    return "Not selected: health/capacity/derived measure outside the environmental feature scope"


def markdown_table(frame, columns):
    table = frame[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *body])


def run_exp1():
    random.seed(SEED)
    np.random.seed(SEED)
    threads = cpu_threads()
    source_hash = file_sha256(INPUT)
    panel = pd.read_csv(INPUT)

    required = {
        "district_key",
        "month",
        TARGET,
        "p3_population_exposed",
        *LAG_COLUMNS,
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"Required columns missing from {INPUT}: {missing}")

    panel["month"] = pd.to_datetime(panel["month"], errors="raise")
    panel = panel.sort_values(["month", "district_key"]).reset_index(drop=True)
    if panel.duplicated(["district_key", "month"]).any():
        raise ValueError("Input panel has duplicate district-month keys.")
    months = panel["month"].drop_duplicates().sort_values().tolist()
    districts = sorted(panel["district_key"].unique())
    if len(panel) != 315 or len(months) != 21 or len(districts) != 15:
        raise ValueError(
            f"Expected a 315-row, 15-district, 21-month panel; found "
            f"{len(panel)} rows, {len(districts)} districts, {len(months)} months."
        )
    if (panel.groupby("month")["district_key"].nunique() != len(districts)).any():
        raise ValueError("Panel is not balanced: at least one month is missing a district.")

    panel["district_key"] = panel["district_key"].astype(str)
    lagged = (
        panel.groupby("district_key", sort=False)[LAG_COLUMNS]
        .shift(1)
        .rename(columns=dict(zip(LAG_COLUMNS, LAG_FEATURES)))
    )
    panel = pd.concat([panel, lagged], axis=1).copy()
    first_month = months[0]
    month_of_year = panel["month"].dt.month
    panel = panel.assign(
        month_index=(
            (panel["month"].dt.year - first_month.year) * 12
            + panel["month"].dt.month
            - first_month.month
            + 1
        ),
        month_sin=np.sin(2 * np.pi * month_of_year / 12),
        month_cos=np.cos(2 * np.pi * month_of_year / 12),
    )

    feature_columns = ["district_key", *CALENDAR_FEATURES, *LAG_FEATURES]
    valid = panel.dropna(subset=[*LAG_FEATURES, TARGET]).copy()
    test_start = months[-6]
    train = valid.loc[valid["month"] < test_start].copy()
    test = valid.loc[valid["month"] >= test_start].copy()
    if train.empty or test.empty:
        raise ValueError("Temporal split created an empty train or test partition.")

    X_train = train[feature_columns]
    y_train = train[TARGET].astype(float)
    X_test = test[feature_columns]
    y_test = test[TARGET].astype(float)
    numeric_features = [*CALENDAR_FEATURES, *LAG_FEATURES]
    cv = TimeSeriesSplit(n_splits=3, test_size=3 * len(districts))
    for train_idx, validation_idx in cv.split(X_train):
        if train.iloc[train_idx]["month"].max() >= train.iloc[validation_idx]["month"].min():
            raise ValueError("A temporal validation fold overlaps or reverses month order.")

    EXP1_DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXP1_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metadata_columns = ["month", "district_key", TARGET]
    train[[*metadata_columns, *feature_columns[1:]]].to_csv(
        EXP1_DATA_DIR / "train.csv", index=False
    )
    test[[*metadata_columns, *feature_columns[1:]]].to_csv(
        EXP1_DATA_DIR / "test.csv", index=False
    )

    exclusion_rows = [
        {
            "column": column,
            "dtype": str(panel[column].dtype),
            "justification": exclusion_reason(column, str(panel[column].dtype)),
        }
        for column in panel.columns
        if column not in feature_columns
    ]
    exclusions = pd.DataFrame(exclusion_rows)
    exclusions.to_csv(EXP1_RESULTS_DIR / "feature_exclusions.csv", index=False)

    lgbm_device = choose_lightgbm_device(X_train, y_train, numeric_features, threads)
    models = build_models(numeric_features, threads, lgbm_device)
    cv_results = []
    cv_scores = {}
    for name, model in models.items():
        results = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            },
            n_jobs=1,
            error_score="raise",
        )
        cv_rmse = -results["test_rmse"]
        cv_mae = -results["test_mae"]
        cv_r2 = results["test_r2"]
        cv_scores[name] = float(cv_rmse.mean())
        cv_results.append(
            {
                "model": name,
                "cv_rmse_mean": float(cv_rmse.mean()),
                "cv_rmse_std": float(cv_rmse.std()),
                "cv_mae_mean": float(cv_mae.mean()),
                "cv_r2_mean": float(cv_r2.mean()),
            }
        )
        print(f"{name}: expanding-window CV RMSE={cv_rmse.mean():.4f}")

    winner = min((name for name in models if name != "Dummy mean"), key=cv_scores.get)
    search = GridSearchCV(
        estimator=models[winner],
        param_grid=tuning_grid(winner),
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    tuned_model = search.best_estimator_
    print(f"Tuned CV winner: {winner}; parameters={search.best_params_}")

    prediction_rows = []
    metric_rows = []
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
    fitted_models[f"{winner} (tuned)"] = tuned_model

    for name, model in fitted_models.items():
        prediction = model.predict(X_test)
        metric_rows.append(
            {
                "model": name,
                "cv_rmse_mean": (
                    float(-search.best_score_)
                    if name == f"{winner} (tuned)"
                    else cv_scores[name]
                ),
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, prediction))),
                "test_mae": float(mean_absolute_error(y_test, prediction)),
                "test_r2": float(r2_score(y_test, prediction)),
            }
        )
        prediction_rows.extend(
            {
                "month": month,
                "district_key": district,
                "model": name,
                "actual_total_visits": float(actual),
                "predicted_total_visits": float(predicted),
            }
            for month, district, actual, predicted in zip(
                test["month"], test["district_key"], y_test, prediction
            )
        )

    metrics = pd.DataFrame(metric_rows).sort_values("test_rmse")
    metrics.to_csv(EXP1_RESULTS_DIR / "benchmark_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        EXP1_RESULTS_DIR / "test_predictions.csv", index=False
    )
    importance = permutation_importance(
        tuned_model,
        X_test,
        y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=20,
        random_state=SEED,
        n_jobs=1,
    )
    importance_table = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_table.head(10).to_csv(
        EXP1_RESULTS_DIR / "top10_feature_importance.csv", index=False
    )

    cv_table = pd.DataFrame(cv_results)
    report = [
        "# Benchmark Results",
        "",
        f"- Target: raw `{TARGET}` (per-exposed rates deferred pending source validation).",
        f"- Temporal test: {test_start:%Y-%m}–{months[-1]:%Y-%m}, {len(test)} rows.",
        f"- Training: {train['month'].min():%Y-%m}–{train['month'].max():%Y-%m}, {len(train)} rows.",
        f"- Seeds: {SEED}; LightGBM device: `{lgbm_device}`; CPU threads: {threads}.",
        f"- Tuned model selected using training-only expanding-window CV: `{winner}`.",
        f"- Best tuned parameters: `{json.dumps(search.best_params_, sort_keys=True)}`.",
        "",
        "## Cross-validation",
        "",
        markdown_table(
            cv_table,
            ["model", "cv_rmse_mean", "cv_rmse_std", "cv_mae_mean", "cv_r2_mean"],
        ),
        "",
        "## Held-out test",
        "",
        markdown_table(
            metrics,
            ["model", "cv_rmse_mean", "test_rmse", "test_mae", "test_r2"],
        ),
        "",
        "## Top 10 permutation importance",
        "",
        markdown_table(
            importance_table.head(10),
            ["feature", "importance_mean", "importance_std"],
        ),
        "",
        "Permutation importance is computed on the held-out timestamps. Correlated environmental predictors can share or mask importance; this table describes predictive contribution, not causal impact.",
        "",
        "## Feature exclusions",
        "",
        markdown_table(
            exclusions,
            ["column", "dtype", "justification"],
        ),
        "",
    ]
    (EXP1_RESULTS_DIR / "benchmark_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    manifest = {
        "input": str(INPUT.relative_to(ROOT)),
        "input_sha256": source_hash,
        "target": TARGET,
        "seed": SEED,
        "districts": len(districts),
        "months": [month.strftime("%Y-%m") for month in months],
        "training_months": [
            train["month"].min().strftime("%Y-%m"),
            train["month"].max().strftime("%Y-%m"),
        ],
        "test_months": [
            test["month"].min().strftime("%Y-%m"),
            test["month"].max().strftime("%Y-%m"),
        ],
        "winning_model_by_cv": winner,
        "best_parameters": search.best_params_,
        "lightgbm_device": lgbm_device,
        "cpu_threads": threads,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lgb.__version__,
    }
    if file_sha256(INPUT) != source_hash:
        raise RuntimeError("Input checksum changed during execution; source was modified.")
    (EXP1_RESULTS_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Results written to {EXP1_RESULTS_DIR.relative_to(ROOT)}")
    print(f"Source SHA-256 unchanged: {source_hash}")


def event_flag_columns(panel):
    if "event_flags" not in panel:
        raise ValueError("Required event_flags column is missing from the source panel.")
    tokens_by_row = panel["event_flags"].fillna("").astype(str).str.split("|")
    tokens = sorted(
        {
            token.strip()
            for row_tokens in tokens_by_row
            for token in row_tokens
            if token.strip()
        }
    )
    slugs = {
        token: re.sub(r"[^a-z0-9]+", "_", token.lower()).strip("_")
        for token in tokens
    }
    if len(set(slugs.values())) != len(slugs):
        raise ValueError("Distinct event flags collide after feature-name normalization.")
    flags = pd.DataFrame(
        {
            f"event_{slugs[token]}": tokens_by_row.apply(
                lambda row_tokens, token=token: int(token in {value.strip() for value in row_tokens})
            )
            for token in tokens
        },
        index=panel.index,
    )
    return flags


def exp2_lightgbm(device_type: str, threads: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=100,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=7,
        min_child_samples=10,
        reg_lambda=1.0,
        colsample_bytree=0.8,
        random_state=SEED,
        data_random_seed=SEED,
        feature_fraction_seed=SEED,
        bagging_seed=SEED,
        deterministic=True,
        force_col_wise=True,
        device_type=device_type,
        n_jobs=threads,
        verbosity=-1,
    )


def exp2_tune_lightgbm(X_train, y_train, numeric_features, threads):
    if not detect_nvidia_gpu():
        print("No NVIDIA GPU detected; Experiment 2 LightGBM will use CPU.")
        device = "cpu"
    else:
        device = "cpu"
        for candidate in ("cuda", "gpu"):
            probe = make_pipeline(
                exp2_lightgbm(candidate, threads),
                numeric_features,
                scale_numeric=True,
            )
            try:
                probe.fit(
                    X_train.iloc[: min(30, len(X_train))],
                    y_train.iloc[: min(30, len(y_train))],
                )
                device = candidate
                print(f"Experiment 2 LightGBM GPU enabled: device_type={device}.")
                break
            except LightGBMError as error:
                print(f"Experiment 2 LightGBM {candidate} unavailable: {error}")
        if device == "cpu":
            print("Experiment 2 LightGBM falling back to 8-thread CPU.")

    estimator = make_pipeline(
        exp2_lightgbm(device, threads),
        numeric_features,
        scale_numeric=True,
    )
    return GridSearchCV(
        estimator=estimator,
        param_grid={
            "model__max_depth": [2, 3],
            "model__min_child_samples": [10, 20],
        },
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        },
        refit="rmse",
        cv=TimeSeriesSplit(n_splits=3, test_size=45),
        n_jobs=1,
        error_score="raise",
    ), device


def write_experiment2_comparison(metrics, importance_table, manifest, train, test):
    exp1_metrics_path = EXP1_RESULTS_DIR / "benchmark_metrics.csv"
    if not exp1_metrics_path.exists():
        exp1_metrics_path = ROOT / "experiments" / "results" / "benchmark_metrics.csv"
    if not exp1_metrics_path.exists():
        raise FileNotFoundError(
            "Experiment 1 benchmark metrics are unavailable; run --experiment 1 first."
        )
    exp1_metrics = pd.read_csv(exp1_metrics_path)
    exp1_tuned = exp1_metrics.loc[exp1_metrics["model"] == "Ridge (tuned)"]
    if exp1_tuned.empty:
        raise ValueError(f"No tuned Ridge metrics found in {exp1_metrics_path}.")
    exp1_best = exp1_tuned.iloc[0]
    exp2_best = metrics.sort_values("test_rmse").iloc[0]

    top_exp1_path = EXP1_RESULTS_DIR / "top10_feature_importance.csv"
    if not top_exp1_path.exists():
        top_exp1_path = ROOT / "experiments" / "results" / "top10_feature_importance.csv"
    exp1_importance = pd.read_csv(top_exp1_path) if top_exp1_path.exists() else pd.DataFrame()
    exp1_top = exp1_importance["feature"].head(10).astype(str).tolist() if not exp1_importance.empty else []
    exp2_top = importance_table["feature"].head(10).astype(str).tolist()
    exp2_environmental_top = [
        feature for feature in exp2_top
        if feature.startswith(("lag1_p3_", "lag1_pw_", "lag1_avg_temp", "lag1_heat_days"))
    ]
    requested_driver_names = {
        "lag1_p3_pm25_ugm3": "PM2.5",
        "lag1_p3_avg_aqi": "Average AQI",
        "lag1_avg_temp_c": "Temperature",
        "lag1_pw_water_pollution_index": "Water pollution index",
    }
    ranked_drivers = (
        importance_table.set_index("feature")
        .reindex(requested_driver_names)
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    ranked_drivers.insert(
        0,
        "driver",
        [requested_driver_names.get(feature, feature) for feature in ranked_drivers["feature"]],
    )
    ranked_drivers.insert(
        1,
        "rank",
        [
            int(importance_table["feature"].tolist().index(feature) + 1)
            if feature in importance_table["feature"].tolist()
            else "not ranked"
            for feature in ranked_drivers["feature"]
        ],
    )

    performance = pd.DataFrame(
        [
            {
                "experiment": "Experiment 1: macro volume",
                "model": exp1_best["model"],
                "target": "total_visits_demand",
                "test_r2": exp1_best["test_r2"],
                "test_rmse": exp1_best["test_rmse"],
                "test_mae": exp1_best["test_mae"],
            },
            {
                "experiment": "Experiment 2: environmental rate",
                "model": exp2_best["model"],
                "target": "environmental_visits_per_1000_residents",
                "test_r2": exp2_best["test_r2"],
                "test_rmse": exp2_best["test_rmse"],
                "test_mae": exp2_best["test_mae"],
            },
        ]
    )
    report = [
        "# Comparative Decision Report",
        "",
        "## Executive summary",
        "",
        "Experiment 1 forecasts total simulated demand for capacity and staffing planning. Experiment 2 forecasts an environmentally linked encounter rate per 1,000 registered residents for environmental monitoring and surge preparedness. They answer different questions and their RMSE/MAE values are in different units; compare R² cautiously and do not treat either experiment as a causal estimate.",
        "",
        "## Best-model test performance",
        "",
        markdown_table(
            performance,
            ["experiment", "model", "target", "test_r2", "test_rmse", "test_mae"],
        ),
        "",
        "Experiment 1 is the macro-volume baseline from its preserved artifact set; Experiment 2 selects the best tuned/baseline model by the predeclared temporal holdout RMSE.",
        "",
        "## Feature-driver shift",
        "",
        f"- Experiment 1 top-ten features: {', '.join(f'`{item}`' for item in exp1_top) or 'not available'}. Demographic volume proxies (especially lagged registered residents and households) dominate its ranking.",
        f"- Experiment 2 top-ten features: {', '.join(f'`{item}`' for item in exp2_top)}.",
        f"- Experiment 2 top-ten environmental features: {', '.join(f'`{item}`' for item in exp2_environmental_top) or 'none'}; explicit PM2.5, AQI, temperature, and water-pollution ranks follow, with full results in `exp2_environmental_rate/results/feature_importance_all.csv`.",
        "",
        "Requested environmental driver ranks (from held-out permutation importance):",
        "",
        markdown_table(ranked_drivers, ["driver", "rank", "importance_mean", "importance_std"]),
        "",
        "The named air, temperature, and water-pollution features rank below district and demographic controls; near-zero or negative held-out permutation importance means they added little predictive signal in this benchmark. This is not evidence of no causal environmental effect.",
        "",
        "Experiment 2 excludes raw population counts from predictors, but registered population is its requested denominator. Districts 1009–1015 have marked resident-versus-exposed-population discrepancies in the source panel; the registered-resident rate is therefore a specified operational index, not a validated epidemiological incidence rate. Its large values can reflect denominator mismatch.",
        "",
        "## Dual-model deployment recommendation",
        "",
        "1. **Capacity and staffing budgets:** use Experiment 1's total-volume forecast as a planning input for facility capacity, workforce, and budget scenarios. Review forecast errors by district and month; retain operational buffers because the target is simulator-derived.",
        "2. **Environmental early warning and surge response:** use Experiment 2 as a monitoring signal alongside direct AQI, water-quality, and temperature thresholds. Trigger human review and preparedness actions rather than automatic resource allocation from model scores alone. Recalibrate only after validating encounter labels and the registered-population denominator against observed data.",
        "3. **Do not infer causal sensitivity from these benchmarks.** Both outcomes and some environmental-health components were generated in a synthetic linkage pipeline. Permutation importance and predictive performance measure associations within this dataset, not causal environmental effects.",
        "",
        "## Reproducibility",
        "",
        f"- Seed: {SEED}; training: {train['month'].min():%Y-%m}–{train['month'].max():%Y-%m} ({len(train)} district-months); test: {test['month'].min():%Y-%m}–{test['month'].max():%Y-%m} ({len(test)} district-months).",
        f"- Experiment 2 winner: `{manifest['winning_model_by_cv']}`; tuning: `{json.dumps(manifest['best_parameters'], sort_keys=True)}`; LightGBM device: `{manifest['lightgbm_device']}`.",
        "- Experiment 1 original artifacts are preserved under `experiments/exp1_macro_volume/original_artifacts/`; reproducible reruns are written separately.",
        "",
    ]
    (ROOT / "experiments" / "COMPARATIVE_DECISION_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )


def run_exp2():
    random.seed(SEED)
    np.random.seed(SEED)
    threads = 8
    source_hash = file_sha256(INPUT)
    panel = pd.read_csv(INPUT)
    required = {
        "district_key",
        "month",
        "residents_registered",
        "event_flags",
        *EXP2_LAG_COLUMNS,
        *EXP2_NUMERATOR_COLUMNS,
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"Required Experiment 2 columns missing: {missing}")

    panel["month"] = pd.to_datetime(panel["month"], errors="raise")
    panel = panel.sort_values(["month", "district_key"]).reset_index(drop=True)
    if panel.duplicated(["district_key", "month"]).any():
        raise ValueError("Input panel has duplicate district-month keys.")
    months = panel["month"].drop_duplicates().sort_values().tolist()
    districts = sorted(panel["district_key"].unique())
    if len(panel) != 315 or len(months) != 21 or len(districts) != 15:
        raise ValueError(
            f"Expected 315 rows, 15 districts, 21 months; found "
            f"{len(panel)}, {len(districts)}, {len(months)}."
        )
    if (panel["residents_registered"] <= 0).any():
        raise ValueError("Target denominator residents_registered must be positive.")

    numerator = panel[EXP2_NUMERATOR_COLUMNS].sum(axis=1)
    panel["environmental_visits_per_1000_residents"] = (
        numerator / panel["residents_registered"] * 1000
    )

    event_flags = event_flag_columns(panel)
    event_flag_lags = [f"lag1_{column}" for column in event_flags.columns]
    lagged_environment = (
        panel.groupby("district_key", sort=False)[EXP2_LAG_COLUMNS]
        .shift(1)
        .rename(columns={column: f"lag1_{column}" for column in EXP2_LAG_COLUMNS})
    )
    lagged_events = (
        event_flags.groupby(panel["district_key"], sort=False)
        .shift(1)
        .rename(columns={column: f"lag1_{column}" for column in event_flags.columns})
    )
    panel = pd.concat([panel, lagged_environment, lagged_events], axis=1).copy()
    first_month = months[0]
    month_of_year = panel["month"].dt.month
    panel = panel.assign(
        month_index=(
            (panel["month"].dt.year - first_month.year) * 12
            + panel["month"].dt.month
            - first_month.month
            + 1
        ),
        month_sin=np.sin(2 * np.pi * month_of_year / 12),
        month_cos=np.cos(2 * np.pi * month_of_year / 12),
    )

    lag_features = [
        *[f"lag1_{column}" for column in EXP2_LAG_COLUMNS],
        *event_flag_lags,
    ]
    feature_columns = ["district_key", *CALENDAR_FEATURES, *lag_features]
    target = "environmental_visits_per_1000_residents"
    valid = panel.dropna(subset=[*lag_features, target]).copy()
    test_start = months[-6]
    train = valid.loc[valid["month"] < test_start].copy()
    test = valid.loc[valid["month"] >= test_start].copy()
    if len(train) != 210 or len(test) != 90:
        raise ValueError(
            f"Expected the approved 210/90 split; got train={len(train)}, test={len(test)}."
        )

    X_train = train[feature_columns]
    y_train = train[target].astype(float)
    X_test = test[feature_columns]
    y_test = test[target].astype(float)
    numeric_features = [*CALENDAR_FEATURES, *lag_features]
    cv = TimeSeriesSplit(n_splits=3, test_size=45)
    for train_idx, validation_idx in cv.split(X_train):
        if train.iloc[train_idx]["month"].max() >= train.iloc[validation_idx]["month"].min():
            raise ValueError("A temporal validation fold overlaps or reverses month order.")
        if len(train.iloc[validation_idx]) % len(districts):
            raise ValueError("A temporal validation fold splits district-month groups.")

    EXP2_DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXP2_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EXP2_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    split_columns = ["month", "district_key", target, *feature_columns[1:]]
    train[split_columns].to_csv(EXP2_DATA_DIR / "train.csv", index=False)
    test[split_columns].to_csv(EXP2_DATA_DIR / "test.csv", index=False)

    excluded_direct = set(EXP2_NUMERATOR_COLUMNS) | {
        "p3_respiratory_clinic_visits",
        "p3_respiratory_er_visits",
        "pw_waterborne_clinic_visits",
        "pw_waterborne_er_visits",
        "p3_population_exposed",
    }
    exclusion_rows = []
    for column in panel.columns:
        if column in feature_columns:
            continue
        if column == target:
            reason = "Target variable; not used as a feature"
        elif column in ZERO_VARIANCE:
            reason = "Zero-variance invariant; derived from demand and capacity"
        elif column in EXP2_DERIVED_EXCLUSIONS:
            reason = EXP2_DERIVED_EXCLUSIONS[column]
        elif column in excluded_direct:
            reason = "Direct current-month target component or clinical subcomponent; excluded from predictors"
        elif column in EXP2_LAG_COLUMNS:
            reason = "Current-month value withheld; only the t-1 lag is modeled"
        elif column == "event_flags":
            reason = "Parsed to binary flags and lagged one month; raw text excluded"
        elif column == "district_key":
            reason = "Used as district categorical feature"
        elif column == "month":
            reason = "Used for chronological split and calendar features"
        elif column == "attractiveness_index_v2":
            reason = "Derived score may contain target descendants"
        elif str(panel[column].dtype) == "object":
            reason = "Text field not included as a model feature"
        else:
            reason = "Not selected; outside approved environmental and ratio-control feature scope"
        exclusion_rows.append(
            {"column": column, "dtype": str(panel[column].dtype), "justification": reason}
        )
    exclusion_rows.extend(
        [
            {
                "column": column,
                "dtype": str(panel[column].dtype),
                "justification": (
                    "Direct target component at time t; summed into target and excluded from features"
                ),
            }
            for column in EXP2_NUMERATOR_COLUMNS
        ]
    )
    exclusions = pd.DataFrame(exclusion_rows).drop_duplicates("column", keep="first")
    exclusions.to_csv(EXP2_RESULTS_DIR / "feature_exclusions.csv", index=False)
    exclusions.to_csv(EXP2_RUN_DIR / "feature_exclusions.csv", index=False)

    dummy = make_pipeline(
        DummyRegressor(strategy="mean"), numeric_features, scale_numeric=True
    ).fit(X_train, y_train)
    ridge_search = GridSearchCV(
        estimator=make_pipeline(
            Ridge(random_state=SEED), numeric_features, scale_numeric=True
        ),
        param_grid={"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        },
        refit="rmse",
        cv=cv,
        n_jobs=1,
        error_score="raise",
    ).fit(X_train, y_train)
    forest = make_pipeline(
        RandomForestRegressor(
            n_estimators=100,
            max_depth=3,
            min_samples_leaf=8,
            max_features=0.8,
            random_state=SEED,
            n_jobs=1,
        ),
        numeric_features,
        scale_numeric=True,
    )
    forest_cv = cross_validate(
        forest,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        },
        n_jobs=1,
        error_score="raise",
    )
    forest.fit(X_train, y_train)
    lightgbm_search, lgbm_device = exp2_tune_lightgbm(
        X_train, y_train, numeric_features, threads
    )
    lightgbm_search.fit(X_train, y_train)

    tuned_candidates = {
        "Ridge (tuned)": (ridge_search.best_estimator_, float(-ridge_search.best_score_)),
        "Random forest": (
            forest,
            float(-forest_cv["test_rmse"].mean()),
        ),
        "LightGBM (tuned)": (
            lightgbm_search.best_estimator_,
            float(-lightgbm_search.best_score_),
        ),
    }
    winner_name, (winner_model, _) = min(
        tuned_candidates.items(), key=lambda item: item[1][1]
    )
    models = {
        "Dummy mean": dummy,
        **{name: value[0] for name, value in tuned_candidates.items()},
    }
    model_filenames = {
        "Dummy mean": "dummy_mean.joblib",
        "Ridge (tuned)": "ridge_tuned.joblib",
        "Random forest": "random_forest.joblib",
        "LightGBM (tuned)": "lightgbm_tuned.joblib",
    }
    for name, model in models.items():
        joblib.dump(model, EXP2_MODEL_DIR / model_filenames[name], compress=3)
    dummy_cv = cross_validate(
            make_pipeline(
                DummyRegressor(strategy="mean"),
                numeric_features,
                scale_numeric=True,
            ),
            X_train,
            y_train,
            cv=cv,
            scoring={
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            },
            n_jobs=1,
            error_score="raise",
        )

    metric_rows = []
    prediction_rows = []
    for name, model in models.items():
        predictions = model.predict(X_test)
        if name == "Ridge (tuned)":
            cv_rmse = float(-ridge_search.best_score_)
            cv_mae = float(-ridge_search.cv_results_["mean_test_mae"][ridge_search.best_index_])
            cv_r2 = float(ridge_search.cv_results_["mean_test_r2"][ridge_search.best_index_])
        elif name == "LightGBM (tuned)":
            cv_rmse = float(-lightgbm_search.best_score_)
            cv_mae = float(-lightgbm_search.cv_results_["mean_test_mae"][lightgbm_search.best_index_])
            cv_r2 = float(lightgbm_search.cv_results_["mean_test_r2"][lightgbm_search.best_index_])
        elif name == "Random forest":
            cv_rmse = float(-forest_cv["test_rmse"].mean())
            cv_mae = float(-forest_cv["test_mae"].mean())
            cv_r2 = float(forest_cv["test_r2"].mean())
        else:
            cv_rmse = float(-dummy_cv["test_rmse"].mean())
            cv_mae = float(-dummy_cv["test_mae"].mean())
            cv_r2 = float(dummy_cv["test_r2"].mean())
        metric_rows.append(
            {
                "model": name,
                "cv_rmse_mean": cv_rmse,
                "cv_mae_mean": cv_mae,
                "cv_r2_mean": cv_r2,
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
                "test_mae": float(mean_absolute_error(y_test, predictions)),
                "test_r2": float(r2_score(y_test, predictions)),
            }
        )
        prediction_rows.extend(
            {
                "month": month,
                "district_key": district,
                "model": name,
                "actual_environmental_visits_per_1000_residents": float(actual),
                "predicted_environmental_visits_per_1000_residents": float(predicted),
            }
            for month, district, actual, predicted in zip(
                test["month"], test["district_key"], y_test, predictions
            )
        )

    metrics = pd.DataFrame(metric_rows).sort_values("test_rmse")
    metrics.to_csv(EXP2_RESULTS_DIR / "benchmark_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        EXP2_RESULTS_DIR / "test_predictions.csv", index=False
    )
    importance = permutation_importance(
        winner_model,
        X_test,
        y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=20,
        random_state=SEED,
        n_jobs=1,
    )
    importance_table = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_table.head(10).to_csv(
        EXP2_RESULTS_DIR / "top10_feature_importance.csv", index=False
    )
    importance_table.to_csv(
        EXP2_RESULTS_DIR / "feature_importance_all.csv", index=False
    )

    report = [
        "# Experiment 2: Environmental Health Intensity",
        "",
        f"- Target: `{target}` = 1,000 × (environment-linked encounters) / `residents_registered`.",
        "- Numerator: `pw_waterborne_visits_total + p3_respiratory_visits_total + p3_heat_related_visits`; `nonresp_visits` is excluded.",
        f"- Train: {train['month'].min():%Y-%m}–{train['month'].max():%Y-%m} ({len(train)} rows); test: {test['month'].min():%Y-%m}–{test['month'].max():%Y-%m} ({len(test)} rows).",
        f"- Fixed seed: {SEED}; LightGBM device: `{lgbm_device}`; CPU fallback: 8 threads.",
        f"- Winner selected on training-only expanding-window CV: `{winner_name}`.",
        f"- Ridge tuning: `{json.dumps(ridge_search.best_params_, sort_keys=True)}`; LightGBM tuning: `{json.dumps(lightgbm_search.best_params_, sort_keys=True)}`.",
        "",
        "The requested registered-resident denominator is used exactly, but has known discrepancies from `p3_population_exposed` in several districts. Treat the result as a specified rate index, not validated epidemiological incidence. Since the encounter components are synthetic in the source pipeline, results and feature importance are predictive associations, not causal effects.",
        "",
        "## Held-out benchmark",
        "",
        markdown_table(
            metrics,
            ["model", "test_rmse", "test_mae", "test_r2"],
        ),
        "",
        "## Top 10 permutation importance",
        "",
        markdown_table(
            importance_table.head(10),
            ["feature", "importance_mean", "importance_std"],
        ),
        "",
        "## Requested environmental drivers",
        "",
        markdown_table(
            pd.DataFrame(
                [
                    {
                        "feature": feature,
                        "rank": int(
                            importance_table["feature"].tolist().index(feature) + 1
                        ),
                        "importance_mean": float(
                            importance_table.loc[
                                importance_table["feature"] == feature,
                                "importance_mean",
                            ].iloc[0]
                        ),
                    }
                    for feature in [
                        "lag1_p3_pm25_ugm3",
                        "lag1_p3_avg_aqi",
                        "lag1_avg_temp_c",
                        "lag1_pw_water_pollution_index",
                    ]
                ]
            ),
            ["feature", "rank", "importance_mean"],
        ),
        "",
        "Full permutation rankings are saved to `feature_importance_all.csv`.",
        "",
        "The named air, temperature, and water-pollution variables add little held-out predictive signal relative to district and age-share controls in this benchmark. This does not establish that environmental factors have no causal effect.",
        "",
        "All environmental features are lagged one month; current-month encounter components, baseline demand, target descendants, raw population counts, and raw event text are excluded from predictors.",
        "",
        "## Feature exclusions",
        "",
        markdown_table(exclusions, ["column", "dtype", "justification"]),
        "",
    ]
    (EXP2_RESULTS_DIR / "benchmark_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    manifest = {
        "input": str(INPUT.relative_to(ROOT)),
        "input_sha256": source_hash,
        "target": target,
        "target_formula": (
            "1000 * (pw_waterborne_visits_total + p3_respiratory_visits_total "
            "+ p3_heat_related_visits) / residents_registered"
        ),
        "seed": SEED,
        "districts": len(districts),
        "months": [month.strftime("%Y-%m") for month in months],
        "training_months": [
            train["month"].min().strftime("%Y-%m"),
            train["month"].max().strftime("%Y-%m"),
        ],
        "test_months": [
            test["month"].min().strftime("%Y-%m"),
            test["month"].max().strftime("%Y-%m"),
        ],
        "training_rows": len(train),
        "test_rows": len(test),
        "winning_model_by_cv": winner_name,
        "best_parameters": {
            "ridge": ridge_search.best_params_,
            "lightgbm": lightgbm_search.best_params_,
        },
        "lightgbm_device": lgbm_device,
        "cpu_threads": threads,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lgb.__version__,
    }
    if file_sha256(INPUT) != source_hash:
        raise RuntimeError("Input checksum changed during execution; source was modified.")
    (EXP2_RESULTS_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    log_lines = [
        "Experiment 2 deterministic run",
        f"Input SHA-256: {source_hash}",
        f"Target: {target}",
        f"Seed: {SEED}",
        f"Training rows: {len(train)} ({train['month'].min():%Y-%m} to {train['month'].max():%Y-%m})",
        f"Test rows: {len(test)} ({test['month'].min():%Y-%m} to {test['month'].max():%Y-%m})",
        f"LightGBM device: {lgbm_device}; CPU threads: {threads}",
        f"Winning model by CV: {winner_name}",
        f"Best parameters: {json.dumps(manifest['best_parameters'], sort_keys=True)}",
        "",
        metrics.to_csv(index=False).rstrip(),
        "",
    ]
    (EXP2_RESULTS_DIR / "run.log").write_text(
        "\n".join(log_lines), encoding="utf-8"
    )
    write_experiment2_comparison(metrics, importance_table, manifest, train, test)
    print(f"Experiment 2 results written to {EXP2_RESULTS_DIR.relative_to(ROOT)}")
    print(f"Source SHA-256 unchanged: {source_hash}")


def exp3_numeric_pipeline(estimator, scale_features=False):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_features:
        steps.append(("scaler", StandardScaler()))
    return Pipeline([("preprocess", Pipeline(steps)), ("model", estimator)])


def exp3_tune_lightgbm(X_train, y_train, threads):
    if not detect_nvidia_gpu():
        print("No NVIDIA GPU detected; Experiment 3 LightGBM will use CPU.")
        device = "cpu"
    else:
        device = "cpu"
        for candidate in ("cuda", "gpu"):
            probe = exp3_numeric_pipeline(
                exp2_lightgbm(candidate, threads), scale_features=True
            )
            try:
                probe.fit(
                    X_train.iloc[: min(30, len(X_train))],
                    y_train.iloc[: min(30, len(y_train))],
                )
                device = candidate
                print(f"Experiment 3 LightGBM GPU enabled: device_type={device}.")
                break
            except LightGBMError as error:
                print(f"Experiment 3 LightGBM {candidate} unavailable: {error}")
        if device == "cpu":
            print("Experiment 3 LightGBM falling back to 8-thread CPU.")

    estimator = exp3_numeric_pipeline(
        exp2_lightgbm(device, threads), scale_features=True
    )
    search = GridSearchCV(
        estimator=estimator,
        param_grid={
            "model__max_depth": [2, 3],
            "model__min_child_samples": [10, 20],
        },
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        },
        refit="rmse",
        cv=TimeSeriesSplit(n_splits=3, test_size=45),
        n_jobs=1,
        error_score="raise",
    )
    return search, device


def write_triple_benchmark_synthesis(
    exp3_metrics: pd.DataFrame,
    exp3_importance: pd.DataFrame,
    exp3_manifest: dict,
) -> None:
    exp1_metrics_path = EXP1_DIR / "original_artifacts" / "results" / "benchmark_metrics.csv"
    exp1_importance_path = EXP1_DIR / "original_artifacts" / "results" / "top10_feature_importance.csv"
    exp2_metrics_path = EXP2_DIR / "results" / "benchmark_metrics.csv"
    exp2_importance_path = EXP2_DIR / "results" / "top10_feature_importance.csv"
    for path in (exp1_metrics_path, exp2_metrics_path, exp1_importance_path, exp2_importance_path):
        if not path.exists():
            raise FileNotFoundError(f"Cannot synthesize three benchmarks; required artifact missing: {path}")

    exp1_metrics = pd.read_csv(exp1_metrics_path)
    exp2_metrics = pd.read_csv(exp2_metrics_path)
    exp1_best = exp1_metrics.loc[exp1_metrics["model"] == "Ridge (tuned)"]
    if exp1_best.empty:
        raise ValueError("Experiment 1 artifact has no tuned Ridge metrics.")
    exp1_best = exp1_best.iloc[0]
    exp2_manifest_path = EXP2_DIR / "results" / "run_manifest.json"
    exp2_winner = (
        json.loads(exp2_manifest_path.read_text(encoding="utf-8"))["winning_model_by_cv"]
        if exp2_manifest_path.exists()
        else exp2_metrics.sort_values("test_rmse").iloc[0]["model"]
    )
    exp2_best = exp2_metrics.loc[exp2_metrics["model"] == exp2_winner].iloc[0]
    exp3_best = exp3_metrics.loc[exp3_metrics["model"] == exp3_manifest["winning_model_by_cv"]]
    if exp3_best.empty:
        raise ValueError("Experiment 3 winner is missing from its held-out metrics.")
    exp3_best = exp3_best.iloc[0]

    unified = pd.DataFrame(
        [
            {
                "experiment": "Exp 1: Macro volume",
                "cv_selected_model": exp1_best["model"],
                "target": "Total simulated visits",
                "test_rmse": exp1_best["test_rmse"],
                "test_mae": exp1_best["test_mae"],
                "test_r2": exp1_best["test_r2"],
                "metric_units": "visits / district-month",
            },
            {
                "experiment": "Exp 2: Environmental rate",
                "cv_selected_model": exp2_best["model"],
                "target": "Environmental visits / 1,000 registered residents",
                "test_rmse": exp2_best["test_rmse"],
                "test_mae": exp2_best["test_mae"],
                "test_r2": exp2_best["test_r2"],
                "metric_units": "rate points / 1,000 registered residents",
            },
            {
                "experiment": "Exp 3: Pure environmental ablation",
                "cv_selected_model": exp3_best["model"],
                "target": "Environmental visits / 1,000 registered residents",
                "test_rmse": exp3_best["test_rmse"],
                "test_mae": exp3_best["test_mae"],
                "test_r2": exp3_best["test_r2"],
                "metric_units": "rate points / 1,000 registered residents",
            },
        ]
    )
    exp1_importance = pd.read_csv(exp1_importance_path)
    exp2_importance = pd.read_csv(exp2_importance_path)
    exp1_top = exp1_importance["feature"].head(10).astype(str).tolist()
    exp2_top = exp2_importance["feature"].head(10).astype(str).tolist()
    exp3_top = exp3_importance["feature"].head(10).astype(str).tolist()
    exp3_requested = [
        "lag1_p3_no2_ugm3",
        "lag1_pw_water_age_hours",
        "lag1_p3_pm25_ugm3",
        "lag1_p3_avg_aqi",
        "lag1_avg_temp_c",
        "lag1_pw_water_pollution_index",
    ]
    exp3_rank_rows = []
    for feature in exp3_requested:
        matches = exp3_importance.loc[exp3_importance["feature"] == feature]
        exp3_rank_rows.append(
            {
                "feature": feature,
                "rank": int(matches.index[0] + 1) if not matches.empty else "not ranked",
                "importance_mean": (
                    float(matches.iloc[0]["importance_mean"])
                    if not matches.empty
                    else "n/a"
                ),
            }
        )
    exp3_rank_table = pd.DataFrame(exp3_rank_rows)
    report = [
        "# Triple-Benchmark Synthesis Report",
        "",
        "## Executive summary",
        "",
        "The three benchmarks form a planning ladder, not a causal identification design. Experiment 1 predicts overall simulated visit volume; Experiment 2 predicts the specified environmental-encounter rate while retaining district and demographic controls; Experiment 3 removes spatial and demographic shortcuts and tests prediction using lagged physical environmental/infrastructure stressors alone.",
        "",
        "The target is synthetic in the available source pipeline, and the registered-resident denominator differs substantially from `p3_population_exposed` in some districts. Treat Experiments 2–3 rates as the requested analytical index pending label and denominator validation.",
        "",
        "## Unified held-out performance",
        "",
        markdown_table(
            unified,
            ["experiment", "cv_selected_model", "target", "test_rmse", "test_mae", "test_r2", "metric_units"],
        ),
        "",
        "Model rows are selected using expanding-window CV on the training period; the test split is reserved for final reporting. RMSE and MAE for Exp 1 are in absolute visits and must not be compared numerically to rate-point errors in Exp 2–3. R² is also affected by the differing target construction and predictor restrictions; compare it as within-experiment performance, not as a causal sensitivity score.",
        "",
        "## Feature-importance evolution",
        "",
        f"- **Exp 1 — demographic volume:** top features were {', '.join(f'`{item}`' for item in exp1_top)}. Lagged registered residents and households led, reflecting volume scale.",
        f"- **Exp 2 — spatial/age controls:** top features were {', '.join(f'`{item}`' for item in exp2_top)}. District identity and age composition led; environmental measures added comparatively little importance.",
        f"- **Exp 3 — pure environmental ablation:** top features were {', '.join(f'`{item}`' for item in exp3_top)}. Only lagged physical conditions and event indicators are eligible; neither district nor demographic predictors can act as shortcuts.",
        "",
        "Requested physical driver ranks in Experiment 3:",
        "",
        markdown_table(exp3_rank_table, ["feature", "rank", "importance_mean"]),
        "",
        "A high rank indicates predictive contribution in this held-out sample, not causation. Correlated, seasonal, and generated features can obscure or redistribute permutation importance. Negative or near-zero importance means the feature did not improve held-out prediction under this model.",
        "",
        "## Three-tier urban governance use",
        "",
        "1. **Tier 1 — long-term facility budgets:** use Exp 1 total-volume forecasts to scenario-plan clinic capacity, staffing, and budget envelopes. Validate with observed facility utilization and retain uncertainty buffers.",
        "2. **Tier 2 — structural district vulnerability ranking:** use Exp 2's district/age-controlled rate benchmark as a screening and prioritization aid, not an individual or causal risk score. Confirm population denominators and clinical labels before policy allocation.",
        "3. **Tier 3 — environmental surge response:** use Exp 3 as a physical-stressor monitoring layer alongside direct air, water, heat, and event thresholds. Where the model detects useful signal, trigger human review, sampling, inspections, and preparedness—not autonomous clinical or punitive decisions.",
        "",
        "Exp 3 is the strongest ablation of population/spatial shortcut pathways available in this panel, but it does not estimate a causal effect: outcomes and encounter components are simulated, and a 15-district/21-month panel cannot by itself identify causal environmental sensitivity.",
        "",
        "## Reproducibility and preservation",
        "",
        f"- Experiment 3 seed: {SEED}; training rows: {exp3_manifest['training_rows']} ({exp3_manifest['training_months'][0]}–{exp3_manifest['training_months'][1]}); test rows: {exp3_manifest['test_rows']} ({exp3_manifest['test_months'][0]}–{exp3_manifest['test_months'][1]}).",
        f"- Exp 3 CV winner: `{exp3_manifest['winning_model_by_cv']}`; LightGBM device: `{exp3_manifest['lightgbm_device']}`.",
        "- Experiment 1 and 2 artifact directories were read-only during Experiment 3. The source panel was also read-only; run manifests record the input checksum.",
        "- Reproduce Experiment 3 with `python run_experiments.py --experiment 3`; run all sequentially with `python run_experiments.py --experiment all`.",
        "",
    ]
    (ROOT / "experiments" / "TRIPLE_BENCHMARK_SYNTHESIS_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )


def run_exp3():
    random.seed(SEED)
    np.random.seed(SEED)
    threads = 8
    source_hash = file_sha256(INPUT)
    panel = pd.read_csv(INPUT)
    required = {
        "district_key",
        "month",
        "residents_registered",
        "event_flags",
        *EXP3_PHYSICAL_COLUMNS,
        "pw_waterborne_visits_total",
        "p3_respiratory_visits_total",
        "p3_heat_related_visits",
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"Required Experiment 3 columns missing: {missing}")

    panel["month"] = pd.to_datetime(panel["month"], errors="raise")
    panel = panel.sort_values(["month", "district_key"]).reset_index(drop=True)
    if panel.duplicated(["district_key", "month"]).any():
        raise ValueError("Input panel has duplicate district-month keys.")
    months = panel["month"].drop_duplicates().sort_values().tolist()
    districts = sorted(panel["district_key"].unique())
    if len(panel) != 315 or len(months) != 21 or len(districts) != 15:
        raise ValueError(
            f"Expected 315 rows, 15 districts, 21 months; found "
            f"{len(panel)}, {len(districts)}, {len(months)}."
        )
    if (panel["residents_registered"] <= 0).any():
        raise ValueError("Target denominator residents_registered must be positive.")

    target = EXP3_TARGET
    numerator = panel[
        [
            "pw_waterborne_visits_total",
            "p3_respiratory_visits_total",
            "p3_heat_related_visits",
        ]
    ].sum(axis=1)
    panel[target] = numerator / panel["residents_registered"] * 1000

    event_flags = event_flag_columns(panel)
    event_lag_columns = [f"lag1_{column}" for column in event_flags.columns]
    lagged_physical = (
        panel.groupby("district_key", sort=False)[EXP3_PHYSICAL_COLUMNS]
        .shift(1)
        .rename(columns={column: f"lag1_{column}" for column in EXP3_PHYSICAL_COLUMNS})
    )
    lagged_events = (
        event_flags.groupby(panel["district_key"], sort=False)
        .shift(1)
        .rename(columns={column: f"lag1_{column}" for column in event_flags.columns})
    )
    panel = pd.concat([panel, lagged_physical, lagged_events], axis=1).copy()
    feature_columns = [*EXP3_LAG_FEATURES, *event_lag_columns]
    if any(column in {"district_key", "children_pct", "elderly_pct", "month_index", "month_sin", "month_cos"} for column in feature_columns):
        raise AssertionError("Experiment 3 feature space contains a prohibited shortcut.")
    if any(column not in {f"lag1_{name}" for name in [*EXP3_PHYSICAL_COLUMNS, *event_flags.columns]} for column in feature_columns):
        raise AssertionError("Experiment 3 features must be lagged physical or event variables.")

    valid = panel.dropna(subset=[*feature_columns, target]).copy()
    test_start = months[-6]
    train = valid.loc[valid["month"] < test_start].copy()
    test = valid.loc[valid["month"] >= test_start].copy()
    if len(train) != 210 or len(test) != 90:
        raise ValueError(
            f"Expected fixed 210/90 split; got train={len(train)}, test={len(test)}."
        )
    X_train = train[feature_columns]
    y_train = train[target].astype(float)
    X_test = test[feature_columns]
    y_test = test[target].astype(float)
    cv = TimeSeriesSplit(n_splits=3, test_size=45)
    for train_idx, validation_idx in cv.split(X_train):
        train_months = train.iloc[train_idx]["month"]
        validation_months = train.iloc[validation_idx]["month"]
        if train_months.max() >= validation_months.min():
            raise ValueError("Temporal CV fold overlaps or reverses month order.")
        if len(validation_idx) % len(districts):
            raise ValueError("Temporal CV fold splits district groups within a month.")

    EXP3_DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXP3_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EXP3_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    split_columns = ["month", "district_key", target, *feature_columns]
    train[split_columns].to_csv(EXP3_DATA_DIR / "train.csv", index=False)
    test[split_columns].to_csv(EXP3_DATA_DIR / "test.csv", index=False)

    target_components = {
        "pw_waterborne_clinic_visits",
        "pw_waterborne_er_visits",
        "pw_waterborne_visits_total",
        "p3_respiratory_clinic_visits",
        "p3_respiratory_er_visits",
        "p3_respiratory_visits_total",
        "p3_heat_related_visits",
    }
    derived_identities = {
        "nonresp_visits": "Non-environmental baseline excluded from the target",
        "total_visits_demand": "Total-volume identity includes non-environmental baseline",
        "pc_visits_demand": "Derived 70% share of total demand",
        "cpr": "Derived from target-related demand and capacity",
        "unmet_visits": "Derived from demand and capacity",
        "unmet_share": "Derived from demand and capacity",
    }
    exclusions = []
    for column in panel.columns:
        if column in feature_columns:
            continue
        dtype = str(panel[column].dtype)
        if column == target:
            reason = "Experiment 3 target; never included as a predictor"
        elif column in target_components or column.startswith(("pw_waterborne_", "p3_respiratory_")):
            reason = "Direct current-month environmental target component; target lineage leakage"
        elif column == "p3_heat_related_visits":
            reason = "Direct current-month environmental target component; target lineage leakage"
        elif column in derived_identities:
            reason = derived_identities[column]
        elif column in ZERO_VARIANCE:
            reason = "Zero-variance invariant across all panel rows"
        elif column in EXP3_POPULATION_EXCLUSIONS:
            reason = EXP3_POPULATION_EXCLUSIONS[column]
        elif column in EXP3_SHORTCUT_EXCLUSIONS:
            reason = EXP3_SHORTCUT_EXCLUSIONS[column]
        elif column in EXP3_PHYSICAL_COLUMNS:
            reason = "Current-month physical value withheld; only the t-1 lag is modeled"
        elif column == "event_flags":
            reason = "Raw text parsed to binary event flags; only t-1 flags are modeled"
        elif column == "district_key":
            reason = EXP3_SHORTCUT_EXCLUSIONS[column]
        elif column == "month":
            reason = "Used only to sort and form fixed chronological partitions"
        elif str(panel[column].dtype) == "object":
            reason = "Non-numeric text excluded from the pure physical feature space"
        else:
            reason = "Not selected; outside the explicitly approved physical feature set"
        exclusions.append({"column": column, "dtype": dtype, "justification": reason})
    exclusions.extend(
        [
            {
                "column": column,
                "dtype": str(panel[column].dtype),
                "justification": "Direct current-month target component; included only in target numerator",
            }
            for column in (
                "pw_waterborne_visits_total",
                "p3_respiratory_visits_total",
                "p3_heat_related_visits",
            )
            if column not in panel.columns
        ]
    )
    exclusions = pd.DataFrame(exclusions).drop_duplicates("column", keep="first")
    exclusions.to_csv(EXP3_DIR / "feature_exclusions.csv", index=False)
    exclusions.to_csv(EXP3_RESULTS_DIR / "feature_exclusions.csv", index=False)

    dummy_cv = cross_validate(
        exp3_numeric_pipeline(DummyRegressor(strategy="mean"), scale_features=True),
        X_train,
        y_train,
        cv=cv,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        },
        n_jobs=1,
        error_score="raise",
    )
    dummy = exp3_numeric_pipeline(
        DummyRegressor(strategy="mean"), scale_features=True
    ).fit(X_train, y_train)

    ridge_search = GridSearchCV(
        estimator=exp3_numeric_pipeline(Ridge(), scale_features=True),
        param_grid={"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        },
        refit="rmse",
        cv=cv,
        n_jobs=1,
        error_score="raise",
    ).fit(X_train, y_train)

    forest = exp3_numeric_pipeline(
        RandomForestRegressor(
            n_estimators=100,
            max_depth=3,
            min_samples_leaf=8,
            max_features=0.8,
            random_state=SEED,
            n_jobs=1,
        )
    )
    forest_cv = cross_validate(
        forest,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        },
        n_jobs=1,
        error_score="raise",
    )
    forest.fit(X_train, y_train)

    lightgbm_search, lgbm_device = exp3_tune_lightgbm(
        X_train, y_train, threads
    )
    lightgbm_search.fit(X_train, y_train)
    candidates = {
        "Ridge (tuned)": (
            ridge_search.best_estimator_,
            float(-ridge_search.best_score_),
        ),
        "Random forest": (forest, float(-forest_cv["test_rmse"].mean())),
        "LightGBM (tuned)": (
            lightgbm_search.best_estimator_,
            float(-lightgbm_search.best_score_),
        ),
    }
    winner_name, (winner_model, _) = min(candidates.items(), key=lambda item: item[1][1])
    models = {
        "Dummy mean": dummy,
        **{name: candidate[0] for name, candidate in candidates.items()},
    }
    model_files = {
        "Dummy mean": "dummy_mean.joblib",
        "Ridge (tuned)": "ridge_tuned.joblib",
        "Random forest": "random_forest.joblib",
        "LightGBM (tuned)": "lightgbm_tuned.joblib",
    }
    for name, model in models.items():
        joblib.dump(model, EXP3_MODEL_DIR / model_files[name], compress=3)

    cv_rows = {
        "Dummy mean": {
            "cv_rmse_mean": float(-dummy_cv["test_rmse"].mean()),
            "cv_mae_mean": float(-dummy_cv["test_mae"].mean()),
            "cv_r2_mean": float(dummy_cv["test_r2"].mean()),
        },
        "Ridge (tuned)": {
            "cv_rmse_mean": float(-ridge_search.best_score_),
            "cv_mae_mean": float(-ridge_search.cv_results_["mean_test_mae"][ridge_search.best_index_]),
            "cv_r2_mean": float(ridge_search.cv_results_["mean_test_r2"][ridge_search.best_index_]),
        },
        "Random forest": {
            "cv_rmse_mean": float(-forest_cv["test_rmse"].mean()),
            "cv_mae_mean": float(-forest_cv["test_mae"].mean()),
            "cv_r2_mean": float(forest_cv["test_r2"].mean()),
        },
        "LightGBM (tuned)": {
            "cv_rmse_mean": float(-lightgbm_search.best_score_),
            "cv_mae_mean": float(-lightgbm_search.cv_results_["mean_test_mae"][lightgbm_search.best_index_]),
            "cv_r2_mean": float(lightgbm_search.cv_results_["mean_test_r2"][lightgbm_search.best_index_]),
        },
    }
    metric_rows = []
    prediction_rows = []
    for name, model in models.items():
        prediction = model.predict(X_test)
        metric_rows.append(
            {
                "model": name,
                **cv_rows[name],
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, prediction))),
                "test_mae": float(mean_absolute_error(y_test, prediction)),
                "test_r2": float(r2_score(y_test, prediction)),
            }
        )
        prediction_rows.extend(
            {
                "month": month,
                "district_key": district,
                "model": name,
                "actual_environmental_visits_per_1000_residents": float(actual),
                "predicted_environmental_visits_per_1000_residents": float(predicted),
            }
            for month, district, actual, predicted in zip(
                test["month"], test["district_key"], y_test, prediction
            )
        )

    metrics = pd.DataFrame(metric_rows).sort_values("test_rmse")
    metrics.to_csv(EXP3_RESULTS_DIR / "benchmark_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        EXP3_RESULTS_DIR / "test_predictions.csv", index=False
    )
    importance = permutation_importance(
        winner_model,
        X_test,
        y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=20,
        random_state=SEED,
        n_jobs=1,
    )
    importance_table = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    importance_table.insert(0, "rank", np.arange(1, len(importance_table) + 1))
    importance_table.head(10).to_csv(
        EXP3_RESULTS_DIR / "top10_feature_importance.csv", index=False
    )
    importance_table.to_csv(
        EXP3_RESULTS_DIR / "feature_importance_all.csv", index=False
    )

    manifest = {
        "input": str(INPUT.relative_to(ROOT)),
        "input_sha256": source_hash,
        "target": target,
        "target_formula": (
            "1000 * (pw_waterborne_visits_total + p3_respiratory_visits_total "
            "+ p3_heat_related_visits) / residents_registered"
        ),
        "seed": SEED,
        "districts": len(districts),
        "months": [month.strftime("%Y-%m") for month in months],
        "training_months": [
            train["month"].min().strftime("%Y-%m"),
            train["month"].max().strftime("%Y-%m"),
        ],
        "test_months": [
            test["month"].min().strftime("%Y-%m"),
            test["month"].max().strftime("%Y-%m"),
        ],
        "training_rows": len(train),
        "test_rows": len(test),
        "feature_columns": feature_columns,
        "winning_model_by_cv": winner_name,
        "best_parameters": {
            "ridge": ridge_search.best_params_,
            "lightgbm": lightgbm_search.best_params_,
        },
        "lightgbm_device": lgbm_device,
        "cpu_threads": threads,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lgb.__version__,
    }
    if file_sha256(INPUT) != source_hash:
        raise RuntimeError("Input checksum changed during Experiment 3; source was modified.")
    (EXP3_RESULTS_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (EXP3_RESULTS_DIR / "run.log").write_text(
        "\n".join(
            [
                "Experiment 3 deterministic pure-environmental ablation",
                f"Input SHA-256: {source_hash}",
                f"Target: {target}",
                f"Seed: {SEED}",
                f"Training rows: {len(train)} ({train['month'].min():%Y-%m} to {train['month'].max():%Y-%m})",
                f"Test rows: {len(test)} ({test['month'].min():%Y-%m} to {test['month'].max():%Y-%m})",
                f"Eligible physical lag features: {len(feature_columns)}",
                f"Winner by temporal CV: {winner_name}",
                f"LightGBM device: {lgbm_device}; CPU threads: {threads}",
                "",
                metrics.to_csv(index=False).rstrip(),
                "",
            ]
        ),
        encoding="utf-8",
    )
    report = [
        "# Experiment 3: Pure Environmental Ablation",
        "",
        f"- Target: `{target}` = 1,000 × (waterborne + respiratory + heat-related visits) / registered residents.",
        "- `nonresp_visits` is excluded from the target; its value is not an input.",
        f"- Temporal split: train {train['month'].min():%Y-%m}–{train['month'].max():%Y-%m} ({len(train)} rows); test {test['month'].min():%Y-%m}–{test['month'].max():%Y-%m} ({len(test)} rows).",
        f"- Seed: {SEED}; CV-selected model: `{winner_name}`; LightGBM device: `{lgbm_device}`.",
        f"- Predictor matrix: {len(feature_columns)} lagged physical/environmental and event features only; no district, demographic, target-component, population-scale, or composite-index predictors.",
        f"- Lowest held-out RMSE was `{metrics.sort_values('test_rmse').iloc[0]['model']}`; the reported feature importance and synthesis use the training-CV-selected model `{winner_name}` to avoid selecting a winner on the test set.",
        "",
        "The rate denominator is used exactly as specified but was not validated against `p3_population_exposed`; several districts have large denominator discrepancies. Clinical encounters are simulator-generated in the available pipeline. This is a predictive feature ablation and not a causal sensitivity estimate.",
        "",
        "## Held-out benchmark",
        "",
        markdown_table(metrics, ["model", "test_rmse", "test_mae", "test_r2"]),
        "",
        "## Top 10 pure-environmental permutation importance",
        "",
        markdown_table(
            importance_table.head(10),
            ["rank", "feature", "importance_mean", "importance_std"],
        ),
        "",
        "## Requested physical drivers",
        "",
        markdown_table(
            importance_table.loc[
                importance_table["feature"].isin(
                    [
                        "lag1_p3_no2_ugm3",
                        "lag1_pw_water_age_hours",
                        "lag1_p3_pm25_ugm3",
                        "lag1_p3_avg_aqi",
                        "lag1_avg_temp_c",
                        "lag1_pw_water_pollution_index",
                    ]
                ),
                ["rank", "feature", "importance_mean", "importance_std"],
            ],
            ["rank", "feature", "importance_mean", "importance_std"],
        ),
        "",
        "Importance identifies held-out predictive contribution under this ablation; it is not causal attribution.",
        "",
        "## Feature exclusions",
        "",
        markdown_table(exclusions, ["column", "dtype", "justification"]),
        "",
    ]
    (EXP3_RESULTS_DIR / "benchmark_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    write_triple_benchmark_synthesis(metrics, importance_table, manifest)
    print(f"Experiment 3 results written to {EXP3_DIR.relative_to(ROOT)}")
    print(f"Source SHA-256 unchanged: {source_hash}")


def print_artifact_checksums(paths):
    for base in paths:
        if base.is_file():
            print(f"SHA-256 {base.relative_to(ROOT)} {file_sha256(base)}")
        elif base.exists():
            for path in sorted(candidate for candidate in base.rglob("*") if candidate.is_file()):
                print(f"SHA-256 {path.relative_to(ROOT)} {file_sha256(path)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=("1", "2", "3", "all"),
        default="1",
        help="Run Experiment 1, 2, 3, or all sequentially (default: 1).",
    )
    args = parser.parse_args()
    if args.experiment in ("1", "all"):
        run_exp1()
    if args.experiment in ("2", "all"):
        run_exp2()
    if args.experiment in ("3", "all"):
        run_exp3()
    print_artifact_checksums(
        [
            EXP1_DIR / "original_artifacts",
            EXP1_RUN_DIR,
            EXP2_DIR / "results",
            EXP2_RUN_DIR,
            EXP3_DATA_DIR,
            EXP3_MODEL_DIR,
            EXP3_RESULTS_DIR,
            EXP3_DIR / "feature_exclusions.csv",
            ROOT / "experiments" / "COMPARATIVE_DECISION_REPORT.md",
            ROOT / "experiments" / "TRIPLE_BENCHMARK_SYNTHESIS_REPORT.md",
        ]
    )


if __name__ == "__main__":
    main()
