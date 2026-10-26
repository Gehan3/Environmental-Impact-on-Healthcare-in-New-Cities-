"""Cached, schema-aware loading and validation for dashboard inputs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from dashboard.config import (
    DISTRICT_ALIAS_PREFIX,
    DISTRICT_ALIAS_WIDTH,
    DISTRICT_COLUMN,
    EXPERIMENTS,
    MONTH_COLUMN,
    PANEL_PATH,
)


@st.cache_data(show_spinner=False)
def load_panel(path: str | Path = PANEL_PATH) -> pd.DataFrame:
    """Load the panel, normalize its key fields, and validate its structural schema."""
    panel = pd.read_csv(path, encoding="utf-8-sig")
    missing = {DISTRICT_COLUMN, MONTH_COLUMN}.difference(panel.columns)
    if missing:
        raise ValueError(f"Panel is missing required key columns: {sorted(missing)}")
    panel[MONTH_COLUMN] = pd.to_datetime(panel[MONTH_COLUMN], errors="raise")
    panel[DISTRICT_COLUMN] = panel[DISTRICT_COLUMN].astype(str).str.strip()
    if panel.duplicated([DISTRICT_COLUMN, MONTH_COLUMN]).any():
        raise ValueError("Panel contains duplicate district-month keys.")
    return panel.sort_values([MONTH_COLUMN, DISTRICT_COLUMN]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_csv_artifact(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_resource(show_spinner=False)
def load_model_artifacts() -> dict[str, dict[str, object]]:
    """Load only model files present in the configured experiment artifact registry."""
    loaded: dict[str, dict[str, object]] = {}
    for experiment_name, config in EXPERIMENTS.items():
        loaded[experiment_name] = {}
        for model_name, path in config["models"].items():
            model_path = Path(path)
            if model_path.is_file():
                loaded[experiment_name][model_name] = joblib.load(model_path)
    return loaded


def validate_panel_schema(panel: pd.DataFrame) -> list[str]:
    """Return non-fatal schema observations; unknown numeric columns remain supported."""
    observations = []
    if len(panel) != 315:
        observations.append(f"Expected 315 rows in the benchmark panel; found {len(panel)}.")
    district_count = panel[DISTRICT_COLUMN].nunique()
    if district_count != 15:
        observations.append(f"Expected 15 districts; found {district_count}.")
    if panel[MONTH_COLUMN].nunique() != 21:
        observations.append(
            f"Expected 21 monthly timestamps; found {panel[MONTH_COLUMN].nunique()}."
        )
    return observations


def district_alias(district_key: object, ordered_districts: list[str]) -> str:
    """Map observed district identifiers to D01...Dnn without assuming identifier values."""
    normalized = str(district_key)
    try:
        ordinal = ordered_districts.index(normalized) + 1
    except ValueError:
        return normalized
    return f"{DISTRICT_ALIAS_PREFIX}{ordinal:0{DISTRICT_ALIAS_WIDTH}d}"


def with_district_aliases(panel: pd.DataFrame) -> pd.DataFrame:
    ordered = sorted(panel[DISTRICT_COLUMN].astype(str).unique())
    result = panel.copy()
    result["district_alias"] = result[DISTRICT_COLUMN].map(
        lambda value: district_alias(value, ordered)
    )
    return result


def normalize_prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Find actual/prediction fields by their names, not fixed CSV positions."""
    if frame.empty:
        return frame.copy()
    actual_candidates = [column for column in frame if column.startswith("actual_")]
    predicted_candidates = [column for column in frame if column.startswith("predicted_")]
    if not actual_candidates or not predicted_candidates:
        raise ValueError("Prediction artifact must contain actual_* and predicted_* columns.")
    actual_column = actual_candidates[0]
    predicted_column = predicted_candidates[0]
    normalized = frame.copy()
    normalized["actual_value"] = pd.to_numeric(normalized[actual_column], errors="raise")
    normalized["predicted_value"] = pd.to_numeric(
        normalized[predicted_column], errors="raise"
    )
    normalized["month"] = pd.to_datetime(normalized["month"], errors="raise")
    normalized["district_key"] = normalized["district_key"].astype(str)
    return normalized


def load_experiment_artifacts() -> dict[str, dict[str, pd.DataFrame]]:
    return {
        name: {
            "metrics": load_csv_artifact(config["metrics"]),
            "predictions": normalize_prediction_columns(
                load_csv_artifact(config["predictions"])
            ),
            "importance": load_csv_artifact(config["importance"]),
            "exclusions": load_csv_artifact(config["exclusions"]),
        }
        for name, config in EXPERIMENTS.items()
    }


def model_artifact_summary(
    models: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame:
    rows = []
    for experiment_name, config in EXPERIMENTS.items():
        loaded = models.get(experiment_name, {})
        configured = config["models"]
        if not configured:
            rows.append(
                {
                    "Experiment": experiment_name,
                    "Model": "—",
                    "Artifact": "—",
                    "Input feature count": None,
                    "Status": "No serialized model in source artifacts",
                }
            )
            continue
        for model_name, path in configured.items():
            model = loaded.get(model_name)
            input_features = (
                getattr(model, "feature_names_in_", None) if model is not None else None
            )
            feature_count = (
                len(input_features)
                if input_features is not None
                else getattr(model, "n_features_in_", None)
                if model is not None
                else None
            )
            rows.append(
                {
                    "Experiment": experiment_name,
                    "Model": model_name,
                    "Artifact": Path(path).name,
                    "Input feature count": feature_count,
                    "Status": "Loaded" if model is not None else "Missing",
                }
            )
    return pd.DataFrame(rows)
