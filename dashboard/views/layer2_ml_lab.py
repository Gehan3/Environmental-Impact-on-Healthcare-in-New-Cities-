"""Triple-benchmark diagnostics, holdout audit, and leakage inspection."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.config import EXPERIMENTS
from dashboard.data_loader import district_alias, model_artifact_summary


WINNER_BY_KEY = {
    "exp1": "Ridge (tuned)",
    "exp2": "LightGBM (tuned)",
    "exp3": "LightGBM (tuned)",
}


def _metric_matrix(artifacts: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    frames = []
    for experiment_name, config in EXPERIMENTS.items():
        metrics = artifacts[experiment_name]["metrics"]
        if metrics.empty:
            continue
        frame = metrics.copy()
        frame["Experiment"] = config["label"]
        frame["Target"] = config["target"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _render_metric_cards(matrix: pd.DataFrame) -> None:
    if matrix.empty:
        st.warning("Benchmark metrics are not available.")
        return
    experiment_name = st.selectbox("Experiment", list(EXPERIMENTS), key="metric_experiment")
    config = EXPERIMENTS[experiment_name]
    experiment_rows = matrix.loc[matrix["Experiment"].eq(config["label"])]
    models = experiment_rows["model"].astype(str).tolist()
    preferred = WINNER_BY_KEY[config["key"]]
    selected_model = st.selectbox(
        "Model",
        models,
        index=models.index(preferred) if preferred in models else 0,
        key="metric_model",
    )
    row = experiment_rows.loc[experiment_rows["model"].eq(selected_model)].iloc[0]
    columns = st.columns(3)
    for column, label, metric in zip(
        columns,
        ["Test R²", "Test RMSE", "Test MAE"],
        ["test_r2", "test_rmse", "test_mae"],
    ):
        column.metric(label, f"{float(row[metric]):,.4f}")
    st.caption(
        f"{config['target']} · units differ between macro-volume and per-1,000 rate targets."
    )


def _render_feature_evolution(
    artifacts: dict[str, dict[str, pd.DataFrame]],
) -> None:
    frames = []
    for experiment_name, config in EXPERIMENTS.items():
        importance = artifacts[experiment_name]["importance"]
        if importance.empty:
            continue
        frame = importance.copy()
        frame["Experiment"] = config["label"]
        frame["importance_mean"] = pd.to_numeric(
            frame["importance_mean"], errors="coerce"
        )
        frames.append(frame)
    if not frames:
        st.warning("Feature-importance artifacts are not available.")
        return
    combined = pd.concat(frames, ignore_index=True)
    order = [config["label"] for config in EXPERIMENTS.values()]
    figure = px.bar(
        combined,
        x="importance_mean",
        y="feature",
        color="Experiment",
        facet_col="Experiment",
        category_orders={"Experiment": order},
        orientation="h",
        height=760,
        title="Held-out permutation-importance evolution (top 10 per experiment)",
        labels={
            "importance_mean": "Increase in test RMSE when permuted",
            "feature": "Feature",
        },
    )
    figure.update_yaxes(matches=None, autorange="reversed")
    figure.update_layout(showlegend=False, margin={"l": 190, "r": 20, "t": 90})
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Importance is held-out predictive association, not causal effect. The ranking "
        "is conditional on each experiment's feature set and saved model."
    )


def _render_holdout(artifacts: dict[str, dict[str, pd.DataFrame]]) -> None:
    st.subheader("Holdout forecast audit · April–September 2026")
    experiment_name = st.selectbox(
        "Prediction experiment", list(EXPERIMENTS), key="holdout_experiment"
    )
    predictions = artifacts[experiment_name]["predictions"]
    if predictions.empty:
        st.warning("Prediction artifact is not available.")
        return
    models = sorted(predictions["model"].unique())
    model = st.selectbox("Model family / candidate", models, key="holdout_model")
    districts = sorted(predictions["district_key"].unique())
    district = st.selectbox(
        "District",
        districts,
        format_func=lambda value: f"{district_alias(value, districts)} ({value})",
        key="holdout_district",
    )
    selected = predictions.loc[
        predictions["model"].eq(model) & predictions["district_key"].eq(district)
    ].sort_values("month")
    if selected.empty:
        st.info("No prediction rows match the selected filters.")
        return
    selected = selected.copy()
    selected["District"] = selected["district_key"]
    long = selected.melt(
        id_vars=["month", "District"],
        value_vars=["actual_value", "predicted_value"],
        var_name="Series",
        value_name="Value",
    )
    long["Series"] = long["Series"].map(
        {"actual_value": "Actual", "predicted_value": "Predicted"}
    )
    figure = px.line(
        long,
        x="month",
        y="Value",
        color="Series",
        markers=True,
        title=f"{EXPERIMENTS[experiment_name]['label']} · {model} · {district}",
        labels={"month": "Month", "Value": "Target value"},
    )
    st.plotly_chart(figure, width="stretch")
    st.dataframe(selected, width="stretch", hide_index=True)


def _render_exclusions(artifacts: dict[str, dict[str, pd.DataFrame]]) -> None:
    st.subheader("Feature exclusion & leakage audit")
    experiment_name = st.selectbox(
        "Exclusion log", list(EXPERIMENTS), key="exclusion_experiment"
    )
    exclusions = artifacts[experiment_name]["exclusions"]
    if exclusions.empty:
        st.warning("Feature-exclusion log is not available.")
        return
    search = st.text_input("Search columns, dtypes, or justifications")
    filtered = exclusions.copy()
    if search:
        matches = filtered.astype(str).apply(
            lambda column: column.str.contains(search, case=False, regex=False)
        )
        filtered = filtered.loc[matches.any(axis=1)]
    st.dataframe(filtered, width="stretch", hide_index=True)


def render(
    artifacts: dict[str, dict[str, pd.DataFrame]],
    model_artifacts: dict[str, dict[str, object]],
) -> None:
    st.title("Layer 2 · Triple-Benchmark ML & Ablation Lab")
    st.caption(
        "Experiment 1 forecasts simulated visit volume; Experiments 2–3 model the "
        "requested environmental-encounter rate with progressively restricted predictors."
    )
    matrix = _metric_matrix(artifacts)
    st.subheader("Unified benchmark matrix")
    if matrix.empty:
        st.warning("No benchmark metrics were loaded.")
    else:
        preferred_columns = [
            "Experiment",
            "model",
            "Target",
            "cv_rmse_mean",
            "test_rmse",
            "test_mae",
            "test_r2",
        ]
        columns = [column for column in preferred_columns if column in matrix]
        st.dataframe(
            matrix[columns].sort_values(["Experiment", "test_rmse"]),
            width="stretch",
            hide_index=True,
        )
        _render_metric_cards(matrix)

    with st.expander("Serialized model artifact status"):
        st.dataframe(
            model_artifact_summary(model_artifacts),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Saved joblib artifacts are loaded for Experiments 2 and 3. The preserved "
            "Experiment 1 snapshot contains metrics and predictions, but no serialized model."
        )

    _render_feature_evolution(artifacts)
    _render_holdout(artifacts)
    _render_exclusions(artifacts)
