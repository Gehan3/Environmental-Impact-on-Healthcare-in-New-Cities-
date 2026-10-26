"""Descriptive urban, environmental, and health intelligence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.config import (
    DISTRICT_COLUMN,
    KPI_REGISTRY,
    MONTH_COLUMN,
    TIME_SERIES_ENVIRONMENT_FIELDS,
    TIME_SERIES_HEALTH_FIELDS,
    VULNERABILITY_GROUPS,
)
from dashboard.data_loader import with_district_aliases


def _format_kpi(value: object, aggregation: str) -> str:
    if pd.isna(value):
        return "N/A"
    if aggregation == "mean":
        return f"{float(value):,.2f}"
    return f"{float(value):,.0f}"


def _latest_value(frame: pd.DataFrame, field: str, aggregation: str) -> object:
    if field not in frame:
        return np.nan
    values = pd.to_numeric(frame[field], errors="coerce").dropna()
    if values.empty:
        return np.nan
    return values.mean() if aggregation == "mean" else values.sum()


def _render_kpis(panel: pd.DataFrame, district: str | None) -> None:
    latest_month = panel[MONTH_COLUMN].max()
    latest = panel.loc[panel[MONTH_COLUMN].eq(latest_month)]
    if district is not None:
        latest = latest.loc[latest[DISTRICT_COLUMN].eq(district)]
    st.caption(
        f"Latest available month: {latest_month:%B %Y} · "
        + ("selected district" if district else "city-wide district aggregation")
    )
    configured = [
        (label, field, aggregation)
        for label, (field, aggregation) in KPI_REGISTRY.items()
    ]
    columns = st.columns(len(configured))
    for column, (label, field, aggregation) in zip(columns, configured):
        value = _latest_value(latest, field, aggregation)
        column.metric(label, _format_kpi(value, aggregation))


def _render_time_series(panel: pd.DataFrame, district: str) -> None:
    st.subheader("Multi-pillar time-series explorer")
    district_panel = panel.loc[panel[DISTRICT_COLUMN].eq(district)].copy()
    available_environment = [
        field for field in TIME_SERIES_ENVIRONMENT_FIELDS if field in district_panel
    ]
    available_health = [field for field in TIME_SERIES_HEALTH_FIELDS if field in district_panel]
    left, right = st.columns(2)
    environment_fields = left.multiselect(
        "Environmental / infrastructure measures",
        available_environment,
        default=available_environment[:3],
        key="urban_env_series",
    )
    health_fields = right.multiselect(
        "Health visit measures",
        available_health,
        default=available_health[:2],
        key="urban_health_series",
    )
    selected = environment_fields + health_fields
    if not selected:
        st.info("Choose at least one available numeric measure to plot.")
        return

    chart = go.Figure()
    for field in environment_fields:
        chart.add_trace(
            go.Scatter(
                x=district_panel[MONTH_COLUMN],
                y=pd.to_numeric(district_panel[field], errors="coerce"),
                name=field,
                mode="lines+markers",
                yaxis="y",
            )
        )
    for field in health_fields:
        chart.add_trace(
            go.Scatter(
                x=district_panel[MONTH_COLUMN],
                y=pd.to_numeric(district_panel[field], errors="coerce"),
                name=field,
                mode="lines+markers",
                yaxis="y2",
            )
        )
    chart.update_layout(
        height=470,
        hovermode="x unified",
        xaxis_title="Month",
        yaxis_title="Environmental / infrastructure units",
        yaxis2={
            "title": "Clinical visits",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
        legend={"orientation": "h", "y": -0.25},
        margin={"t": 25, "b": 90},
    )
    st.plotly_chart(chart, width="stretch")


def _render_vulnerability(panel: pd.DataFrame) -> None:
    st.subheader("Spatial vulnerability matrix")
    selected_groups = st.multiselect(
        "Indicator groups",
        list(VULNERABILITY_GROUPS),
        default=list(VULNERABILITY_GROUPS),
        key="vulnerability_groups",
    )
    fields = [
        field
        for group in selected_groups
        for field in VULNERABILITY_GROUPS[group]
        if field in panel
    ]
    if not fields:
        st.info("No fields from the selected groups are present in this panel version.")
        return

    district_data = (
        panel.groupby(DISTRICT_COLUMN, as_index=False)[fields]
        .mean(numeric_only=True)
        .sort_values(DISTRICT_COLUMN)
    )
    aliased = with_district_aliases(district_data)
    normalized = district_data[fields].copy()
    for field in fields:
        values = pd.to_numeric(normalized[field], errors="coerce")
        minimum, maximum = values.min(), values.max()
        normalized[field] = (
            (values - minimum) / (maximum - minimum)
            if pd.notna(minimum) and pd.notna(maximum) and maximum != minimum
            else 0.0
        )
    figure = px.imshow(
        normalized,
        x=fields,
        y=aliased["district_alias"],
        color_continuous_scale="YlOrRd",
        zmin=0,
        zmax=1,
        aspect="auto",
        labels={"x": "Indicator", "y": "District", "color": "Relative level"},
    )
    figure.update_layout(height=max(450, len(district_data) * 28), margin={"t": 20})
    st.caption("Heatmap values are min–max normalized within each indicator across districts.")
    st.plotly_chart(figure, width="stretch")
    with st.expander("District indicator averages"):
        display = aliased[[DISTRICT_COLUMN, "district_alias"]].copy()
        display = pd.concat([display, district_data[fields]], axis=1)
        st.dataframe(display, width="stretch", hide_index=True)


def _render_field_inspector(panel: pd.DataFrame) -> None:
    st.subheader("Extensible field inspector")
    numeric = panel.select_dtypes(include="number")
    if numeric.empty:
        st.info("No numeric columns are available.")
        return
    selected = st.multiselect(
        "Numeric fields (automatically discovered)",
        numeric.columns.tolist(),
        default=numeric.columns[: min(12, len(numeric.columns))].tolist(),
        key="numeric_inspector_fields",
    )
    if not selected:
        return
    left, right = st.columns(2)
    with left:
        if len(selected) >= 2:
            correlation = numeric[selected].corr()
            st.plotly_chart(
                px.imshow(
                    correlation,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    aspect="auto",
                    title="Selected-field correlation",
                ),
                width="stretch",
            )
        else:
            st.info("Select at least two fields to view correlations.")
    with right:
        distribution_field = st.selectbox(
            "Distribution field", selected, key="inspector_distribution"
        )
        st.plotly_chart(
            px.histogram(
                panel,
                x=distribution_field,
                nbins=30,
                title=f"Distribution: {distribution_field}",
            ),
            width="stretch",
        )


def render(panel: pd.DataFrame, selected_district: str | None = None) -> None:
    st.title("Layer 1 · Urban & Environmental Health Intelligence")
    st.caption(
        "Descriptive observability from the municipal panel; charts summarize observed or "
        "simulated source fields without making causal claims."
    )
    district_values = sorted(panel[DISTRICT_COLUMN].astype(str).unique())
    district_options = ["City-wide", *district_values]
    default_index = (
        district_values.index(selected_district) + 1
        if selected_district in district_values
        else 0
    )
    district_choice = st.selectbox(
        "KPI aggregation",
        district_options,
        index=default_index,
        format_func=lambda value: (
            "City-wide" if value == "City-wide" else with_district_aliases(
                panel[[DISTRICT_COLUMN]].drop_duplicates()
            ).set_index(DISTRICT_COLUMN).loc[value, "district_alias"] + f" ({value})"
        ),
        key="kpi_district",
    )
    _render_kpis(panel, None if district_choice == "City-wide" else district_choice)

    chosen_district = st.selectbox(
        "District for time-series exploration",
        district_values,
        index=district_values.index(selected_district)
        if selected_district in district_values
        else 0,
        format_func=lambda value: with_district_aliases(
            panel[[DISTRICT_COLUMN]].drop_duplicates()
        ).set_index(DISTRICT_COLUMN).loc[value, "district_alias"]
        + f" ({value})",
        key="timeseries_district",
    )
    _render_time_series(panel, chosen_district)
    _render_vulnerability(panel)
    _render_field_inspector(panel)
