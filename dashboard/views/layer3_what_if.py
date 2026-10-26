"""Executive what-if simulator and budget-constrained policy comparison."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.config import (
    FINANCIAL_ASSUMPTIONS,
    SCENARIO_PACKAGES,
)
from dashboard.data_loader import with_district_aliases
from dashboard.simulation_engine import (
    compare_scenarios,
    derive_baseline_pressure,
    simulate_policy_package,
)


CALIBRATION_MODES = [
    "Raw nominal capacity",
    "Peak-surge / stress-calibrated capacity",
]


def _scenario_controls(districts: list[str]) -> tuple[dict[str, float], str, list[str]]:
    assumptions = FINANCIAL_ASSUMPTIONS
    with st.sidebar.form("what_if_controls"):
        st.subheader("Scenario controls")
        growth = st.slider(
            "Population growth in target districts (%)",
            int(assumptions["population_growth_min_pct"]),
            int(assumptions["population_growth_max_pct"]),
            int(assumptions["default_population_growth_pct"]),
        )
        capacity = st.slider(
            "Center capacity expansion (%)",
            int(assumptions["capacity_expansion_min_pct"]),
            int(assumptions["capacity_expansion_max_pct"]),
            int(assumptions["default_capacity_expansion_pct"]),
        )
        reach = st.slider(
            "Preventive program reach (% of environmental demand)",
            0,
            int(assumptions["prevention_reach_max_pct"]),
            int(assumptions["default_prevention_reach_pct"]),
        )
        access = st.slider(
            "Access-time / triage improvement (minutes)",
            int(assumptions["access_improvement_min_minutes"]),
            int(assumptions["access_improvement_max_minutes"]),
            int(assumptions["default_access_improvement_minutes"]),
        )
        budget = st.slider(
            "5-year budget limit (M EGP)",
            float(assumptions["budget_min_m_egp"]),
            float(assumptions["budget_max_m_egp"]),
            float(assumptions["default_budget_m_egp"]),
            step=1.0,
        )
        package_label = st.selectbox("Active package", list(SCENARIO_PACKAGES))
        target_districts = st.multiselect(
            "Target districts",
            districts,
            default=districts[:1],
            format_func=lambda value: with_district_aliases(
                pd.DataFrame({"district_key": districts})
            ).set_index("district_key").loc[value, "district_alias"]
            + f" ({value})",
        )
        st.form_submit_button("Run simulation", type="primary")
    settings = {
        "population_growth_pct": float(growth),
        "capacity_expansion_pct": float(capacity),
        "prevention_reach_pct": float(reach),
        "access_improvement_minutes": float(access),
        "budget_limit_m_egp": float(budget),
    }
    return settings, SCENARIO_PACKAGES[package_label], target_districts


def _pressure_chart(
    baseline: pd.DataFrame,
    scenario: pd.DataFrame,
    calibration_mode: str,
) -> None:
    baseline_plot = baseline[["district_key", "baseline_pressure"]].rename(
        columns={"baseline_pressure": "Pressure"}
    )
    baseline_plot["Series"] = "Baseline"
    scenario_plot = scenario[["district_key", "scenario_pressure"]].rename(
        columns={"scenario_pressure": "Pressure"}
    )
    scenario_plot["Series"] = "Active scenario"
    chart_data = pd.concat([baseline_plot, scenario_plot], ignore_index=True)
    aliases = with_district_aliases(baseline[["district_key"]].drop_duplicates())
    chart_data = chart_data.merge(aliases, on="district_key", how="left")
    figure = px.bar(
        chart_data,
        x="district_alias",
        y="Pressure",
        color="Series",
        barmode="group",
        color_discrete_map={"Baseline": "#718096", "Active scenario": "#159a83"},
        labels={"district_alias": "District", "Pressure": "Capacity pressure ratio"},
        title=f"Capacity pressure by district · {calibration_mode}",
    )
    figure.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="red",
        annotation_text="Capacity limit = 1.0",
    )
    figure.update_layout(height=490, legend={"orientation": "h", "y": -0.2})
    st.plotly_chart(figure, width="stretch")


def _render_calibration_note(
    baseline: pd.DataFrame,
    calibration_mode: str,
) -> None:
    if calibration_mode == "Raw nominal capacity":
        st.info(
            "Raw mode uses source `pc_visits_demand / pc_effective_capacity`; "
            f"observed range is {baseline['baseline_pressure'].min():.3f}–"
            f"{baseline['baseline_pressure'].max():.3f}."
        )
    else:
        st.warning(
            "Stress-calibrated mode is a transparent scenario visualization, not observed "
            "capacity utilization. It preserves the district ordering of raw pressure and "
            "linearly maps its six-month min/max to the configured 0.75–1.35 range. "
            "Capacity is recalculated as baseline demand divided by that calibrated ratio."
        )


def render(panel: pd.DataFrame) -> None:
    st.title("Layer 3 · Executive What-If Simulation")
    st.caption(
        "Scenario calculations are deterministic and use configurable policy/financial "
        "assumptions; they are not a forecast or capital-cost estimate."
    )
    calibration_mode = st.radio(
        "Capacity pressure basis",
        CALIBRATION_MODES,
        horizontal=True,
        index=1,
        help="Choose source nominal capacity or a clearly labeled stress-calibrated capacity.",
    )
    try:
        baseline = derive_baseline_pressure(panel, calibration_mode)
    except (ValueError, KeyError) as error:
        st.error(f"Cannot run the policy simulation: {error}")
        return

    districts = baseline["district_key"].astype(str).tolist()
    settings, active_package, targets = _scenario_controls(districts)
    if not targets:
        st.warning("Select at least one target district to run a policy scenario.")
        return
    active = simulate_policy_package(baseline, settings, active_package, targets)
    _render_calibration_note(baseline, calibration_mode)
    _pressure_chart(baseline, active, calibration_mode)

    st.subheader("Budget-constrained package comparison")
    comparison = compare_scenarios(baseline, settings, targets)
    st.dataframe(
        comparison[
            [
                "Rank",
                "Scenario",
                "5-yr Cost (M EGP)",
                "Budget Status",
                "Overloaded Districts",
                "Avg. Pressure",
            ]
        ].style.format(
            {
                "5-yr Cost (M EGP)": "{:.2f}",
                "Avg. Pressure": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    active_cost = float(active["cost_m_egp"].iloc[0])
    if active_cost > settings["budget_limit_m_egp"]:
        st.error(
            f"Active package exceeds the configured budget by "
            f"{active_cost - settings['budget_limit_m_egp']:.2f} M EGP."
        )
    else:
        st.success(f"Active package is within budget at {active_cost:.2f} M EGP.")

    st.subheader("Epistemic disclaimer")
    st.warning(
        "Data-driven inputs: district-month visit volumes, environmental encounter "
        "components, nominal effective capacity, population, and environment-demand "
        "shares are loaded from the source panel. Synthetic assumptions in "
        "`dashboard/config.py`: stress-calibration range, new-center capacity uplift, "
        "prevention efficacy, triage/access elasticity, and all 5-year package costs. "
        "No external financial quotes or causal intervention effects are supplied; "
        "replace these assumptions with validated local inputs before investment decisions."
    )
