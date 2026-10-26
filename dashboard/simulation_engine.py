"""Deterministic, transparent district capacity and policy scenario calculations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from dashboard.config import (
    FINANCIAL_ASSUMPTIONS,
    MONTH_COLUMN,
    SCENARIO_PACKAGES,
)


def derive_baseline_pressure(
    panel: pd.DataFrame,
    calibration_mode: str = "Peak-surge / stress-calibrated capacity",
) -> pd.DataFrame:
    """Build a district baseline from the latest available six-month monthly means."""
    required = {
        "district_key",
        MONTH_COLUMN,
        "pc_visits_demand",
        "pc_effective_capacity",
        "pw_waterborne_visits_total",
        "p3_respiratory_visits_total",
        "p3_heat_related_visits",
    }
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Simulation panel is missing required fields: {sorted(missing)}")

    months = sorted(panel[MONTH_COLUMN].drop_duplicates())
    window = months[-int(FINANCIAL_ASSUMPTIONS["baseline_window_months"]) :]
    recent = panel.loc[panel[MONTH_COLUMN].isin(window)].copy()
    if recent.empty:
        raise ValueError("No rows available for the simulation baseline window.")

    recent["environmental_demand"] = recent[
        [
            "pw_waterborne_visits_total",
            "p3_respiratory_visits_total",
            "p3_heat_related_visits",
        ]
    ].sum(axis=1)
    baseline = (
        recent.groupby("district_key", as_index=False)
        .agg(
            demand=("pc_visits_demand", "mean"),
            nominal_capacity=("pc_effective_capacity", "mean"),
            environmental_demand=("environmental_demand", "mean"),
        )
    )
    if (baseline["nominal_capacity"] <= 0).any():
        raise ValueError("Effective capacity must be positive in every district.")
    baseline["environmental_share"] = (
        baseline["environmental_demand"] / baseline["demand"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)
    baseline["raw_pressure"] = baseline["demand"] / baseline["nominal_capacity"]

    if calibration_mode == "Raw nominal capacity":
        baseline["baseline_pressure"] = baseline["raw_pressure"]
        baseline["effective_capacity"] = baseline["nominal_capacity"]
        baseline["calibration_factor"] = 1.0
    elif calibration_mode == "Peak-surge / stress-calibrated capacity":
        target_min = float(FINANCIAL_ASSUMPTIONS["calibrated_pressure_min"])
        target_max = float(FINANCIAL_ASSUMPTIONS["calibrated_pressure_max"])
        low = float(baseline["raw_pressure"].min())
        high = float(baseline["raw_pressure"].max())
        if np.isclose(low, high):
            calibrated = pd.Series(
                (target_min + target_max) / 2,
                index=baseline.index,
                dtype=float,
            )
        else:
            calibrated = target_min + (
                (baseline["raw_pressure"] - low) / (high - low)
            ) * (target_max - target_min)
        baseline["baseline_pressure"] = calibrated
        baseline["effective_capacity"] = (
            baseline["demand"] / baseline["baseline_pressure"].replace(0, np.nan)
        )
        baseline["calibration_factor"] = (
            baseline["effective_capacity"] / baseline["nominal_capacity"]
        )
    else:
        raise ValueError(f"Unknown capacity calibration mode: {calibration_mode!r}")

    return baseline.sort_values("district_key").reset_index(drop=True)


def _scenario_cost(package: str, settings: Mapping[str, float]) -> float:
    assumptions = FINANCIAL_ASSUMPTIONS
    costs = assumptions["scenario_costs_m_egp"]
    default_capacity = float(assumptions["default_capacity_expansion_pct"])
    selected_capacity = float(settings["capacity_expansion_pct"])
    default_reach = float(assumptions["default_prevention_reach_pct"])
    selected_reach = float(settings["prevention_reach_pct"])
    access_default_minutes = abs(float(assumptions["default_access_improvement_minutes"]))
    selected_access_minutes = abs(float(settings["access_improvement_minutes"]))

    capacity_scale = selected_capacity / default_capacity if default_capacity else 0.0
    prevention_scale = selected_reach / default_reach if default_reach else 0.0
    access_scale = selected_access_minutes / access_default_minutes if access_default_minutes else 0.0

    if package == "center_prevention":
        return float(
            costs["center_fixed"]
            + costs["capacity_expansion_at_default"] * capacity_scale
            + costs["prevention_at_default"] * prevention_scale
        )
    if package == "capacity":
        return float(costs["capacity_expansion_at_default"] * capacity_scale)
    if package == "access":
        return float(costs["access_program_at_default"] * access_scale)
    if package == "prevention":
        return float(costs["prevention_at_default"] * prevention_scale)
    raise ValueError(f"Unknown scenario package: {package!r}")


def simulate_policy_package(
    baseline: pd.DataFrame,
    settings: Mapping[str, float],
    package: str,
    targeted_districts: Sequence[str],
) -> pd.DataFrame:
    """Apply configurable demand/capacity levers to a copy of baseline district metrics."""
    if package not in set(SCENARIO_PACKAGES.values()):
        raise ValueError(f"Unknown scenario package: {package!r}")
    result = baseline.copy()
    targeted = set(str(value) for value in targeted_districts)
    applies = result["district_key"].astype(str).isin(targeted)

    result["scenario_demand"] = result["demand"].astype(float)
    result["scenario_capacity"] = result["effective_capacity"].astype(float)
    population_factor = 1 + float(settings["population_growth_pct"]) / 100
    result.loc[applies, "scenario_demand"] *= population_factor

    if package in {"center_prevention", "prevention"}:
        reach = float(settings["prevention_reach_pct"]) / 100
        efficacy = float(FINANCIAL_ASSUMPTIONS["prevention_efficacy"])
        reduction = (
            result["environmental_demand"] * population_factor * reach * efficacy
        )
        result.loc[applies, "scenario_demand"] -= reduction.loc[applies]

    if package == "center_prevention":
        capacity_uplift = (
            float(FINANCIAL_ASSUMPTIONS["new_center_capacity_pct"])
            + float(settings["capacity_expansion_pct"])
        ) / 100
        result.loc[applies, "scenario_capacity"] *= 1 + capacity_uplift
    elif package == "capacity":
        result.loc[applies, "scenario_capacity"] *= (
            1 + float(settings["capacity_expansion_pct"]) / 100
        )
    elif package == "access":
        minutes = abs(float(settings["access_improvement_minutes"]))
        elasticity = float(FINANCIAL_ASSUMPTIONS["triage_capacity_gain_per_minute"])
        result.loc[applies, "scenario_capacity"] *= 1 + minutes * elasticity

    result["scenario_pressure"] = (
        result["scenario_demand"] / result["scenario_capacity"]
    )
    result["cost_m_egp"] = _scenario_cost(package, settings)
    result["overloaded"] = result["scenario_pressure"] > 1.0
    return result


def compare_scenarios(
    baseline: pd.DataFrame,
    settings: Mapping[str, float],
    targeted_districts: Sequence[str],
) -> pd.DataFrame:
    """Evaluate and rank all four policy packages under the selected budget limit."""
    budget = float(settings["budget_limit_m_egp"])
    rows = []
    for label, package in SCENARIO_PACKAGES.items():
        simulated = simulate_policy_package(
            baseline, settings, package, targeted_districts
        )
        cost = float(simulated["cost_m_egp"].iloc[0])
        rows.append(
            {
                "Scenario": label,
                "5-yr Cost (M EGP)": cost,
                "Budget Status": "Feasible" if cost <= budget else "Exceeds",
                "Overloaded Districts": int(simulated["overloaded"].sum()),
                "Avg. Pressure": float(simulated["scenario_pressure"].mean()),
                "Package": package,
            }
        )
    rows.sort(
        key=lambda row: (
            row["Budget Status"] != "Feasible",
            row["Overloaded Districts"],
            row["Avg. Pressure"],
            row["5-yr Cost (M EGP)"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["Rank"] = rank
    return pd.DataFrame(rows)
