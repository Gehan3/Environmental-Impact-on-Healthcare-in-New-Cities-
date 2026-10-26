"""Streamlit entry point for the smart-city digital-twin decision dashboard."""

from __future__ import annotations

import streamlit as st

from dashboard.config import DISTRICT_COLUMN
from dashboard.data_loader import (
    load_experiment_artifacts,
    load_model_artifacts,
    load_panel,
    validate_panel_schema,
)
from dashboard.views import layer1_urban_bi, layer2_ml_lab, layer3_what_if


st.set_page_config(
    page_title="Smart-City Digital Twin",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    st.sidebar.title("Smart-City Digital Twin")
    layer = st.sidebar.radio(
        "Decision-support layer",
        [
            "Layer 1 · Urban intelligence",
            "Layer 2 · ML benchmark lab",
            "Layer 3 · What-if simulation",
        ],
    )
    try:
        panel = load_panel()
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Unable to load the municipal panel: {error}")
        st.stop()

    observations = validate_panel_schema(panel)
    if observations:
        st.sidebar.warning(" · ".join(observations))
    st.sidebar.caption(
        f"{len(panel):,} district-month rows · "
        f"{panel[DISTRICT_COLUMN].nunique()} districts · "
        f"{panel['month'].nunique()} months"
    )

    if layer.startswith("Layer 1"):
        layer1_urban_bi.render(panel)
    elif layer.startswith("Layer 2"):
        try:
            artifacts = load_experiment_artifacts()
            models = load_model_artifacts()
        except (OSError, ValueError, KeyError) as error:
            st.error(f"Unable to load ML benchmark artifacts: {error}")
            st.stop()
        layer2_ml_lab.render(artifacts, models)
    else:
        layer3_what_if.render(panel)


if __name__ == "__main__":
    main()
