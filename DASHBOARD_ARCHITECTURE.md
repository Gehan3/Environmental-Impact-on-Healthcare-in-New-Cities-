# Smart-City Digital Twin Dashboard

## Purpose and boundaries

The Streamlit/Plotly application presents the municipal panel and triple-benchmark results as three decision-support layers. It is a descriptive and scenario-planning tool, not a causal model, an operational clinical system, or a validated capital-cost estimator. The original panel and all experiment artifact directories are read-only inputs to the dashboard.

## Architecture

```text
app.py
└── dashboard/
    ├── config.py                 # Schemas, artifact registry, KPI and policy assumptions
    ├── data_loader.py            # Cached CSV/model loading, schema checks and normalization
    ├── simulation_engine.py      # CPR calibration, policy levers, costs and ranking
    └── views/
        ├── layer1_urban_bi.py    # Descriptive urban/environment/health intelligence
        ├── layer2_ml_lab.py      # Benchmark, feature importance, holdout and exclusions
        └── layer3_what_if.py     # Executive policy/budget simulation
```

The loaders use named CSV headers and tolerate a UTF-8 BOM. New numeric panel columns appear automatically in the Layer 1 field inspector. KPI fields, explorer indicators, experiment locations, and policy assumptions are registered in `dashboard/config.py`; adding a schema mapping there is preferable to hard-coding positional indexes in a view.

### Layer 1 — Descriptive intelligence

- KPI cards aggregate the latest available month city-wide or by district.
- District aliases D01...D15 are assigned from sorted observed keys, not fixed district IDs.
- The time-series view overlays environmental/infrastructure measurements and clinical visits on separate axes.
- The spatial matrix normalizes each chosen indicator across districts for visual comparison; raw district averages remain available in a table.
- The field inspector discovers numeric columns at runtime and provides distribution and correlation views.

The source's `p4_open_requests_end_month` is used for active complaints because `active_complaints` is not a source column. If a future panel provides the preferred name, update `KPI_REGISTRY` to select it.

### Layer 2 — Benchmark diagnostics

Experiment metrics, prediction partitions, feature importance, exclusions, and model paths are resolved through the `EXPERIMENTS` registry. Prediction value columns are discovered by the `actual_*` and `predicted_*` prefixes.

The saved Experiment 1 snapshot has metrics, holdout predictions, and importance, but does not contain a serialized model. Experiments 2 and 3 have four serialized models each; the dashboard loads and reports only files that exist. The displayed experiments and reports do not refit models.

### Layer 3 — What-if and budget support

The simulation starts from the latest six months of district means for `pc_visits_demand`, `pc_effective_capacity`, and the environmental visit components.

Two capacity modes are available:

1. **Raw nominal capacity:** district baseline pressure is `mean(pc_visits_demand) / mean(pc_effective_capacity)`. This reflects the source's nominal capacity and its low CPR scale.
2. **Peak-surge / stress-calibrated capacity:** preserve district ranking but linearly map the observed raw-pressure minimum and maximum to 0.75 and 1.35, respectively. If the input values are constant, all districts receive the midpoint 1.05. Calibrated capacity is explicitly derived as baseline demand divided by the calibrated CPR. This is a visualization assumption, not measured utilization.

Scenario levers are applied only to selected target districts:

- Population growth scales demand by `1 + growth_pct / 100`.
- Prevention reduces demand by environmental component visits × program reach × configured efficacy.
- Capacity expansion scales effective capacity by the selected percent; the new-center package adds the configured new-center capacity uplift.
- Access/triage improvement scales capacity by `1 + abs(minutes) × capacity_gain_per_minute`.
- Capacity pressure is scenario demand / scenario capacity; overloaded means CPR > 1.

Five-year package costs use configurable baseline anchors in `FINANCIAL_ASSUMPTIONS["scenario_costs_m_egp"]`. At the default controls, A costs 40.5 M EGP (24.0 center + 9.0 expansion + 7.5 prevention), B costs 9.0 M, C costs 16.0 M, and D costs 7.5 M. Variable components scale linearly with their respective slider against the default slider value. Ranking sorts feasible packages ahead of infeasible ones, then fewer overloaded districts, lower average pressure, and lower cost. These are synthetic planning assumptions, not vendor quotes or empirically estimated intervention effects.

## Configuration and data updates

When panel columns change:

1. Keep `district_key` and `month` as the unique structural keys.
2. Add or update display/KPI field names in `KPI_REGISTRY`, `TIME_SERIES_ENVIRONMENT_FIELDS`, `TIME_SERIES_HEALTH_FIELDS`, or `VULNERABILITY_GROUPS`.
3. Keep model artifact paths and target names in `EXPERIMENTS` synchronized with the corresponding experiment output.
4. If new serialized models are added, register their paths under the experiment's `models` mapping. Loading is path-driven; unavailable files are reported rather than silently substituted.

When real financial/operations data become available, replace the assumption values and cost anchors in `FINANCIAL_ASSUMPTIONS` with validated sources and add provenance, units, effective dates, and uncertainty ranges. Review the scenario formulas and tests before using them for budget allocation. The current configuration intentionally keeps all synthetic assumptions in one module.

## Run and verify

The pinned runtime dependencies support Python 3.10–3.12. Existing model files were serialized with scikit-learn 1.9.1; the cloud-compatible runtime pins 1.7.2, and the smoke suite checks each saved model can produce holdout predictions. Scikit-learn warns about cross-version persistence; regenerate and version model artifacts together with the runtime before treating them as long-lived production models.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Run the headless smoke and deterministic simulation checks with:

```bash
python -m unittest discover -s tests -v
```

The checks load panel and experiment artifacts, verify saved model availability and schema, exercise deterministic policy calculations, compile/import dashboard modules, and assert that source and protected artifact SHA-256 hashes do not change.

## Epistemic limitations

Experiments 2 and 3 use simulator-generated encounter outcomes and a registered-resident denominator that needs validation. Saved feature importance is predictive association. The calibrated capacity mode, prevention efficacy, center/triage effects, and all package costs are illustrative assumptions. Do not interpret the dashboard as demonstrating causal environmental effects or deploy its scenario ranks without local validation.
