# Environmental Impact on Public Health: Smart-City Digital Twin for Healthcare Investment

A smart-city digital twin framework designed for the New Administrative Capital (NAC), Egypt. This project integrates multi-pillar environmental datasets (Water Pollution, Solid Waste, Air Quality, Weather) with demographic profiles and healthcare facility capacities to simulate public health impacts, Clinical Pressure Ratios (CPR), and healthcare investment gaps across districts.

---

## 📁 Repository Structure

```text
Environmental-Impact-on-Healthcare-in-New-Cities-/
│
├── Data/
│   └── Processed/
│       └── master_panel.csv      # Read-only ML/dashboard panel
├── dashboard/                    # Config, loaders, simulation engine, and views
├── experiments/                  # Preserved benchmark data, models, and reports
├── tests/                         # Dashboard schema and simulation checks
├── app.py                         # Streamlit dashboard entry point
├── link_all_data.py               # Data linking and simulation pipeline
├── run_experiments.py              # Reproducible triple-benchmark runner
├── requirements.txt               # ML, Streamlit, and Plotly dependencies
└── README.md                      # Project documentation

⚙️ Getting Started & Installation
1. Clone the Repository
Bash
git clone [https://github.com/Gehan3/Environmental-Impact-on-Healthcare-in-New-Cities-.git](https://github.com/Gehan3/Environmental-Impact-on-Healthcare-in-New-Cities-.git)
cd Environmental-Impact-on-Healthcare-in-New-Cities-
2. Set Up Virtual Environment (Recommended)
Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
3. Install Dependencies
Bash
pip install -r requirements.txt
🚀 Running the Pipeline
To execute the data linking, index calculation, and generate the preprocessed master panel:

Bash
python link_all_data.py
Outputs will be automatically saved inside Data/preprocessed data/.

📊 Outputs Generated
master_panel_linked.csv: Unified district-month panel containing all environmental indicators, population metrics, health demands, and stress indices.

event_district_month.csv: Expanded calendar mapping environmental and civic events across affected districts and months.

## Reproducible ML Experiments

All three experiments read `Data/Processed/master_panel.csv` without modifying it. Use Python 3.12 and install the pinned dependencies above:

```bash
python run_experiments.py --experiment 1    # Macro-volume demand forecast
python run_experiments.py --experiment 2    # Environmental-rate benchmark
python run_experiments.py --experiment 3    # Pure-environmental ablation
python run_experiments.py --experiment all  # Run all three sequentially
```

Experiment 1 artifacts are preserved under `experiments/exp1_macro_volume/`; reruns write to `experiments/reruns/exp1_macro_volume/`. Experiment 2's existing files remain under `experiments/exp2_environmental_rate/`; reruns write to `experiments/reruns/exp2_environmental_rate/`. Experiment 3 writes its splits, serialized models, metrics, logs, and feature reports to `experiments/exp3_pure_environmental/`. `--all` runs each pipeline sequentially but writes reruns to separate paths so existing Experiment 1 and 2 artifacts are not overwritten.

All experiments use the same February 2025–March 2026 training and April–September 2026 test windows, seed 42, and expanding-window temporal validation. Runs print artifact SHA-256 hashes for repeatability checks. LightGBM tries an available NVIDIA GPU backend and otherwise falls back to 8-thread CPU.

Experiment 1 forecasts raw `total_visits_demand`. Experiments 2 and 3 predict `1,000 * (pw_waterborne_visits_total + p3_respiratory_visits_total + p3_heat_related_visits) / residents_registered`, excluding `nonresp_visits`. Experiment 3 restricts predictors to lagged environmental, infrastructure, weather, and event features, without spatial/demographic shortcuts. The registered-resident denominator requires validation: several districts show large differences from `p3_population_exposed`. Outcomes are simulator-generated; metrics and importance are predictive associations, not causal estimates. See the [Phase 1 audit](experiments/DIAGNOSTIC_REPORT_AND_PLAN.md), [Experiment 2 comparison](experiments/COMPARATIVE_DECISION_REPORT.md), and [triple-benchmark synthesis](experiments/TRIPLE_BENCHMARK_SYNTHESIS_REPORT.md).

For complete target lineage, exact feature lists/exclusion rationale, temporal CV and tuning settings, findings, limitations, and the artifact map, see [Experiment Methodology and Findings](experiments/EXPERIMENT_METHODOLOGY_AND_FINDINGS.md).

## Smart-City Decision Dashboard

The modular Streamlit dashboard provides three views: descriptive urban/environmental intelligence, transparent ML benchmark diagnostics, and a configurable budget-constrained what-if simulator. Its data, experiment, and policy configuration is centralized in `dashboard/config.py`. Source data and benchmark artifacts are loaded read-only.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Run the headless dashboard schema, model-loading, simulation determinism, and artifact-integrity checks with:

```bash
python -m unittest discover -s tests -v
```

See [DASHBOARD_ARCHITECTURE.md](DASHBOARD_ARCHITECTURE.md) for the layer design, CPR calibration formula, scenario cost assumptions, extension points, and interpretation limits. Financial and intervention values in the simulation are illustrative assumptions—not local cost estimates or causal effect sizes.

The pinned dashboard/runtime dependencies target Python 3.10–3.12. The saved scikit-learn estimators were originally serialized with scikit-learn 1.9.1; the cloud-compatible pin is 1.7.2. Headless validation loads each saved model and verifies 90-row holdout inference under the pinned runtime. Re-serialize artifacts with the deployed runtime during a future model release to remove scikit-learn cross-version persistence warnings.
