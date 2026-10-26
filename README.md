# Environmental Impact on Public Health: Smart-City Digital Twin for Healthcare Investment

A smart-city digital twin framework designed for the New Administrative Capital (NAC), Egypt. This project integrates multi-pillar environmental datasets (Water Pollution, Solid Waste, Air Quality, Weather) with demographic profiles and healthcare facility capacities to simulate public health impacts, Clinical Pressure Ratios (CPR), and healthcare investment gaps across districts.

---

## 📁 Repository Structure

```text
Environmental-Impact-on-Healthcare-in-New-Cities-/
│
├── Data/
│   ├── Raw data/                 # Raw monthly tabular datasets
│   │   ├── district_population_profile.csv
│   │   ├── environment_event_calendar.csv
│   │   ├── healthcare_facilities.csv
│   │   ├── p2_solid_waste_monthly.csv
│   │   ├── p3_air_quality_health_monthly.csv
│   │   ├── p4_service_requests_monthly.csv
│   │   ├── population_monthly_clean.csv
│   │   ├── pw_water_pollution_monthly.csv
│   │   └── weather_monthly.csv
│   │
│   └── preprocessed data/        # Cleaned and linked output panel
│       ├── master_panel_linked.csv
│       └── event_district_month.csv
│
├── link_all_data.py              # Main data preprocessing, merging & simulation pipeline
├── requirements.txt              # Project dependencies (pandas, numpy)
├── .gitignore                    # Git ignore file (excludes venv/ and cache)
└── README.md                     # Project documentation

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
