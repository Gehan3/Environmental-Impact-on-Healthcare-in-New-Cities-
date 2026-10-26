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

