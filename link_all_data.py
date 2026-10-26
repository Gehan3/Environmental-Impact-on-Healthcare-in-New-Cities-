"""
link_all_data.py - Links all generated datasets into one district-month panel and
computes the KPI framework (Level 1 pillar KPIs -> Level 2 sub-indices ->
Level 3 outcomes -> Level 4 scores -> Level 5 city KPIs).

All joins use two keys:
    district_key  (1001..1015)   district
    month         (YYYY-MM-01)   month

Usage:
    python link_all_data.py
    (defaults: inputs from <script folder>/Data/Raw data, outputs to <script folder>/Data/Linked data;
     override with --data-dir, --facilities-dir, --out)

Requirements: pandas, numpy only.

Relationship map (what links to what):
    population_monthly_clean.csv       --+
    pw_water_pollution_monthly.csv     --|   (water replaces electricity)
    p2_solid_waste_monthly.csv         --+-- district_key + month --> master_panel_linked.csv
    p3_air_quality_health_monthly.csv  --|
    p4_service_requests_monthly.csv    --+
    healthcare_facilities.csv          -- district_key only (static) --> repeated for every month
    district_population_profile.csv    -- district_key only (static) --> base attractiveness (+ optional columns)
    weather_monthly.csv                -- month only (same weather for all districts) --> added to every district
    environment_event_calendar.csv     -- expanded to (district x month) rows --> event_district_month.csv

Outputs:
    master_panel_linked.csv    district x month: pillar KPIs, sub-indices, ESI, health demand, scores
    city_kpis_monthly.csv      month: Level 5 city Command Center KPIs
    event_district_month.csv   event x district x month
"""
import argparse, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ["district_key", "month"]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "Data" / "Raw data"
DEFAULT_OUT_DIR = SCRIPT_DIR / "Data" / "Linked data"
FACILITIES_FILE = "healthcare_facilities.csv"
# Accepted column names in the facilities file (first match is used)
FACILITY_ALIASES = {
    "fclass": ["fclass", "amenity", "facility_type", "type"],
    "healthcare_centers": ["healthcare_centers", "centers", "n_centers", "center_count"],
    "healthcare_capacity": ["healthcare_capacity", "capacity", "capacity_visits", "monthly_capacity"],
}
PRIMARY_CARE_CLASSES = ["clinic", "doctors"]
HOSPITAL_CLASSES = ["hospital"]
# Optional district-level columns taken from district_population_profile.csv when present
PROFILE_OPTIONAL = ["avg_travel_time_min", "accessible_15min_pct", "income_index", "staff_current"]

# ---- Assumptions (placeholders, replace with calibrated values) ----
HOSP_OPD_SHARE = 0.25              # share of hospital capacity that works as outpatient clinics
PC_SHARE = 0.70                    # primary-care share of all visits
VISITS_PER_PERSON_MONTH = 0.2125   # non-respiratory visits per person per month
VISITS_PER_STAFF_MONTH = 400       # visits one staff member can serve per month (HD8)
NEED_HIGH = 50                     # Need Score at or above this = high-need district (C6)

# ---- Level 2: sub-index specs -> (Level 1 KPI column, good, bad, weight) ----
# Score per KPI: s = 100 * clip((x - good) / (bad - good), 0, 1); 100 = worst.
# KPIs missing from the data are skipped and the weights are renormalised.
SUBINDEX_SPECS = {
    "aqs": [("aq1_pm25", 15, 75, 0.30), ("aq2_pm10", 45, 200, 0.25),
            ("aq3_unhealthy_days_pct", 0, 50, 0.25), ("aq4_dust_storm_days", 0, 6, 0.20)],
    "wps": [("wp1_ecoli_fail_pct", 0, 10, 0.30), ("wp2_turbidity_ntu", 1, 5, 0.15),
            ("wp3_chlorine_residual_mg_l", 0.5, 0.1, 0.20), ("wp4_leak_sewage_per_10k", 0, 5, 0.20),
            ("wp5_water_complaints_per_1k", 0, 10, 0.15)],
    "sns": [("sn1_bin_fill_pct", 50, 95, 0.25), ("sn2_overflow_pct", 0, 30, 0.35),
            ("sn3_missed_collection_pct", 0, 20, 0.25), ("sn4_hours_at_overflow", 0, 48, 0.15)],
    "rpi": [("sr1_requests_per_1k", 5, 40, 0.35), ("sr2_env_health_share_pct", 10, 50, 0.25),
            ("sr3_resolution_days", 2, 14, 0.20), ("sr4_repeat_complaint_pct", 0, 30, 0.20)],
    "hts": [("ht2_heat_wave_days", 0, 10, 0.70), ("ht3_longest_heat_spell", 0, 7, 0.30)],
    "agi": [("ha3_avg_travel_min", 10, 30, 0.50), ("ha4_coverage_15min_pct", 100, 50, 0.50)],
}
# ---- Level 3: Environmental Stress Index weights ----
ESI_WEIGHTS = {"aqs": 0.35, "wps": 0.30, "sns": 0.20, "hts": 0.15}
ESI_NAMES = {"aqs": "air", "wps": "water", "sns": "waste", "hts": "heat"}


def load(path, **kw):
    """Reads a CSV (BOM-safe) and strips spaces from column names."""
    path = Path(path).resolve()
    if not path.exists():
        found = sorted(p.name for p in path.parent.glob("*.csv")) if path.parent.exists() else []
        sys.exit(f"Missing file: {path}\n  CSV files in that folder: {found or 'none (folder not found or empty)'}"
                 f"\n  Use --data-dir to point to the folder that holds the raw files.")
    df = pd.read_csv(path, encoding="utf-8-sig", **kw)
    df.columns = df.columns.str.strip()
    return df


def normalize_keys(df, name, monthly=True):
    """district_key -> int, month -> 'YYYY-MM-01' text, so every table joins on identical keys."""
    need = ["district_key"] + (["month"] if monthly else [])
    lost = [c for c in need if c not in df.columns]
    if lost:
        sys.exit(f"{name}: missing key column(s) {lost}. Columns found: {list(df.columns)}")
    df["district_key"] = pd.to_numeric(df.district_key, errors="coerce")
    if df.district_key.isna().any():
        sys.exit(f"{name}: {int(df.district_key.isna().sum())} rows have a non-numeric or empty district_key")
    df["district_key"] = df.district_key.astype(int)
    if monthly:
        df["month"] = pd.to_datetime(df.month).dt.strftime("%Y-%m-01")
    return df


def prepare_facilities(fac):
    """Maps facility column names to the expected ones; one row = one center when no count column exists."""
    if "district_key" not in fac.columns:
        sys.exit(f"{FACILITIES_FILE}: no district_key column. Columns found: {list(fac.columns)}")
    for target, names in FACILITY_ALIASES.items():
        hit = next((n for n in names if n in fac.columns), None)
        if hit and hit != target:
            fac = fac.rename(columns={hit: target})
    if "fclass" not in fac.columns:
        sys.exit(f"{FACILITIES_FILE}: no facility class column {FACILITY_ALIASES['fclass']}. "
                 f"Columns found: {list(fac.columns)}")
    if "healthcare_capacity" not in fac.columns:
        sys.exit(f"{FACILITIES_FILE}: no capacity column {FACILITY_ALIASES['healthcare_capacity']}. "
                 f"Columns found: {list(fac.columns)}")
    if "healthcare_centers" not in fac.columns:
        print(f"  {FACILITIES_FILE}: no centers column, counting each row as one center")
        fac["healthcare_centers"] = 1
    n_bad = fac.district_key.isna().sum()
    if n_bad:
        print(f"  {FACILITIES_FILE}: dropping {int(n_bad)} facilities with no district_key")
        fac = fac[fac.district_key.notna()].copy()
    fac = normalize_keys(fac, FACILITIES_FILE, monthly=False)
    fac["fclass"] = fac.fclass.astype(str).str.strip().str.lower()
    used = PRIMARY_CARE_CLASSES + HOSPITAL_CLASSES
    print(f"  {FACILITIES_FILE}: classes found {fac.fclass.value_counts().to_dict()}")
    if not fac.fclass.isin(used).any():
        sys.exit(f"{FACILITIES_FILE}: none of the classes {used} found; update PRIMARY_CARE_CLASSES / HOSPITAL_CLASSES")
    return fac


def check_panel(name, df, expect_rows=None):
    """Checks that (district_key, month) is a unique key and no district is missing."""
    dup = df.duplicated(KEYS).sum()
    n_d, n_m = df.district_key.nunique(), df.month.nunique()
    status = "OK" if dup == 0 else f"!! {dup} duplicate rows"
    print(f"  {name:<28} rows={len(df):>4}  districts={n_d}  months={n_m}  {status}")
    if dup:
        sys.exit(f"Key is not unique in {name}")
    if expect_rows and len(df) != expect_rows:
        print(f"  !! expected {expect_rows} rows in {name}")


def prefix(df, cols, pre):
    return df[KEYS + cols].rename(columns={c: pre + c for c in cols})


def build_events(cal, districts, months):
    """Turns the event table (start/end + list of districts) into district x month rows."""
    rows = []
    for _, e in cal.iterrows():
        txt = str(e.affected_districts)
        named = [int(x) for x in re.findall(r"\b(10\d{2})\b", txt)]
        is_all = "all" in txt.lower()
        targets = districts if is_all else named
        for m in pd.date_range(pd.Timestamp(e.start_date).replace(day=1),
                               pd.Timestamp(e.end_date).replace(day=1), freq="MS"):
            if m not in months:
                continue
            for d in targets:
                rows.append(dict(district_key=d, month=m.strftime("%Y-%m-%d"),
                                 event_id=e.event_id, event_type=e.event_type,
                                 hit_directly=(d in named) or (not is_all),
                                 affects_pillars=e.affects_pillars))
    return pd.DataFrame(rows)


# ---------- KPI framework helpers ----------
def opt(df, *names):
    """First column found among names, else None."""
    for n in names:
        if n in df.columns:
            return df[n]
    return None


def as_pct(x):
    """Percentage on a 0-100 scale; columns stored as 0-1 fractions are multiplied by 100."""
    if x is None:
        return None
    return x * 100 if x.max() <= 1 else x


def s(x, good, bad):
    """0-100 stress score, 100 = worst. Works for 'higher is better' KPIs by setting good > bad."""
    return 100 * np.clip((x - good) / (bad - good), 0, 1)


def weighted_index(parts):
    """Weighted mean of (score, weight) pairs; weights renormalised over non-missing scores per row."""
    if not parts:
        return np.nan
    num = sum(sc.fillna(0) * w for sc, w in parts)
    den = sum(sc.notna() * w for sc, w in parts)
    return num / den.replace(0, np.nan)


def annual_growth(m, col):
    """Annual growth in %: year-over-year when 12 months of history exist, else annualised month-over-month.
    Expects m sorted by district_key, month with a default index."""
    g = m.groupby("district_key")[col]
    yoy = m[col] / g.shift(12) - 1
    mom = (m[col] / g.shift(1)) ** 12 - 1
    out = 100 * yoy.fillna(mom)
    return out.groupby(m.district_key).bfill()


def level1_kpis(m):
    """Level 1: raw pillar KPIs per district-month. Returns the KPIs that could not be computed."""
    days = pd.to_datetime(m.month).dt.days_in_month
    pop = m.residents_registered
    src = {
        # Pillar 1: Air quality
        "aq1_pm25": m.p3_pm25_ugm3,
        "aq2_pm10": m.p3_pm10_ugm3,
        "aq3_unhealthy_days_pct": 100 * m.p3_days_aqi_gt150 / days,     # proxy: data counts AQI > 150
        "aq4_dust_storm_days": opt(m, "dust_storm_days"),
        # Pillar 2: Water pollution
        "wp1_ecoli_fail_pct": as_pct(m.pw_ecoli_positive_pct),
        "wp2_turbidity_ntu": m.pw_turbidity_ntu,                        # proxy: mean NTU, not % samples over limit
        "wp3_chlorine_residual_mg_l": m.pw_chlorine_residual_mg_l,      # proxy: mean residual, not % compliant samples
        "wp4_leak_sewage_per_10k": 1e4 * (m.pw_leak_incidents + m.pw_sewage_overflow_incidents) / pop,
        "wp5_water_complaints_per_1k": 1e3 * m.p4_water_complaints / pop,
        # Pillar 3: Waste and sanitation
        "sn1_bin_fill_pct": as_pct(m.p2_avg_bin_fill_pct),
        "sn2_overflow_pct": as_pct(m.p2_bin_hours_ge85_pct),            # proxy: % bin-hours at or above 85%
        "sn3_missed_collection_pct": 100 - as_pct(m.p2_collection_efficiency_pct),
        "sn4_hours_at_overflow": opt(m, "p2_hours_at_overflow"),
        # Pillar 4: Service requests
        "sr1_requests_per_1k": m.p4_complaints_per_1000,
        "sr2_env_health_share_pct": 100 * (m.p4_water_complaints + m.p4_waste_complaints
                                           + m.p4_pollution_complaints) / m.p4_total_complaints.replace(0, np.nan),
        "sr3_resolution_days": m.p4_avg_resolution_hours / 24,          # mean, framework asks for median
        "sr4_repeat_complaint_pct": opt(m, "p4_repeat_complaint_pct"),
        # Pillar 5: Population and demographics
        "pd1_population": pop,
        "pd2_growth_pct_yr": annual_growth(m, "residents_registered"),
        "pd3_elderly_pct": as_pct(m.elderly_pct),
        "pd4_children_pct": as_pct(m.children_pct),
        "pd6_density_km2": m.density_per_km2,
        # Pillar 6: Healthcare access and capacity
        "ha1_capacity_visits": m.pc_effective_capacity,
        "ha2_centers_per_10k": 1e4 * (m.pc_centers + m.hospital_centers) / pop,
        "ha3_avg_travel_min": opt(m, "avg_travel_time_min"),
        "ha4_coverage_15min_pct": as_pct(opt(m, "accessible_15min_pct")),
        # Climate driver: heat
        "ht1_avg_temp_c": m.avg_temp_c,
        "ht2_heat_wave_days": opt(m, "heat_days_ge40c", "heat_days_ge38c"),  # proxy when only >= 38 C exists
        "ht3_longest_heat_spell": opt(m, "longest_heat_spell_days"),
    }
    missing = [k for k, v in src.items() if v is None]
    for k, v in src.items():
        if v is not None:
            m[k] = v
    m["pd5_vulnerable_pct"] = m.pd3_elderly_pct + m.pd4_children_pct
    return missing


def level2_subindices(m):
    """Level 2: one 0-100 sub-index per pillar (and heat)."""
    for idx, spec in SUBINDEX_SPECS.items():
        parts = [(s(m[c], good, bad), w) for c, good, bad, w in spec if c in m.columns]
        m[idx] = weighted_index(parts)


def city_kpis(m):
    """Level 5: city Command Center KPIs per month."""
    w = m.residents_registered
    t = m.assign(_w=w, _esi_w=m.environmental_stress_index * w,
                 _over=m.cpr > 1, _high_esi=m.environmental_stress_index >= 60,
                 _high_need=m.need_score >= NEED_HIGH)
    if "ha4_coverage_15min_pct" in m.columns:
        t["_cov_w"] = m.ha4_coverage_15min_pct * w
    g = t.groupby("month")
    city = pd.DataFrame({
        "c1_total_visits_demand": g.total_visits_demand.sum(),
        "c1_total_unmet_visits": g.unmet_visits.sum(),
        "c2_districts_over_capacity": g._over.sum(),
        "c3_districts_high_esi": g._high_esi.sum(),
        "c4_city_esi": g._esi_w.sum() / g._w.sum(),
        "c6_open_opportunities": g._high_need.sum(),
    })
    if "_cov_w" in t.columns:
        city["c4_city_coverage_15min_pct"] = g._cov_w.sum() / g._w.sum()
    # C5 (surge alerts) needs the Module 2 forecast; until then High is triggered by C2 only
    city["c7_preparedness_status"] = np.select(
        [city.c2_districts_over_capacity >= 3,
         (city.c2_districts_over_capacity >= 1) | (city.c3_districts_high_esi >= 1)],
        ["High", "Moderate"], default="Normal")
    return city.reset_index().round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                    help="folder with population, pw, p2..p4 and the other raw files")
    ap.add_argument("--facilities-dir", default=None,
                    help=f"folder with {FACILITIES_FILE} (default = data-dir)")
    ap.add_argument("--population-file", default="population_monthly_clean.csv")
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="output folder")
    a = ap.parse_args()
    D = Path(a.data_dir)
    F = Path(a.facilities_dir) if a.facilities_dir else D
    O = Path(a.out)
    print(f"Data folder  : {D.resolve()}\nOutput folder: {O.resolve()}\n")

    # ---------- 1) Load ----------
    pop = load(D / a.population_file)
    pw = load(D / "pw_water_pollution_monthly.csv")
    wx = load(D / "weather_monthly.csv")
    p2 = load(D / "p2_solid_waste_monthly.csv")
    p3 = load(D / "p3_air_quality_health_monthly.csv")
    p4 = load(D / "p4_service_requests_monthly.csv")
    prof = load(D / "district_population_profile.csv")
    fac = load(F / FACILITIES_FILE)
    cal = load(D / "environment_event_calendar.csv")

    for n, df in [("population", pop), ("pw water", pw), ("p2 waste", p2),
                  ("p3 air/health", p3), ("p4 service requests", p4)]:
        normalize_keys(df, n)
    wx["month"] = pd.to_datetime(wx.month).dt.strftime("%Y-%m-01")
    if wx.month.duplicated().any():
        sys.exit("weather_monthly.csv: month is not unique (one row per month expected)")
    prof = normalize_keys(prof, "district profile", monthly=False)
    if prof.district_key.duplicated().any():
        sys.exit("district_population_profile.csv: district_key is not unique")
    fac = prepare_facilities(fac)

    # Same period for all tables: Jan 2025 -> Sep 2026
    pop = pop[pop.month <= "2026-09-01"].copy()

    # ---------- 2) Key checks ----------
    print("Key check (district_key + month):")
    for n, df in [("population", pop), ("pw water", pw), ("p2 waste", p2),
                  ("p3 air/health", p3), ("p4 service requests", p4)]:
        check_panel(n, df, expect_rows=315)
    base = set(map(tuple, pop[KEYS].values))
    for n, df in [("pw", pw), ("p2", p2), ("p3", p3), ("p4", p4)]:
        miss = base - set(map(tuple, df[KEYS].values))
        if miss:
            sys.exit(f"{n}: missing {len(miss)} rows compared to population")
    known = set(prof.district_key)
    bad = set(fac.district_key) - known
    if bad:
        sys.exit(f"Facilities in unknown districts: {bad}")
    print("  All tables match on the same 315 rows and every facility has a district.\n")

    # ---------- 3) Healthcare capacity (static per district) ----------
    pc = (fac[fac.fclass.isin(PRIMARY_CARE_CLASSES)]
          .groupby("district_key").agg(pc_centers=("healthcare_centers", "sum"),
                                       pc_capacity_visits=("healthcare_capacity", "sum")))
    hp = (fac[fac.fclass.isin(HOSPITAL_CLASSES)]
          .groupby("district_key").agg(hospital_centers=("healthcare_centers", "sum"),
                                       hospital_capacity_visits=("healthcare_capacity", "sum")))
    cap = pc.join(hp, how="outer").fillna(0).reset_index()
    cap["pc_effective_capacity"] = cap.pc_capacity_visits + HOSP_OPD_SHARE * cap.hospital_capacity_visits

    # ---------- 4) Merge ----------
    m = pop[KEYS + ["residents_registered", "households", "children_pct", "elderly_pct",
                    "daytime_workers_inflow", "students_day", "daytime_population",
                    "density_per_km2", "attractiveness_index"]].copy()
    m = m.rename(columns={"attractiveness_index": "attractiveness_index_old"})
    m = m.merge(wx, on="month", how="left")
    m = m.merge(prefix(pw, ["metered_consumption_m3", "water_supplied_m3", "nrw_pct", "night_flow_ratio",
                            "leak_incidents", "supply_interruption_hours", "water_age_hours",
                            "chlorine_residual_mg_l", "turbidity_ntu", "ecoli_positive_pct",
                            "sewage_overflow_incidents", "water_pollution_index",
                            "waterborne_clinic_visits", "waterborne_er_visits",
                            "waterborne_visits_total"], "pw_"), on=KEYS)
    m = m.merge(prefix(p2, ["bins_installed", "avg_bin_fill_pct", "bin_hours_ge85_pct",
                            "overflow_incidents", "collection_efficiency_pct", "collection_disruption_pp", "waste_generated_tons",
                            "uncollected_tons", "organic_tons", "recyclable_tons",
                            "recycling_diversion_pct", "odor_pest_risk_index"], "p2_"), on=KEYS)
    m = m.merge(prefix(p3, ["population_exposed", "avg_aqi", "max_aqi", "days_aqi_gt150", "pm25_ugm3",
                            "pm10_ugm3", "no2_ugm3", "o3_ugm3", "respiratory_clinic_visits",
                            "respiratory_er_visits", "respiratory_visits_total",
                            "heat_related_visits", "env_attractiveness_score"], "p3_"), on=KEYS)
    m = m.merge(prefix(p4, ["total_complaints", "water_complaints", "waste_complaints",
                            "pollution_complaints", "infrastructure_complaints", "complaints_per_1000",
                            "avg_resolution_hours", "sla_breach_pct", "open_requests_end_month",
                            "citizen_satisfaction_pct", "service_need_index"], "p4_"), on=KEYS)
    # All pillar event flags in one column
    flag_src = [d[KEYS + ["event_flag"]] for d in (pw, p2, p3, p4) if "event_flag" in d.columns]
    if flag_src:
        flags = (pd.concat(flag_src)
                 .groupby(KEYS).event_flag
                 .apply(lambda s: "|".join(sorted({x for v in s.dropna() for x in str(v).split("|") if x})))
                 .rename("event_flags").reset_index())
    else:
        flags = pd.DataFrame(columns=KEYS + ["event_flags"])
    m = m.merge(flags, on=KEYS, how="left").merge(cap, on="district_key", how="left")
    m["event_flags"] = m.event_flags.fillna("")
    cap_cols = [c for c in cap.columns if c != "district_key"]
    no_cap = sorted(m.loc[m.pc_effective_capacity.fillna(0) == 0, "district_key"].unique())
    if no_cap:
        print(f"  !! districts with no primary-care capacity (CPR left empty): {no_cap}")
    m[cap_cols] = m[cap_cols].fillna(0)
    extra = [c for c in PROFILE_OPTIONAL if c in prof.columns]
    if extra:
        m = m.merge(prof[["district_key"] + extra], on="district_key", how="left")
    m = m.sort_values(KEYS).reset_index(drop=True)

    # ---------- 5) Health indicators (link between demand and capacity) ----------
    E = m.p3_population_exposed
    age = 1 + 1.2 * (m.children_pct - 0.28) + 2.5 * (m.elderly_pct - 0.04)
    m["nonresp_visits"] = (E * VISITS_PER_PERSON_MONTH * age).round(0)
    m["total_visits_demand"] = (m.nonresp_visits + m.p3_respiratory_visits_total
                                + m.pw_waterborne_visits_total + m.p3_heat_related_visits)   # HD1
    m["pc_visits_demand"] = (PC_SHARE * m.total_visits_demand).round(0)
    m["cpr"] = m.pc_visits_demand / m.pc_effective_capacity.replace(0, np.nan)                                   # HD6
    m["wait_days"] = 0.4 + 9 * np.maximum(m.cpr - 0.75, 0) ** 1.3
    m["unmet_visits"] = np.maximum(m.pc_visits_demand - 0.95 * m.pc_effective_capacity, 0).round(0)  # HD7
    m["unmet_share"] = m.unmet_visits / m.pc_visits_demand
    m["service_quality"] = np.clip(1 - 0.5 * (m.wait_days / 5) - m.unmet_share, 0, 1)

    # ---------- 6) Level 1-2: pillar KPIs and sub-indices ----------
    missing_kpis = level1_kpis(m)
    level2_subindices(m)
    m = m.copy()   # defragment after many column inserts

    # ---------- 7) Level 3: Environmental Impact and Health Demand ----------
    m["environmental_stress_index"] = weighted_index([(m[k], w) for k, w in ESI_WEIGHTS.items()])   # EI1
    m["esi_band"] = pd.cut(m.environmental_stress_index, [-0.1, 30, 60, 100.1],
                           right=False, labels=["Low", "Moderate", "High"])                         # EI2
    contrib = pd.DataFrame({k: m[k] * w for k, w in ESI_WEIGHTS.items()})
    m["esi_dominant_stressor"] = contrib.fillna(-1).idxmax(axis=1).map(ESI_NAMES)                    # EI3
    by_d = m.groupby("district_key").environmental_stress_index
    m["esi_trend_12m"] = m.environmental_stress_index - by_d.shift(12)                               # EI4
    m["esi_volatility_12m"] = by_d.transform(lambda x: x.rolling(12, min_periods=6).std())           # EI5

    total = m.total_visits_demand.replace(0, np.nan)
    m["hd2_demand_change_pct"] = 100 * (total / m.groupby("district_key").total_visits_demand.shift(1) - 1)
    m["hd3_resp_share_pct"] = 100 * m.p3_respiratory_visits_total / total
    m["hd4_waterborne_share_pct"] = 100 * m.pw_waterborne_visits_total / total
    # HD5 proxy until the Module 2 counterfactual exists: share of environment-sensitive visits
    m["hd5_env_sensitive_share_pct"] = 100 * (m.p3_respiratory_visits_total + m.pw_waterborne_visits_total
                                              + m.p3_heat_related_visits) / total
    m["hd8_staff_required"] = m.pc_visits_demand / VISITS_PER_STAFF_MONTH
    if "staff_current" in m.columns:
        m["hd8_staff_gap"] = m.hd8_staff_required - m.staff_current
    m["hd_demand_growth_pct_yr"] = annual_growth(m, "total_visits_demand")

    # ---------- 8) Level 4: decision scores ----------
    m["need_score"] = weighted_index([
        (s(m.cpr, 0.8, 1.5), 0.40),
        (s(m.pd2_growth_pct_yr, 0, 10), 0.20),
        (m.agi, 0.20),
        (s(m.pd5_vulnerable_pct, 30, 55), 0.10),
        (m.environmental_stress_index, 0.10)])
    # Forecast risk: ESI volatility only; the prediction-interval width is added after Module 2
    m["forecast_risk"] = weighted_index([(s(m.esi_volatility_12m, 0, 10), 0.5)])
    competition = s(m.ha2_centers_per_10k, 0, 3)
    parts = [(s(100 * m.unmet_share, 0, 30), 0.35),
             (s(m.hd_demand_growth_pct_yr, 0, 10), 0.20),
             (100 - competition, 0.15),
             (100 - m.forecast_risk, 0.10)]
    if "income_index" in m.columns:
        inc = m.income_index
        parts.append((100 * (inc - inc.min()) / (inc.max() - inc.min()), 0.20))
    m["commercial_attractiveness_score"] = weighted_index(parts)

    # Residential attractiveness (closed loop): base + (100-ESI) + service need + healthcare service quality
    y26 = (m[m.month >= "2026-01-01"].groupby("district_key")
           .agg(esi0=("environmental_stress_index", "mean"), need0=("p4_service_need_index", "mean")))
    m = m.merge(y26, on="district_key").merge(prof[["district_key", "base_attractiveness"]], on="district_key")
    m["attractiveness_index_v2"] = np.clip(
        m.base_attractiveness - 0.30 * (m.environmental_stress_index - m.esi0) / 100
        - 0.15 * (m.p4_service_need_index - m.need0) / 100 + 0.30 * (m.service_quality - 1), 0, 1)
    m = m.drop(columns=["esi0", "need0", "base_attractiveness"]).sort_values(KEYS)
    m = m.round(3)

    # ---------- 9) Level 5: city KPIs ----------
    m = m.copy()
    city = city_kpis(m)

    # ---------- 10) Events ----------
    ev = build_events(cal, sorted(known), set(pd.to_datetime(m.month.unique())))

    # ---------- 11) Save + report ----------
    O.mkdir(parents=True, exist_ok=True)
    m_out = m.drop(columns=[c for c in m.columns if c.startswith("_")])
    m_out.to_csv(O / "master_panel_linked.csv", index=False, encoding="utf-8-sig")
    city.to_csv(O / "city_kpis_monthly.csv", index=False, encoding="utf-8-sig")
    ev.to_csv(O / "event_district_month.csv", index=False, encoding="utf-8-sig")
    print(f"master_panel_linked.csv : {m.shape[0]} rows x {m.shape[1]} columns, missing values = {int(m.isna().sum().sum())}")
    print(f"city_kpis_monthly.csv   : {len(city)} rows (one per month)")
    print(f"event_district_month.csv: {len(ev)} rows (event x district x month)")
    if missing_kpis:
        print("\nKPIs not in the data (skipped, sub-index weights renormalised):")
        print("  " + ", ".join(missing_kpis))
    if "heat_days_ge40c" not in m.columns:
        print("  ht2_heat_wave_days uses heat_days_ge38c as a proxy for days >= 40 C")

    print("\nLink checks (signs should make sense):")
    cols = ["pw_water_pollution_index", "pw_waterborne_visits_total", "p2_collection_efficiency_pct",
            "p4_water_complaints", "p4_citizen_satisfaction_pct", "p3_avg_aqi", "p3_respiratory_visits_total",
            "environmental_stress_index", "cpr"]
    m["_resp_per_1000"] = 1000 * m.p3_respiratory_visits_total / m.p3_population_exposed
    m["_wb_per_1000"] = 1000 * m.pw_waterborne_visits_total / m.p3_population_exposed
    cols = cols + ["_resp_per_1000", "_wb_per_1000"]
    c = m[cols].corr()
    print("  Water pollution <-> waterborne visits (per 1000)  :", round(c.loc["pw_water_pollution_index", "_wb_per_1000"], 2))
    print("  Water pollution <-> water complaints              :", round(c.loc["pw_water_pollution_index", "p4_water_complaints"], 2))
    print("  AQI <-> respiratory visits (per 1000)             :", round(c.loc["p3_avg_aqi", "_resp_per_1000"], 2))
    print("  Environmental stress <-> citizen satisfaction     :", round(c.loc["environmental_stress_index", "p4_citizen_satisfaction_pct"], 2))

if __name__ == "__main__":
    main()