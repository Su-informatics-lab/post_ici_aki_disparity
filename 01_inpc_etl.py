#!/usr/bin/env python3
"""
Post-ICI AKI × SDoH — INPC ETL (v3 — phenotype rigor)
Runs on Quartz HPC at IU.
Reads from: /N/project/depot/hw56/irAKI_data/structured_data/

Purpose-built OMOP CSV dump of post-ICI patients from INPC.
Replicates AoU base model (demographics + NCI-CCI + cancer type +
ICI class + nephrotoxins → AKI) without SDoH (no surveys in INPC).

PHENOTYPE RIGOR CHANGELOG (v3, harmonized with dualr-graph):
  - ADDED baseline Cr fallback: median [-365,-7], fallback to last [-365,-1]
  - ADDED pre-ICI AKI washout: exclude Cr ≥1.5× in [-90, 0] days
  - FIXED Cr plausibility floor: >= 0.1 mg/dL (was > 0)
  - FIXED nephrotoxin timing: ±90d of ICI (was any exposure ever)
  - ADDED baseline eGFR (CKD-EPI 2021 race-free equation)

CCI FIX LOG (v2, 2026-05-16):
  - REMOVED hierarchy pre-processing (zeroing raw flags)
  - FIXED MI scoring: OR logic (max 1pt), was additive (2pt)
  - FIXED Paralysis/CVD: score independently (was hierarchical)
  - ADDED NCI continuous index alongside Charlson integer score
  - Uses shared nci_cci_scoring.py module

Output: results/inpc/*.csv  (same schema as AoU for 01b_psm.R + 02_models.R)
Usage:  python 01_inpc_etl.py
Then:   Rscript 01b_psm.R inpc
        Rscript 02_models.R inpc
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Import corrected NCI-CCI scoring
from nci_cci_scoring import (
    NCI_CCI_CONDITIONS,
    NCI_CODESETS,
    compute_charlson_score,
    compute_nci_index,
)

DATA = "/N/project/depot/hw56/irAKI_data/structured_data"
RESULTS = "results/inpc"
os.makedirs(RESULTS, exist_ok=True)

# ── CONSORT flowchart tracking ────────────────────────────────────
consort = {}

print("=" * 70)
print("POST-ICI AKI — INPC ETL (v3 — phenotype rigor)")
print("=" * 70)
print(f"  Data: {DATA}")
print(f"  Output: {RESULTS}/")


def save(df, filename):
    path = os.path.join(RESULTS, filename)
    df.to_csv(path, index=False)
    print(f"  Saved: {path} ({len(df):,} rows)")


def parse_date(s):
    """Handle both ISO (2023-02-27T00:00:00) and SAS (28APR2022) date formats."""
    return pd.to_datetime(s, format="mixed", dayfirst=False, errors="coerce")


# ═══════════════════════════════════════════════════════════════════
# LOAD CORE TABLES
# ═══════════════════════════════════════════════════════════════════
print("\n  Loading core tables...")
person = pd.read_csv(f"{DATA}/r6335_person.csv", low_memory=False)
consort["total_inpc"] = len(person)
print(f"  person: {len(person):,}")

drug = pd.read_csv(
    f"{DATA}/r6335_drug_exposure.csv",
    low_memory=False,
    usecols=[
        "person_id",
        "drug_concept_id",
        "drug_source_concept_id",
        "drug_exposure_start_date",
        "drug_source_value",
    ],
)
print(f"  drug_exposure: {len(drug):,}")

concept = pd.read_csv(
    f"{DATA}/r6335_concept.csv",
    encoding="cp1252",
    low_memory=False,
    usecols=["concept_id", "concept_name", "vocabulary_id", "concept_class_id"],
)
print(f"  concept: {len(concept):,}")

cond = pd.read_csv(
    f"{DATA}/r6335_condition_occurrence.csv",
    low_memory=False,
    usecols=[
        "person_id",
        "condition_concept_id",
        "condition_start_date",
        "condition_source_value",
    ],
)
print(f"  condition_occurrence: {len(cond):,}")

# Measurement (UPPERCASE columns in INPC dump)
meas = pd.read_csv(f"{DATA}/r6335_measurement.csv", low_memory=False)
# Standardize column names to lowercase
meas.columns = [c.lower() for c in meas.columns]
print(f"  measurement: {len(meas):,}")


# ═══════════════════════════════════════════════════════════════════
# STEP 1: ICI COHORT + AKI PHENOTYPING
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1: ICI Cohort + AKI Phenotyping")
print("=" * 70)

# ── 1a. Find ICI drugs via concept name matching ─────────────────
ICI_AGENTS = [
    "nivolumab",
    "pembrolizumab",
    "atezolizumab",
    "durvalumab",
    "avelumab",
    "cemiplimab",
    "ipilimumab",
    "tremelimumab",
    "dostarlimab",
    "relatlimab",
]

# Also search drug_source_value for INPC (free-text medication name)
ici_concept = concept[
    concept.concept_name.str.lower().apply(
        lambda x: any(a in str(x) for a in ICI_AGENTS) if pd.notna(x) else False
    )
]
ici_concept_ids = set(ici_concept.concept_id.tolist())
print(f"  ICI concept IDs from concept table: {len(ici_concept_ids)}")

# Match by drug_concept_id
ici_by_concept = drug[drug.drug_concept_id.isin(ici_concept_ids)]
print(
    f"    drug_concept_id matches: {len(ici_by_concept):,} records, "
    f"{ici_by_concept.person_id.nunique():,} patients"
)

# Match by drug_source_concept_id (PD-L1 drugs often mapped here in INPC)
drug["drug_source_concept_id"] = pd.to_numeric(
    drug["drug_source_concept_id"], errors="coerce"
)
ici_by_source_concept = drug[drug.drug_source_concept_id.isin(ici_concept_ids)]
print(
    f"    drug_source_concept_id matches: {len(ici_by_source_concept):,} records, "
    f"{ici_by_source_concept.person_id.nunique():,} patients"
)

# Match by source value (free text keyword)
ici_by_source = drug[
    drug.drug_source_value.str.lower().apply(
        lambda x: any(a in str(x) for a in ICI_AGENTS) if pd.notna(x) else False
    )
]
print(
    f"    drug_source_value keyword matches: {len(ici_by_source):,} records, "
    f"{ici_by_source.person_id.nunique():,} patients"
)

ici_drug = pd.concat(
    [ici_by_concept, ici_by_source_concept, ici_by_source]
).drop_duplicates()
print(f"  ICI drug records (union): {len(ici_drug):,}")

# Index date + drug list per patient
ici_drug["drug_exposure_start_date"] = parse_date(ici_drug["drug_exposure_start_date"])
ici_patients = (
    ici_drug.groupby("person_id")
    .agg(
        ici_index_date=("drug_exposure_start_date", "min"),
    )
    .reset_index()
)

# Build drug list — resolve names from BOTH concept ID columns
# (PD-L1 drugs may only have names via drug_source_concept_id)
drug_list = ici_drug.copy()
drug_list = drug_list.merge(
    concept[["concept_id", "concept_name"]].rename(
        columns={"concept_name": "name_from_concept"}
    ),
    left_on="drug_concept_id",
    right_on="concept_id",
    how="left",
).drop(columns=["concept_id"], errors="ignore")
drug_list = drug_list.merge(
    concept[["concept_id", "concept_name"]].rename(
        columns={"concept_name": "name_from_source_concept"}
    ),
    left_on="drug_source_concept_id",
    right_on="concept_id",
    how="left",
).drop(columns=["concept_id"], errors="ignore")
# Prefer name_from_concept, fall back to name_from_source_concept, then drug_source_value
drug_list["resolved_name"] = (
    drug_list["name_from_concept"]
    .fillna(drug_list["name_from_source_concept"])
    .fillna(drug_list["drug_source_value"])
)
drug_names = (
    drug_list.groupby("person_id")
    .resolved_name.apply(lambda x: list(x.dropna().str.lower().unique()))
    .reset_index()
)
drug_names.columns = ["person_id", "ici_drugs"]
ici_patients = ici_patients.merge(drug_names, on="person_id", how="left")
consort["ici_treated"] = len(ici_patients)
print(f"  ICI-treated patients: {len(ici_patients):,}")

# ── QC: ICI class breakdown ──────────────────────────────────────
_has_pd1 = ici_patients.ici_drugs.apply(
    lambda x: (
        any(
            a in " ".join(str(d) for d in x)
            for a in ["nivolumab", "pembrolizumab", "cemiplimab", "dostarlimab"]
        )
        if isinstance(x, list)
        else False
    )
)
_has_pdl1 = ici_patients.ici_drugs.apply(
    lambda x: (
        any(
            a in " ".join(str(d) for d in x)
            for a in ["atezolizumab", "durvalumab", "avelumab"]
        )
        if isinstance(x, list)
        else False
    )
)
_has_ctla4 = ici_patients.ici_drugs.apply(
    lambda x: (
        any(a in " ".join(str(d) for d in x) for a in ["ipilimumab", "tremelimumab"])
        if isinstance(x, list)
        else False
    )
)
print(f"    anti-PD-1:  {_has_pd1.sum():,}")
print(f"    anti-PD-L1: {_has_pdl1.sum():,}")
print(f"    anti-CTLA-4: {_has_ctla4.sum():,}")

# ── 1b. Cancer diagnosis filter ──────────────────────────────────
# INPC condition_source_value format: "1284^^H35.341" — extract after ^^
cond["icd_raw"] = cond.condition_source_value.str.extract(r"\^\^(.+)$", expand=False)
cond["icd_raw"] = cond.icd_raw.fillna(cond.condition_source_value)
cond["icd_clean"] = (
    cond.icd_raw.str.replace(".", "", regex=False).str.upper().str.strip()
)

cancer_mask = cond.icd_clean.str.match(r"^C\d|^D0\d|^D[1234]\d", na=False)
cancer_pts = cond[cancer_mask].person_id.unique()
ici_cancer = ici_patients[ici_patients.person_id.isin(cancer_pts)]
consort["cancer_pts_total"] = len(cancer_pts)
consort["ici_cancer"] = len(ici_cancer)
print(f"  ICI + cancer: {len(ici_cancer):,}")

# ── 1c. Creatinine extraction ────────────────────────────────────
cr_concept = 3016723
cr = meas[meas.measurement_concept_id == cr_concept].copy()
cr = cr[
    cr.value_as_number.notna()
    & (cr.value_as_number >= 0.1)  # v3: plausibility floor (was > 0)
    & (cr.value_as_number < 30)
]
cr["measurement_date"] = parse_date(cr["measurement_date"])
cr = cr[cr.person_id.isin(ici_cancer.person_id)]
print(f"  Creatinine measurements (ICI+cancer): {len(cr):,}")

# ── 1d. Baseline + follow-up ─────────────────────────────────────
cr_merged = cr.merge(ici_cancer[["person_id", "ici_index_date"]], on="person_id")
cr_merged["days_from_ici"] = (
    cr_merged.measurement_date - cr_merged.ici_index_date
).dt.days

# Baseline: median Cr in [-365, -7] with fallback to last Cr [-365, -1]
# Primary window: stable outpatient window excluding peri-ICI labs
bl_main = cr_merged[(cr_merged.days_from_ici >= -365) & (cr_merged.days_from_ici <= -7)]
bl_primary = (
    bl_main.groupby("person_id")
    .agg(
        baseline_cr=("value_as_number", "median"),
        n_baseline=("value_as_number", "count"),
    )
    .reset_index()
)
# Fallback: last available Cr in [-365, -1] for patients without [-365, -7] data
# This rescues patients who only have labs in the week before ICI initiation
bl_fallback_pool = cr_merged[
    (cr_merged.days_from_ici >= -365) & (cr_merged.days_from_ici <= -1)
].sort_values(["person_id", "days_from_ici"])
bl_fallback = (
    bl_fallback_pool.groupby("person_id")
    .tail(1)[["person_id", "value_as_number"]]
    .rename(columns={"value_as_number": "baseline_cr"})
)
bl_fallback["n_baseline"] = 1
# Combine: primary where available, fallback for the rest
pids_primary = set(bl_primary.person_id)
bl_fallback_only = bl_fallback[~bl_fallback.person_id.isin(pids_primary)]
baseline = pd.concat([bl_primary, bl_fallback_only], ignore_index=True)
n_primary = len(bl_primary)
n_fallback = len(bl_fallback_only)
print(f"  Patients with baseline Cr: {len(baseline):,}")
print(f"    Primary [-365, -7] median: {n_primary:,}")
print(f"    Fallback [-365, -1] last:  {n_fallback:,}")
consort["has_baseline_cr"] = len(baseline)
consort["baseline_cr_primary"] = n_primary
consort["baseline_cr_fallback"] = n_fallback

followup_cr = cr_merged[
    (cr_merged.days_from_ici >= 1) & (cr_merged.days_from_ici <= 365)
]
followup = (
    followup_cr.groupby("person_id")
    .agg(
        max_followup_cr=("value_as_number", "max"),
        n_followup=("value_as_number", "count"),
    )
    .reset_index()
)
print(f"  Patients with follow-up Cr: {len(followup):,}")
consort["has_followup_cr"] = len(followup)

eligible = ici_cancer.merge(baseline, on="person_id").merge(followup, on="person_id")

# ── 1e. ESKD exclusion ───────────────────────────────────────────
eskd_codes = ["N186", "Z992", "Z490", "Z491", "Z492", "Z940"]
eskd_mask = cond.icd_clean.apply(
    lambda x: any(str(x).startswith(c) for c in eskd_codes) if pd.notna(x) else False
)
eskd_pts = cond[eskd_mask].person_id.unique()
pre_eskd = len(eligible)
eligible = eligible[~eligible.person_id.isin(eskd_pts)]
eligible = eligible[eligible.baseline_cr < 4.0]
consort["pre_eskd_exclusion"] = pre_eskd
consort["excluded_eskd"] = pre_eskd - len(eligible)
consort["excluded_no_baseline"] = consort["ici_cancer"] - consort["has_baseline_cr"]
print(f"  Excluded ESKD/transplant/baseline≥4: {pre_eskd - len(eligible)}")

# ── 1e2. Pre-ICI AKI washout ────────────────────────────────────
# Exclude patients with Cr ≥1.5× baseline in [-90, 0] days before ICI
# These have prevalent (not incident) AKI, contaminating the case definition
pre_ici_cr = cr_merged[
    (cr_merged.days_from_ici >= -90) & (cr_merged.days_from_ici <= 0)
]
pre_ici_cr = pre_ici_cr.merge(eligible[["person_id", "baseline_cr"]], on="person_id")
pre_ici_cr["pre_ratio"] = pre_ici_cr.value_as_number / pre_ici_cr.baseline_cr
washout_pids = set(pre_ici_cr[pre_ici_cr.pre_ratio >= 1.5].person_id.tolist())
pre_washout = len(eligible)
eligible = eligible[~eligible.person_id.isin(washout_pids)].copy()
print(f"  Washout (Cr ≥1.5× in 90d pre-ICI): {len(washout_pids)} excluded")
print(f"  Eligible cohort: {len(eligible):,}")
consort["excluded_washout"] = len(washout_pids)
consort["eligible"] = len(eligible)

# ── 1f. AKI phenotyping ──────────────────────────────────────────
eligible["max_cr_ratio"] = eligible.max_followup_cr / eligible.baseline_cr
eligible["max_delta_cr"] = eligible.max_followup_cr - eligible.baseline_cr

eligible["severity"] = (eligible.max_cr_ratio >= 1.5).astype(int)
eligible["aki_delta03"] = (eligible.max_delta_cr >= 0.3).astype(int)
eligible["aki_kdigo2"] = (eligible.max_cr_ratio >= 2.0).astype(int)
eligible["aki_kdigo3"] = (eligible.max_cr_ratio >= 3.0).astype(int)

# 180-day window
followup_180 = cr_merged[
    (cr_merged.days_from_ici >= 1) & (cr_merged.days_from_ici <= 180)
]
max_180 = followup_180.groupby("person_id").value_as_number.max().reset_index()
max_180.columns = ["person_id", "max_cr_180"]
eligible = eligible.merge(max_180, on="person_id", how="left")
eligible["aki_180d"] = (
    ((eligible.max_cr_180 / eligible.baseline_cr) >= 1.5).astype(int).fillna(0)
)

cases = eligible.severity.sum()
controls = (eligible.severity == 0).sum()
consort["cases"] = int(cases)
consort["controls"] = int(controls)
print(f"  Cases (Cr ≥1.5×): {cases:,} ({cases/len(eligible)*100:.1f}%)")
print(f"  Controls:          {controls:,}")

# ── Save CONSORT numbers ─────────────────────────────────────────
consort_df = pd.DataFrame([consort]).T.reset_index()
consort_df.columns = ["step", "n"]
save(consort_df, "00_consort_numbers.csv")
print("\n  CONSORT flowchart:")
for step, n in consort.items():
    print(f"    {step:35s} {int(n):>10,}")

save(eligible, "01_eligible_cohort.csv")


# ═══════════════════════════════════════════════════════════════════
# STEP 2: DEMOGRAPHICS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Demographics")
print("=" * 70)

demo = person[person.person_id.isin(eligible.person_id)].copy()

# Sex
gender_map = {8507: "Male", 8532: "Female"}
demo["sex_at_birth"] = demo.gender_concept_id.map(gender_map).fillna("Other")

# Race
race_map = {
    8516: "Black",
    8515: "Asian",
    8527: "White",
    8557: "Native_Hawaiian_PI",
    8657: "AIAN",
}
demo["race"] = demo.race_concept_id.map(race_map).fillna("Other")

# Ethnicity
eth_map = {38003563: "Hispanic", 38003564: "Not_Hispanic"}
demo["ethnicity"] = demo.ethnicity_concept_id.map(eth_map).fillna("Unknown")

# Age at ICI
demo = demo.merge(eligible[["person_id", "ici_index_date"]], on="person_id")
demo["age_at_ici"] = demo.ici_index_date.dt.year - demo.year_of_birth


def age_group(age):
    if pd.isna(age):
        return "Unknown"
    if age < 45:
        return "18-44"
    if age < 55:
        return "45-54"
    if age < 65:
        return "55-64"
    if age < 75:
        return "65-74"
    return "75+"


demo["age_group"] = demo.age_at_ici.apply(age_group)
print(f"  Demographics: {len(demo):,}")
for col in ["sex_at_birth", "race", "age_group"]:
    print(f"    {col}: {demo[col].value_counts().to_dict()}")

save(
    demo[
        [
            "person_id",
            "sex_at_birth",
            "race",
            "ethnicity",
            "age_group",
            "age_at_ici",
            "year_of_birth",
        ]
    ],
    "02_demographics.csv",
)

# ── Baseline eGFR (CKD-EPI 2021 race-free) ─────────────────────
# eGFR = 142 × min(Scr/κ, 1)^α × max(Scr/κ, 1)^(-1.200) × 0.9938^Age [× 1.012 if F]
# Standard covariate for any kidney injury study
print("  Computing baseline eGFR (CKD-EPI 2021 race-free)...")
egfr_df = demo[["person_id", "sex_at_birth", "age_at_ici"]].merge(
    eligible[["person_id", "baseline_cr"]], on="person_id"
)
scr = egfr_df.baseline_cr.values.astype(np.float64)
age_arr = egfr_df.age_at_ici.values.astype(np.float64)
is_female = (egfr_df.sex_at_birth == "Female").values
kappa = np.where(is_female, 0.7, 0.9)
alpha = np.where(is_female, -0.241, -0.302)
scr_over_k = scr / kappa
egfr_vals = (
    142
    * np.power(np.minimum(scr_over_k, 1.0), alpha)
    * np.power(np.maximum(scr_over_k, 1.0), -1.200)
    * np.power(0.9938, age_arr)
    * np.where(is_female, 1.012, 1.0)
)
egfr_df["baseline_egfr"] = egfr_vals.astype(np.float32)
n_ckd3 = (egfr_vals < 60).sum()
print(
    f"    eGFR: median={np.nanmedian(egfr_vals):.1f}, "
    f"IQR=[{np.nanpercentile(egfr_vals, 25):.1f}, {np.nanpercentile(egfr_vals, 75):.1f}], "
    f"<60 (CKD ≥3): {n_ckd3} ({n_ckd3/len(egfr_df)*100:.1f}%)"
)
demo = demo.merge(egfr_df[["person_id", "baseline_egfr"]], on="person_id", how="left")


# ═══════════════════════════════════════════════════════════════════
# STEP 3: NCI CHARLSON COMORBIDITY INDEX (v2 — corrected scoring)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: NCI Charlson Comorbidity Index (v2 — corrected scoring)")
print("=" * 70)

# Get diagnoses for eligible patients
eligible_cond = cond[cond.person_id.isin(eligible.person_id)].copy()
print(f"  Diagnosis records: {len(eligible_cond):,}")

# Build NCI-CCI flags
charlson = pd.DataFrame({"person_id": eligible.person_id.values})
for condition in NCI_CCI_CONDITIONS:
    charlson[condition] = 0

for condition, codes in NCI_CODESETS.items():
    all_prefixes = []
    for ver_codes in codes.values():
        all_prefixes.extend(ver_codes)
    mask = eligible_cond.icd_clean.apply(
        lambda x: (
            any(str(x).startswith(p) for p in all_prefixes) if pd.notna(x) else False
        )
    )
    flagged = eligible_cond[mask].person_id.unique()
    charlson.loc[charlson.person_id.isin(flagged), condition] = 1

# Cast to int8
for c in NCI_CCI_CONDITIONS:
    charlson[c] = charlson[c].astype("int8")

# ── SCORING (v2 fix) ─────────────────────────────────────────────
charlson["charlson_score"] = compute_charlson_score(charlson)
charlson["nci_index"] = compute_nci_index(charlson)

# ── BACKWARD COMPAT: keep old column name for R scripts ──────────
charlson["nci_cci_score"] = charlson["charlson_score"]

# QC
both_diab = (
    (charlson["Diabetes"] == 1) & (charlson["Diabetes_Complicated"] == 1)
).sum()
both_mi = ((charlson["Acute_MI"] == 1) & (charlson["History_MI"] == 1)).sum()
both_para_cvd = (
    (charlson["Paralysis"] == 1) & (charlson["Cerebrovascular_Disease"] == 1)
).sum()
print(f"  QC: {both_diab} pts w/ both diabetes flags (raw, NOT zeroed)")
print(f"  QC: {both_mi} pts w/ both MI types → Charlson MI = 1pt (OR, not 2)")
print(f"  QC: {both_para_cvd} pts w/ paralysis+CVD → both score independently")
print(
    f"  Charlson score: median {charlson.charlson_score.median():.0f}, "
    f"IQR {charlson.charlson_score.quantile(0.25):.0f}–"
    f"{charlson.charlson_score.quantile(0.75):.0f}, "
    f"max {charlson.charlson_score.max():.0f}"
)
print(
    f"  NCI index: median {charlson.nci_index.median():.3f}, "
    f"max {charlson.nci_index.max():.3f}"
)

for c in NCI_CCI_CONDITIONS:
    print(f"    {c:40s} {charlson[c].sum():>6,}  ({charlson[c].mean()*100:.1f}%)")

save(charlson, "03_nci_charlson.csv")


# ═══════════════════════════════════════════════════════════════════
# STEP 4: CANCER TYPE + NEPHROTOXINS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: Cancer Type + Nephrotoxins")
print("=" * 70)

# Cancer type
cancer_cond = cond[cond.person_id.isin(eligible.person_id) & cancer_mask].copy()
cancer_cond["condition_start_date"] = parse_date(cancer_cond["condition_start_date"])
cancer_cond = cancer_cond.merge(
    eligible[["person_id", "ici_index_date"]], on="person_id"
)


def classify_cancer(code):
    if pd.isna(code):
        return "Other_Solid"
    code = str(code).upper()
    if code.startswith("C34"):
        return "Lung"
    if code.startswith("C43"):
        return "Melanoma"
    if code.startswith("C64") or code.startswith("C65"):
        return "Renal_Cell"
    if code.startswith("C67"):
        return "Urothelial"
    if code.startswith("C50"):
        return "Breast"
    if (
        code.startswith("C10")
        or code.startswith("C11")
        or code.startswith("C12")
        or code.startswith("C13")
        or code.startswith("C14")
        or code.startswith("C32")
    ):
        return "Head_Neck"
    if code.startswith("C22"):
        return "Hepatocellular"
    if code.startswith("C18") or code.startswith("C19") or code.startswith("C20"):
        return "Colorectal"
    if (
        code.startswith("C81")
        or code.startswith("C82")
        or code.startswith("C83")
        or code.startswith("C84")
        or code.startswith("C85")
        or code.startswith("C91")
        or code.startswith("C92")
    ):
        return "Hematologic"
    return "Other_Solid"


cancer_cond["cancer_type"] = cancer_cond.icd_clean.apply(classify_cancer)
cancer_primary = (
    cancer_cond.groupby("person_id")
    .cancer_type.agg(lambda x: x.value_counts().index[0])
    .reset_index()
)


# ICI regimen classification
def classify_ici(drugs_list):
    if not isinstance(drugs_list, (list, np.ndarray)):
        return "unknown"
    drugs_str = " ".join([str(d).lower() for d in drugs_list])
    has_pd1 = any(
        a in drugs_str
        for a in ["nivolumab", "pembrolizumab", "cemiplimab", "dostarlimab"]
    )
    has_pdl1 = any(a in drugs_str for a in ["atezolizumab", "durvalumab", "avelumab"])
    has_ctla4 = any(a in drugs_str for a in ["ipilimumab", "tremelimumab"])
    if has_ctla4 and (has_pd1 or has_pdl1):
        return "combo_pd1_ctla4"
    if has_pd1:
        return "anti_pd1"
    if has_pdl1:
        return "anti_pdl1"
    if has_ctla4:
        return "anti_ctla4"
    return "other_ici"


ici_regimen = ici_cancer[["person_id", "ici_drugs"]].copy()
ici_regimen["ici_regimen"] = ici_regimen.ici_drugs.apply(classify_ici)

# Merge cancer + ICI
covariates = eligible[["person_id"]].merge(cancer_primary, on="person_id", how="left")
covariates = covariates.merge(
    ici_regimen[["person_id", "ici_regimen"]], on="person_id", how="left"
)
covariates.cancer_type = covariates.cancer_type.fillna("Unknown")
covariates.ici_regimen = covariates.ici_regimen.fillna("unknown")

# Nephrotoxins — time-windowed ±90d of ICI ("concurrent" per manuscript)
NEPHROTOXIN_CLASSES = {
    "ppi": [
        "omeprazole",
        "pantoprazole",
        "lansoprazole",
        "esomeprazole",
        "rabeprazole",
    ],
    "nsaid": [
        "ibuprofen",
        "naproxen",
        "diclofenac",
        "meloxicam",
        "celecoxib",
        "indomethacin",
        "ketorolac",
    ],
    "acei_arb": [
        "lisinopril",
        "enalapril",
        "ramipril",
        "benazepril",
        "losartan",
        "valsartan",
        "irbesartan",
        "olmesartan",
        "candesartan",
    ],
    "diuretic": [
        "furosemide",
        "hydrochlorothiazide",
        "chlorthalidone",
        "bumetanide",
        "torsemide",
        "spironolactone",
    ],
}

eligible_drug = drug[drug.person_id.isin(eligible.person_id)].copy()
eligible_drug["drug_exposure_start_date"] = parse_date(
    eligible_drug["drug_exposure_start_date"]
)
# Join ICI index dates and compute days from ICI
eligible_drug = eligible_drug.merge(
    eligible[["person_id", "ici_index_date"]], on="person_id"
)
eligible_drug["days_from_ici"] = (
    eligible_drug.drug_exposure_start_date - eligible_drug.ici_index_date
).dt.days
# Restrict to ±90 days of ICI (v3: was any exposure ever)
eligible_drug_windowed = eligible_drug[
    (eligible_drug.days_from_ici >= -90) & (eligible_drug.days_from_ici <= 90)
].copy()
# Resolve concept names for matching
eligible_drug_windowed = eligible_drug_windowed.merge(
    concept[["concept_id", "concept_name"]],
    left_on="drug_concept_id",
    right_on="concept_id",
    how="left",
).drop(columns=["concept_id"], errors="ignore")

for drug_class, agents in NEPHROTOXIN_CLASSES.items():
    by_name = eligible_drug_windowed.concept_name.str.lower().apply(
        lambda x: any(a in str(x) for a in agents) if pd.notna(x) else False
    )
    by_source = eligible_drug_windowed.drug_source_value.str.lower().apply(
        lambda x: any(a in str(x) for a in agents) if pd.notna(x) else False
    )
    flagged = eligible_drug_windowed[by_name | by_source].person_id.unique()
    covariates[drug_class] = covariates.person_id.isin(flagged).astype(int)
    print(
        f"  {drug_class}: {covariates[drug_class].sum():,} ({covariates[drug_class].mean()*100:.1f}%)"
    )

# Collapsed factors
covariates["cancer_type_collapsed"] = covariates.cancer_type.apply(
    lambda x: x if x in ["Lung", "Melanoma"] else "Other"
)
covariates["ici_collapsed"] = covariates.ici_regimen.apply(
    lambda x: "anti_pd1" if x == "anti_pd1" else "other_combo"
)

save(covariates, "05_covariates.csv")


# ═══════════════════════════════════════════════════════════════════
# STEP 5: MATCHING VARIABLES
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: Matching Variables")
print("=" * 70)

dx_counts = (
    eligible_cond.groupby("person_id")
    .agg(n_diagnoses=("condition_concept_id", "nunique"))
    .reset_index()
)

cond_dates = cond[cond.person_id.isin(eligible.person_id)].copy()
cond_dates["condition_start_date"] = parse_date(cond_dates["condition_start_date"])
ehr_length = (
    cond_dates.groupby("person_id")
    .agg(
        first_dx=("condition_start_date", "min"),
        last_dx=("condition_start_date", "max"),
    )
    .reset_index()
)
ehr_length["ehr_length_days"] = (ehr_length.last_dx - ehr_length.first_dx).dt.days
ref_date = ehr_length.first_dx.min()
ehr_length["enrollment_days"] = (ehr_length.first_dx - ref_date).dt.days

match_vars = dx_counts.merge(
    ehr_length[["person_id", "enrollment_days", "ehr_length_days"]], on="person_id"
)
save(match_vars, "06_matching_variables.csv")


# ═══════════════════════════════════════════════════════════════════
# STEP 6: REGRESSION BASE ASSEMBLY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6: Regression Base Assembly")
print("=" * 70)

reg = eligible[
    [
        "person_id",
        "severity",
        "ici_index_date",
        "baseline_cr",
        "max_cr_ratio",
        "max_delta_cr",
        "aki_delta03",
        "aki_kdigo2",
        "aki_kdigo3",
        "aki_180d",
    ]
].copy()
reg = reg.merge(
    demo[
        ["person_id", "sex_at_birth", "race", "ethnicity", "age_group", "baseline_egfr"]
    ],
    on="person_id",
)
reg = reg.merge(charlson, on="person_id")
reg = reg.merge(covariates, on="person_id")
reg = reg.merge(match_vars, on="person_id")

print(f"  Regression base: {len(reg):,} rows, {reg.shape[1]} cols")
print(f"  Cases: {reg.severity.sum():,}  Controls: {(reg.severity==0).sum():,}")
print(f"  Charlson score: median {reg.charlson_score.median():.0f}")
print(f"  NCI index: median {reg.nci_index.median():.3f}")
print(
    f"  eGFR: median {reg.baseline_egfr.median():.1f}, "
    f"<60: {(reg.baseline_egfr < 60).sum()} ({(reg.baseline_egfr < 60).mean()*100:.1f}%)"
)

save(reg, "07_pre_matching_base.csv")


# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("INPC ETL COMPLETE (v3 — phenotype rigor)")
print("=" * 70)
print(f"  Output: {RESULTS}/")
print(f"  Next:   Rscript 01b_psm.R inpc")
print(f"          Rscript 02_models.R inpc")
print(f"\n  V3 PHENOTYPE RIGOR:")
print(f"    ✅ Baseline Cr: median [-365,-7] + fallback to last [-365,-1]")
print(f"    ✅ Pre-ICI AKI washout: Cr ≥1.5× in [-90,0] excluded")
print(f"    ✅ Cr plausibility: ≥0.1 mg/dL (was >0)")
print(f"    ✅ Nephrotoxins: ±90d of ICI (was lifetime)")
print(f"    ✅ Baseline eGFR: CKD-EPI 2021 race-free")
print(f"\n  V2 NCI-CCI FIX SUMMARY:")
print(f"    ✅ MI scoring: OR logic (max 1pt)")
print(f"    ✅ Paralysis/CVD: independent scoring")
print(f"    ✅ No hierarchy pre-processing (raw flags preserved)")
print(f"    ✅ Both Charlson (integer) and NCI Index (continuous) computed")
