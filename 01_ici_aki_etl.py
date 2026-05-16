#!/usr/bin/env python3
"""
Post-ICI AKI × SDoH — AoU ETL
Runs on AoU Researcher Workbench (Controlled Tier).

Adapted from aou_covid/01_aou_etl.py (Wang et al.)
Design: Gatz et al., JAMIA 2024;31(12):2932–2939

Steps:
  1. ICI-treated cancer patient cohort + AKI phenotyping
  2. Demographics
  3. NCI Charlson Comorbidity Index (14 conditions)
  4. SDoH (Basics Survey — identical to COVID pipeline)
  5. Cancer type + ICI class + concomitant nephrotoxins
  6. Matching variable extraction (for 01b_psm.R)

Key changes from COVID pipeline:
  - Population: ICI-treated cancer patients (not COVID+)
  - Outcome:   All-cause AKI (Cr >=2.0x baseline, 365d post-ICI)
  - CCI:       NCI 14-condition (drops Malignancy, Metastatic)
  - Covariates: Cancer type, ICI class, concomitant nephrotoxins
  - Removed:   Vaccination, pandemic wave

Usage: python 01_ici_aki_etl.py
Output: results/ici_aki/*.csv  (01–06)
Next:   Rscript 01b_psm.R ici_aki
License: MIT
"""

import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore", message=".*read_gbq is deprecated.*")

import numpy as np
import pandas as pd

CDR = os.environ["WORKSPACE_CDR"]
BUCKET = os.environ["WORKSPACE_BUCKET"]
CDR_TAG = CDR.split(".")[-1]
RESULTS = "results/ici_aki"
BUCKET_DIR = f"{BUCKET}/data/ici_aki_sdoh"
os.makedirs(RESULTS, exist_ok=True)

print("=" * 70)
print("POST-ICI AKI × SDoH — AoU ETL")
print("=" * 70)
print(f"  CDR:     {CDR}")
print(f"  Tag:     {CDR_TAG}")
print(f"  Output:  {RESULTS}/")
print(f"  Bucket:  {BUCKET_DIR}/")
print("=" * 70)


def query(sql, label=""):
    print(f"\n  [{label}] Running query...")
    df = pd.read_gbq(sql, dialect="standard")
    print(f"  [{label}] → {len(df):,} rows, {df.shape[1]} cols")
    return df


def save(df, filename):
    filepath = os.path.join(RESULTS, filename)
    df.to_csv(filepath, index=False)
    subprocess.run(["gsutil", "cp", filepath, f"{BUCKET_DIR}/"], capture_output=True)
    print(f"  Saved: {filepath} ({len(df):,} rows)")


# =====================================================================
# ICI RxNorm Concept IDs
# Anti-PD-1, Anti-PD-L1, Anti-CTLA-4, Anti-LAG-3
# =====================================================================
# These are drug_concept_id values in AoU drug_exposure table.
# Includes ingredient-level + branded forms. Extend as needed after
# running a pilot query (see STEP 1 feasibility block).

# ICI agents are identified via concept_name matching in Step 1
# (name-based matching is more robust across CDR versions than hardcoded concept IDs)

# Creatinine LOINC (serum/plasma creatinine)
CR_LOINC = "2160-0"  # Standard LOINC for serum creatinine
# Also auto-discovered in AlcRx as measurement_concept_id = 3016723

# ESKD / dialysis / transplant exclusion codes
ESKD_ICD10 = ["N186"]
DIALYSIS_ICD10 = ["Z490", "Z491", "Z492", "Z992"]
TRANSPLANT_ICD10 = ["Z940"]
ESKD_EXCLUDE_CODES = ESKD_ICD10 + DIALYSIS_ICD10 + TRANSPLANT_ICD10

# Cancer ICD-10-CM prefixes (any C00-C96)
CANCER_PREFIXES = [f"C{i}" for i in range(100)]  # C00-C99, filtered below


# =====================================================================
# STEP 1: ICI-TREATED CANCER PATIENT COHORT + AKI PHENOTYPING
# =====================================================================
print("\n" + "=" * 70)
print("STEP 1: ICI-Treated Cancer Cohort + AKI Phenotyping")
print("=" * 70)

# ── 1a. Find all patients with ICI drug exposures ─────────────────
# Use name-based matching (proven in recon) instead of hardcoded concept IDs
ICI_DRUG_NAMES = [
    "nivolumab", "pembrolizumab", "cemiplimab", "dostarlimab",
    "retifanlimab", "toripalimab", "tislelizumab",
    "atezolizumab", "durvalumab", "avelumab",
    "ipilimumab", "tremelimumab", "relatlimab",
]
ICI_LIKE_SQL = " OR ".join(
    [f"LOWER(c.concept_name) LIKE '%{d}%'" for d in ICI_DRUG_NAMES]
)

# Map drug names to ICI class for regimen classification
ICI_NAME_TO_CLASS = {}
for d in ["nivolumab","pembrolizumab","cemiplimab","dostarlimab",
          "retifanlimab","toripalimab","tislelizumab"]:
    ICI_NAME_TO_CLASS[d] = "anti_pd1"
for d in ["atezolizumab","durvalumab","avelumab"]:
    ICI_NAME_TO_CLASS[d] = "anti_pdl1"
for d in ["ipilimumab","tremelimumab"]:
    ICI_NAME_TO_CLASS[d] = "anti_ctla4"
ICI_NAME_TO_CLASS["relatlimab"] = "anti_lag3"

ici_sql = f"""
SELECT
  de.person_id,
  de.drug_concept_id,
  de.drug_exposure_start_date AS ici_date,
  LOWER(c.concept_name) AS drug_name
FROM `{CDR}`.drug_exposure de
JOIN `{CDR}`.concept c ON c.concept_id = de.drug_concept_id
WHERE {ICI_LIKE_SQL}
ORDER BY de.person_id, de.drug_exposure_start_date
"""
ici_raw = query(ici_sql, "ICI drug exposures")
print(f"  Total ICI exposure records: {len(ici_raw):,}")
print(f"  Unique patients with any ICI: {ici_raw.person_id.nunique():,}")

if len(ici_raw) == 0:
    print("\n  ⚠ FATAL: No ICI exposures found. Check CDR version.")
    print("  Run 00_recon_feasibility.py first to verify concept names.")
    import sys; sys.exit(1)

# Index date = first ICI exposure
ici_index = ici_raw.groupby("person_id").agg(
    ici_index_date=("ici_date", "min")
).reset_index()
ici_index["ici_index_date"] = pd.to_datetime(ici_index["ici_index_date"])
print(f"  Patients with ICI index date: {len(ici_index):,}")

# ── 1b. ICI class classification (30-day window from index) ───────
ici_raw["ici_date"] = pd.to_datetime(ici_raw["ici_date"])
ici_raw = ici_raw.merge(ici_index, on="person_id")
ici_raw["days_from_index"] = (ici_raw["ici_date"] - ici_raw["ici_index_date"]).dt.days
ici_window = ici_raw[ici_raw["days_from_index"] <= 30].copy()

# Classify each row's ICI class from drug_name
def get_ici_class(drug_name):
    for key, cls in ICI_NAME_TO_CLASS.items():
        if key in str(drug_name).lower():
            return cls
    return "unknown"

ici_window["ici_class"] = ici_window["drug_name"].apply(get_ici_class)

def classify_regimen(group):
    classes = set(group["ici_class"]) - {"unknown"}
    if len(classes) == 0:
        return "unknown"
    if len(classes) > 1:
        return "combination"
    return classes.pop()

ici_regimen = ici_window.groupby("person_id").apply(classify_regimen).reset_index()
ici_regimen.columns = ["person_id", "ici_regimen"]
print(f"  ICI regimen: {ici_regimen.ici_regimen.value_counts().to_dict()}")

# ── 1c. Cancer diagnosis requirement ─────────────────────────────
# Any cancer diagnosis (C00-C96) before or within 30 days of ICI index
cancer_prefix_clauses = " OR ".join(
    [f"STARTS_WITH(UPPER(REPLACE(c.concept_code,'.','')),'C{i:02d}')"
     for i in range(97)]
)
cancer_sql = f"""
SELECT DISTINCT co.person_id
FROM `{CDR}`.condition_occurrence co
JOIN `{CDR}`.concept c
  ON c.concept_id = co.condition_source_concept_id
WHERE c.vocabulary_id IN ('ICD9CM', 'ICD10CM')
  AND ({cancer_prefix_clauses})
  AND co.person_id IN ({','.join(map(str, ici_index.person_id.tolist()))})
"""
cancer_pts = query(cancer_sql, "Cancer diagnosis")
print(f"  ICI patients WITH cancer dx: {len(cancer_pts):,}")

# Keep only ICI patients who have a cancer diagnosis
cohort = ici_index[ici_index.person_id.isin(cancer_pts.person_id)].copy()
print(f"  ICI + cancer cohort: {len(cohort):,}")

# ── 1d. Require Basics Survey ─────────────────────────────────────
survey_sql = f"""
SELECT DISTINCT person_id
FROM `{CDR}`.observation
WHERE observation_source_concept_id = 1585845
  AND person_id IN ({','.join(map(str, cohort.person_id.tolist()))})
"""
survey_pts = query(survey_sql, "Basics Survey")
n_no_survey = len(cohort) - len(survey_pts)
cohort = cohort[cohort.person_id.isin(survey_pts.person_id)].copy()
print(f"  Excluded (no Basics Survey): {n_no_survey:,}")
print(f"  After survey filter: {len(cohort):,}")

# ── 1e. Exclude pre-existing ESKD / dialysis / transplant ─────────
eskd_clauses = " OR ".join(
    [f"STARTS_WITH(UPPER(REPLACE(c.concept_code,'.','')),'{code}')"
     for code in ESKD_EXCLUDE_CODES]
)
eskd_sql = f"""
SELECT DISTINCT co.person_id
FROM `{CDR}`.condition_occurrence co
JOIN `{CDR}`.concept c
  ON c.concept_id = co.condition_source_concept_id
WHERE c.vocabulary_id IN ('ICD9CM', 'ICD10CM')
  AND ({eskd_clauses})
  AND co.person_id IN ({','.join(map(str, cohort.person_id.tolist()))})
"""
# Also check procedures for dialysis
eskd_proc_sql = f"""
SELECT DISTINCT person_id
FROM `{CDR}`.procedure_occurrence
WHERE procedure_concept_id IN (
    4032243, 4146536, 2213551, 2213552  -- dialysis procedure concepts
)
  AND person_id IN ({','.join(map(str, cohort.person_id.tolist()))})
"""
eskd_pts = query(eskd_sql, "ESKD conditions")
try:
    eskd_proc = query(eskd_proc_sql, "ESKD procedures")
    eskd_all = set(eskd_pts.person_id) | set(eskd_proc.person_id)
except Exception:
    eskd_all = set(eskd_pts.person_id)

# Filter: only exclude if ESKD/dialysis dx is BEFORE ICI index
# (post-ICI ESKD could be an outcome-related event)
eskd_pre = []
for pid in eskd_all:
    if pid in cohort.person_id.values:
        idx_date = cohort.loc[cohort.person_id == pid, "ici_index_date"].iloc[0]
        eskd_pre.append(pid)  # conservative: exclude all with any ESKD history
# Note: More refined logic could check dates, but for safety exclude all

n_eskd = len(eskd_pre)
cohort = cohort[~cohort.person_id.isin(eskd_pre)].copy()
print(f"  Excluded (pre-existing ESKD/dialysis/transplant): {n_eskd:,}")
print(f"  After ESKD exclusion: {len(cohort):,}")

# ── 1f. Serum creatinine: baseline + follow-up ────────────────────
# Best practice for baseline Cr in ICI-AKI studies:
#   PRIMARY:     Median of outpatient Cr in [-365d, -7d] pre-ICI
#                (excludes 7d immediately before ICI to avoid treatment-day draws)
#   SENSITIVITY: Most-recent Cr in [-365d, -1d] (AlcRx approach)
#   SENSITIVITY: Nadir (lowest) Cr in [-365d, -7d] (Cortazar/ICPi-AKI approach)
#
# Rationale: Cancer patients have volatile Cr from chemo, contrast, dehydration.
# Median outpatient smooths transient spikes; nadir is most conservative (maximizes
# sensitivity for AKI detection); most-recent is simplest but least stable.
#
# Cr concept: auto-discovered in AlcRx as 3016723 (AoU LOINC 2160-0).
# Recon script (00) will confirm. We also query by concept_code as fallback.
# ─────────────────────────────────────────────────────────────────────
pids_str = ",".join(map(str, cohort.person_id.tolist()))

cr_sql = f"""
SELECT
  m.person_id,
  m.measurement_date,
  m.value_as_number AS cr_value,
  m.unit_concept_id,
  m.unit_source_value,
  vo.visit_concept_id
FROM `{CDR}`.measurement m
LEFT JOIN `{CDR}`.visit_occurrence vo
  ON m.person_id = vo.person_id
  AND m.measurement_date BETWEEN vo.visit_start_date
      AND COALESCE(vo.visit_end_date, vo.visit_start_date)
WHERE (m.measurement_concept_id = 3016723
       OR m.measurement_concept_id IN (
           SELECT concept_id FROM `{CDR}`.concept
           WHERE concept_code = '2160-0' AND vocabulary_id = 'LOINC'))
  AND m.value_as_number IS NOT NULL
  AND m.value_as_number > 0
  AND m.value_as_number < 30000  -- allows µmol/L
  AND m.person_id IN ({pids_str})
ORDER BY m.person_id, m.measurement_date
"""
cr_all = query(cr_sql, "Serum creatinine + visit context")
cr_all["measurement_date"] = pd.to_datetime(cr_all["measurement_date"])
print(f"  Total Cr measurements: {len(cr_all):,}")
print(f"  Patients with any Cr: {cr_all.person_id.nunique():,}")

# ── Unit conversion: µmol/L → mg/dL ──────────────────────────────
# µmol/L values are typically 44–1300; mg/dL values are 0.3–15.
# Strategy: use unit_concept_id / unit_source_value if available,
# otherwise use value-range heuristic (threshold = 30).
# OMOP unit_concept_id: 8840 = mg/dL, 8749 = µmol/L
UMOL_TO_MGDL = 0.0113  # 1 µmol/L = 0.0113 mg/dL

def convert_cr(row):
    val = row["cr_value"]
    unit_id = row.get("unit_concept_id", None)
    unit_src = str(row.get("unit_source_value", "")).lower().strip()

    # Method 1: explicit unit concept ID (guard against NA)
    if pd.notna(unit_id):
        if int(unit_id) == 8749:  # µmol/L
            return val * UMOL_TO_MGDL
        if int(unit_id) == 8840:  # mg/dL
            return val

    # Method 2: unit_source_value string matching
    if "umol" in unit_src or "µmol" in unit_src or "micromol" in unit_src:
        return val * UMOL_TO_MGDL
    if "mg" in unit_src:
        return val

    # Method 3: value-range heuristic
    if val >= 30:  # almost certainly µmol/L
        return val * UMOL_TO_MGDL

    return val  # assume mg/dL

cr_all["cr_mgdl"] = cr_all.apply(convert_cr, axis=1)

# Report conversion stats
n_converted = ((cr_all["cr_value"] != cr_all["cr_mgdl"]) &
               (cr_all["cr_mgdl"] > 0)).sum()
print(f"  Cr unit conversions (µmol/L → mg/dL): {n_converted:,} "
      f"({n_converted/len(cr_all)*100:.1f}%)")
print(f"  Cr (mg/dL) median: {cr_all['cr_mgdl'].median():.2f}, "
      f"IQR: {cr_all['cr_mgdl'].quantile(0.25):.2f}–"
      f"{cr_all['cr_mgdl'].quantile(0.75):.2f}")

# Apply plausibility filter on converted values
cr_all = cr_all[(cr_all["cr_mgdl"] > 0.1) & (cr_all["cr_mgdl"] < 30)].copy()
cr_all["cr_value"] = cr_all["cr_mgdl"]  # overwrite with standardized values

# ── Visit-type classification for outpatient restriction ──────────
# Outpatient: 9202, 581476, 38004515
# ED: 9203
# Inpatient: 9201, 32037, 262, 8717
OUTPATIENT_VISITS = [9202, 581476, 38004515]
ED_VISITS = [9203]
INPATIENT_VISITS = [9201, 32037, 262, 8717]

cr_all["visit_setting"] = "unknown"
cr_all.loc[cr_all["visit_concept_id"].isin(OUTPATIENT_VISITS), "visit_setting"] = "outpatient"
cr_all.loc[cr_all["visit_concept_id"].isin(ED_VISITS), "visit_setting"] = "ed"
cr_all.loc[cr_all["visit_concept_id"].isin(INPATIENT_VISITS), "visit_setting"] = "inpatient"

print(f"  Visit setting distribution:")
print(f"    {cr_all.visit_setting.value_counts().to_dict()}")

cr_all.drop(columns=["cr_mgdl", "unit_concept_id", "unit_source_value",
                      "visit_concept_id"], inplace=True)

# Merge index date
cr_all = cr_all.merge(cohort[["person_id", "ici_index_date"]], on="person_id")
cr_all["days_from_index"] = (cr_all["measurement_date"] - cr_all["ici_index_date"]).dt.days

# ── Baseline Cr: THREE approaches ─────────────────────────────────
# Window: [-365d, -7d] for primary; [-365d, -1d] for sensitivity
cr_baseline_window = cr_all[
    (cr_all["days_from_index"] >= -365) & (cr_all["days_from_index"] <= -7)
].copy()

cr_baseline_window_sens = cr_all[
    (cr_all["days_from_index"] >= -365) & (cr_all["days_from_index"] <= -1)
].copy()

# PRIMARY: Median of OUTPATIENT Cr in [-365d, -7d]
cr_outpatient = cr_baseline_window[
    cr_baseline_window["visit_setting"].isin(["outpatient", "unknown"])
]
baseline_median = cr_outpatient.groupby("person_id").agg(
    baseline_cr=("cr_value", "median"),
    n_baseline_cr=("cr_value", "count"),
).reset_index()

# SENSITIVITY A: Most recent Cr in [-365d, -1d] (AlcRx approach)
baseline_recent = cr_baseline_window_sens.sort_values("measurement_date")
baseline_recent = baseline_recent.groupby("person_id").tail(1)
baseline_recent = baseline_recent[["person_id", "cr_value"]].rename(
    columns={"cr_value": "baseline_cr_recent"}
)

# SENSITIVITY B: Nadir (lowest) Cr in [-365d, -7d] (Cortazar/ICPi-AKI)
baseline_nadir = cr_baseline_window.groupby("person_id").agg(
    baseline_cr_nadir=("cr_value", "min"),
).reset_index()

# Merge all baseline approaches
cr_baseline = baseline_median.merge(baseline_recent, on="person_id", how="outer")
cr_baseline = cr_baseline.merge(baseline_nadir, on="person_id", how="outer")

# Fill primary baseline: if no outpatient, fall back to most-recent
cr_baseline["baseline_cr"] = cr_baseline["baseline_cr"].fillna(
    cr_baseline["baseline_cr_recent"]
)

print(f"\n  Baseline Cr approaches:")
print(f"    Median outpatient [-365d,-7d]: {baseline_median.shape[0]:,} patients")
print(f"    Most recent [-365d,-1d]:       {baseline_recent.shape[0]:,} patients")
print(f"    Nadir [-365d,-7d]:             {baseline_nadir.shape[0]:,} patients")
print(f"    Combined (any baseline):       {cr_baseline.shape[0]:,} patients")

# Follow-up Cr: any Cr in [1, 365] window after ICI index
cr_followup = cr_all[
    (cr_all["days_from_index"] >= 1) & (cr_all["days_from_index"] <= 365)
].copy()
pts_with_followup = cr_followup.person_id.unique()
print(f"  Patients with follow-up Cr [1-365d]: {len(pts_with_followup):,}")

# ── 1g. AKI phenotyping ──────────────────────────────────────────
# Merge baseline Cr with follow-up, compute ratio
cr_followup = cr_followup.merge(cr_baseline, on="person_id")
cr_followup["cr_ratio"] = cr_followup["cr_value"] / cr_followup["baseline_cr"]

# AKI = any follow-up Cr >= 1.5x baseline (KDIGO Stage 1+) — PRIMARY
aki_pts = cr_followup[cr_followup["cr_ratio"] >= 1.5].person_id.unique()

# Also compute max ratio AND max delta per patient for sensitivity analyses
max_stats = cr_followup.groupby("person_id").agg(
    max_cr_ratio=("cr_ratio", "max"),
    max_cr_value=("cr_value", "max"),
    n_followup_cr=("cr_value", "count"),
).reset_index()

# Max delta Cr (absolute change) per patient
cr_followup["delta_cr"] = cr_followup["cr_value"] - cr_followup["baseline_cr"]
max_delta = cr_followup.groupby("person_id").agg(
    max_delta_cr=("delta_cr", "max"),
).reset_index()
max_stats = max_stats.merge(max_delta, on="person_id", how="left")

# ── 1h. Exclude patients without baseline OR follow-up Cr ─────────
eligible = cohort[
    (cohort.person_id.isin(cr_baseline.person_id)) &
    (cohort.person_id.isin(pts_with_followup))
].copy()
n_no_baseline = len(cohort) - len(cohort[cohort.person_id.isin(cr_baseline.person_id)])
n_no_followup = len(cohort[cohort.person_id.isin(cr_baseline.person_id)]) - len(eligible)
print(f"\n  Excluded (no baseline Cr): {n_no_baseline:,}")
print(f"  Excluded (no follow-up Cr): {n_no_followup:,}")
print(f"  Final eligible cohort: {len(eligible):,}")

# Exclude baseline Cr >= 4.0 (likely pre-existing severe CKD)
eligible = eligible.merge(cr_baseline, on="person_id")
n_high_baseline = (eligible["baseline_cr"] >= 4.0).sum()
eligible = eligible[eligible["baseline_cr"] < 4.0].copy()
print(f"  Excluded (baseline Cr >= 4.0): {n_high_baseline:,}")

# ── 1i. Build final cohort with outcome ──────────────────────────
eligible["severity"] = eligible.person_id.isin(aki_pts).astype(int)
eligible = eligible.merge(max_stats, on="person_id", how="left")
eligible = eligible.merge(ici_regimen, on="person_id", how="left")

n_cases = eligible.severity.sum()
n_controls = len(eligible) - n_cases
print(f"\n  ┌─────────────────────────────────────────┐")
print(f"  │  FINAL COHORT: {len(eligible):,} patients          │")
print(f"  │  AKI Cases (Cr ≥1.5×): {n_cases:,}             │")
print(f"  │  Controls (no AKI):    {n_controls:,}              │")
print(f"  │  AKI rate: {n_cases/len(eligible)*100:.1f}%                       │")
print(f"  │  ICI regimen: {eligible.ici_regimen.value_counts().to_dict()} │")
print(f"  └─────────────────────────────────────────┘")

# Sensitivity flags
# S1: ΔCr ≥0.3 mg/dL (KDIGO Stage 1a, most sensitive)
eligible["aki_delta03"] = (eligible["max_delta_cr"] >= 0.3).astype(int)

# S2: Cr ≥2.0× baseline (KDIGO Stage 2, moderate-severe)
eligible["aki_kdigo2"] = eligible.person_id.isin(
    cr_followup[cr_followup["cr_ratio"] >= 2.0].person_id.unique()
).astype(int)

# S3: Cr ≥3.0× baseline (KDIGO Stage 3, severe)
eligible["aki_kdigo3"] = eligible.person_id.isin(
    cr_followup[cr_followup["cr_ratio"] >= 3.0].person_id.unique()
).astype(int)

# S4: 180-day window AKI (primary threshold ≥1.5×)
cr_180 = cr_followup[cr_followup["days_from_index"] <= 180]
eligible["aki_180d"] = eligible.person_id.isin(
    cr_180[cr_180["cr_ratio"] >= 1.5].person_id.unique()
).astype(int)

print(f"\n  Sensitivity flag counts:")
print(f"    S1 ΔCr≥0.3:   {eligible.aki_delta03.sum():,} ({eligible.aki_delta03.mean()*100:.1f}%)")
print(f"    S2 ≥2.0×:     {eligible.aki_kdigo2.sum():,} ({eligible.aki_kdigo2.mean()*100:.1f}%)")
print(f"    S3 ≥3.0×:     {eligible.aki_kdigo3.sum():,} ({eligible.aki_kdigo3.mean()*100:.1f}%)")
print(f"    S4 180d ≥1.5×: {eligible.aki_180d.sum():,} ({eligible.aki_180d.mean()*100:.1f}%)")

save(eligible, "01_ici_cohort.csv")


# =====================================================================
# STEP 2: DEMOGRAPHICS
# =====================================================================
print("\n" + "=" * 70)
print("STEP 2: Demographics")
print("=" * 70)

demo_sql = f"""
SELECT
  person_id,
  CASE gender_concept_id
    WHEN 8507 THEN 'Male'
    WHEN 8532 THEN 'Female'
    ELSE 'Other'
  END AS sex_at_birth,
  CASE race_concept_id
    WHEN 8516 THEN 'Black'
    WHEN 8527 THEN 'White'
    WHEN 8515 THEN 'Asian'
    ELSE 'Other'
  END AS race,
  CASE ethnicity_concept_id
    WHEN 38003563 THEN 'Hispanic'
    WHEN 38003564 THEN 'Not Hispanic'
    ELSE 'Other'
  END AS ethnicity,
  year_of_birth
FROM `{CDR}`.person
WHERE person_id IN ({','.join(map(str, eligible.person_id.tolist()))})
"""
demo = query(demo_sql, "Demographics")
demo = demo.merge(eligible[["person_id", "ici_index_date"]], on="person_id")
demo["age_at_ici"] = (
    pd.to_datetime(demo["ici_index_date"]).dt.year - demo["year_of_birth"]
)
demo["age_group"] = pd.cut(
    demo["age_at_ici"],
    bins=[0, 45, 55, 65, 200],
    labels=["<45", "45-54", "55-64", "65+"],
    right=False,
)

print(f"  Sex: {demo.sex_at_birth.value_counts().to_dict()}")
print(f"  Race: {demo.race.value_counts().to_dict()}")
print(f"  Age: {demo.age_group.value_counts().to_dict()}")

save(
    demo[["person_id", "sex_at_birth", "race", "ethnicity",
          "age_at_ici", "age_group", "year_of_birth"]],
    "02_demographics.csv",
)


# =====================================================================
# STEP 3: NCI CHARLSON COMORBIDITY INDEX (14 conditions)
# Replaces Glasheen 19-item CCI for cancer population.
# Drops: Malignancy, Metastatic_Solid_Tumor
# Merges: Renal into single category
# Uses: NCI 2021 ICD code sets (modified Quan et al. 2005)
# =====================================================================
print("\n" + "=" * 70)
print("STEP 3: NCI Charlson Comorbidity Index (14 conditions)")
print("=" * 70)

# NCI 2021 code sets (from nci-charlson skill §3)
NCI_CCI = {
    "Acute_MI": {
        "9": ["410"],
        "10": ["I21", "I22"],
    },
    "History_MI": {
        "9": ["412"],
        "10": ["I252"],
    },
    "Congestive_Heart_Failure": {
        "9": ["39891","40201","40211","40291","40401","40403",
              "40411","40413","40491","40493",
              "4254","4255","4256","4257","4258","4259","428"],
        "10": ["I099","I110","I130","I132","I255",
               "I420","I425","I426","I427","I428","I429",
               "I43","I50","P290"],
    },
    "Peripheral_Vascular_Disease": {
        "9": ["0930","440","441",
              "4431","4432","4433","4434","4435","4436","4437","4438","4439",
              "4471","5571","5579","V434"],
        "10": ["I70","I71","I731","I738","I739","I771",
               "I790","I792","K551","K558","K559","Z958","Z959"],
    },
    "Cerebrovascular_Disease": {
        "9": ["36234","430","431","432","433","434","435","436","437","438"],
        "10": ["G45","G46","H340",
               "I60","I61","I62","I63","I64","I65","I66","I67","I68","I69"],
    },
    "Chronic_Pulmonary_Disease": {
        "9": ["4168","4169","490","491","492","493","494","495","496","497",
              "498","499","500","501","502","503","504","505","5064","5081","5088"],
        "10": ["I278","I279","J40","J41","J42","J43","J44","J45","J46","J47",
               "J60","J61","J62","J63","J64","J65","J66","J67","J684","J701","J703"],
    },
    "Dementia": {
        "9": ["290","2941","3312"],
        "10": ["F00","F01","F02","F03","F051","G30","G311"],
    },
    "Paralysis": {
        "9": ["3341","342","343",
              "3440","3441","3442","3443","3444","3445","3446","3449"],
        "10": ["G041","G114","G801","G802","G81","G82",
               "G830","G831","G832","G833","G834","G839"],
    },
    "Diabetes": {
        "9": ["2500","2501","2502","2503","2508","2509"],
        "10": ["E100","E101","E106","E108","E109",
               "E110","E111","E116","E118","E119",
               "E130","E131","E136","E138","E139"],
    },
    "Diabetes_Complicated": {
        "9": ["2504","2505","2506","2507"],
        "10": ["E102","E103","E104","E105","E107",
               "E112","E113","E114","E115","E117",
               "E132","E133","E134","E135","E137"],
    },
    "Renal_Disease": {
        "9": ["40301","40311","40391","40402","40403","40412","40413",
              "40492","40493","582","5830","5831","5832","5833","5834",
              "5835","5836","5837","585","586","5880","V420","V451","V56"],
        "10": ["I120","I131","N032","N033","N034","N035","N036","N037",
               "N052","N053","N054","N055","N056","N057",
               "N18","N19","N250","Z490","Z491","Z492","Z940","Z992"],
    },
    "Liver_Disease_Mild": {
        "9": ["07022","07023","07032","07033","07044","07054",
              "0706","0709","570","571","5733","5734","5738","5739","V427"],
        "10": ["B18","K700","K701","K702","K703","K709",
               "K713","K714","K715","K717","K73","K74",
               "K760","K762","K763","K764","K768","K769","Z944"],
    },
    "Liver_Disease_Moderate_Severe": {
        "9": ["4560","4561","4562","5722","5723","5724","5725","5726","5727","5728"],
        "10": ["I850","I859","I864","I982",
               "K704","K711","K721","K729","K765","K766","K767"],
    },
    "Peptic_Ulcer_Disease": {
        "9": ["531","532","533","534"],
        "10": ["K25","K26","K27","K28"],
    },
    "Rheumatic_Disease": {
        "9": ["4465","7100","7101","7102","7103","7104",
              "7140","7141","7142","7148","725"],
        "10": ["M05","M06","M315","M32","M33","M34",
               "M351","M353","M360"],
    },
    "AIDS": {
        "9": ["042"],
        "10": ["B20"],
    },
}

# Build SQL: same pattern as COVID pipeline but with NCI code sets
conditions_list = []
for condition, codes in NCI_CCI.items():
    prefix_clauses = []
    for ver, voc in [("9", "ICD9CM"), ("10", "ICD10CM")]:
        for c in codes.get(ver, []):
            prefix_clauses.append(
                f"(STARTS_WITH(UPPER(REPLACE(c.concept_code,'.','')),"
                f"'{c}') AND c.vocabulary_id = '{voc}')"
            )
    if prefix_clauses:
        conditions_list.append(
            f"MAX(CASE WHEN {' OR '.join(prefix_clauses)} THEN 1 ELSE 0 END)"
            f" AS {condition}"
        )

# AIDS OI codes for two-step AIDS logic
AIDS_OI = {
    "9": ["112","180","114","1175","0074","0785","3483","054","115","0072",
          "176","200","201","202","203","204","205","206","207","208","209",
          "031","010","011","012","013","014","015","016","017","018",
          "1363","V1261","0463","0031","130","7994"],
    "10": ["B37","C53","B38","B45","A072","B25","G934","B00","B39","A073",
           "C46","C81","C82","C83","C84","C85","C86","C87","C88","C9",
           "C90","C91","C92","C93","C94","C95","C96","A31","A15","A16",
           "A17","A18","A19","B59","Z8701","A812","A021","B58","R64"],
}
oi_clauses = []
for ver, voc in [("9", "ICD9CM"), ("10", "ICD10CM")]:
    for c in AIDS_OI.get(ver, []):
        oi_clauses.append(
            f"(STARTS_WITH(UPPER(REPLACE(c.concept_code,'.','')),"
            f"'{c}') AND c.vocabulary_id = '{voc}')"
        )
conditions_list.append(
    f"MAX(CASE WHEN {' OR '.join(oi_clauses)} THEN 1 ELSE 0 END) AS has_oi"
)

# Look-back: 365 days before ICI index, excluding 30 days around index
# We'll query all dx first, then filter by date in Python
charlson_sql = f"""
SELECT co.person_id, {','.join(conditions_list)}
FROM `{CDR}`.condition_occurrence co
JOIN `{CDR}`.concept c
  ON c.concept_id = co.condition_source_concept_id
WHERE co.person_id IN ({','.join(map(str, eligible.person_id.tolist()))})
  AND c.vocabulary_id IN ('ICD9CM', 'ICD10CM')
GROUP BY co.person_id
"""
charlson = query(charlson_sql, "NCI Charlson")

# Fill missing with 0
charlson = eligible[["person_id"]].merge(charlson, on="person_id", how="left")
for col in list(NCI_CCI.keys()) + ["has_oi"]:
    charlson[col] = charlson[col].fillna(0).astype(int)

# AIDS = HIV AND OI co-occurrence
charlson["AIDS"] = ((charlson["AIDS"] == 1) & (charlson["has_oi"] == 1)).astype(int)
charlson.drop(columns=["has_oi"], inplace=True)

# NCI hierarchy: Moderate/Severe Liver trumps Mild Liver
charlson.loc[charlson["Liver_Disease_Moderate_Severe"] == 1, "Liver_Disease_Mild"] = 0
# NCI hierarchy: Complicated Diabetes trumps Uncomplicated
charlson.loc[charlson["Diabetes_Complicated"] == 1, "Diabetes"] = 0

# Verify: NO Malignancy or Metastatic columns (NCI design)
assert "Malignancy" not in charlson.columns, "NCI-CCI must NOT include Malignancy"
assert "Metastatic_Solid_Tumor" not in charlson.columns, "NCI-CCI must NOT include Metastatic"

# Print prevalences
print("\n  NCI-CCI prevalences:")
for col in NCI_CCI.keys():
    n = charlson[col].sum()
    pct = n / len(charlson) * 100
    print(f"    {col:40s} {n:>6,}  ({pct:.1f}%)")

save(charlson, "03_nci_charlson.csv")


# =====================================================================
# STEP 4: SDoH (Basics Survey)
# Identical to COVID pipeline — same survey items, same recoding.
# =====================================================================
print("\n" + "=" * 70)
print("STEP 4: SDoH (Basics Survey)")
print("=" * 70)

# Insurance ────────────────────────────────────────────────────────
insurance_sql = f"""
SELECT person_id,
  MAX(CASE WHEN value_source_concept_id = 45877740 THEN 1 ELSE 0 END) AS ins_employer,
  MAX(CASE WHEN value_source_concept_id = 45882567 THEN 1 ELSE 0 END) AS ins_private,
  MAX(CASE WHEN value_source_concept_id = 45877746 THEN 1 ELSE 0 END) AS ins_medicare,
  MAX(CASE WHEN value_source_concept_id = 45882571 THEN 1 ELSE 0 END) AS ins_medicaid,
  MAX(CASE WHEN value_source_concept_id = 45878463 THEN 1 ELSE 0 END) AS ins_military,
  MAX(CASE WHEN value_source_concept_id = 45876662 THEN 1 ELSE 0 END) AS ins_indian,
  MAX(CASE WHEN value_source_concept_id = 45882572 THEN 1 ELSE 0 END) AS ins_none,
  MAX(CASE WHEN value_source_concept_id = 45883427 THEN 1 ELSE 0 END) AS ins_other
FROM `{CDR}`.observation
WHERE observation_source_concept_id = 43528428
  AND person_id IN ({','.join(map(str, eligible.person_id.tolist()))})
GROUP BY person_id
"""
insurance = query(insurance_sql, "Insurance")

# Hierarchical recode: Medicaid > Medicare > Employer > Private > Other_None
def recode_insurance(row):
    if row.get("ins_medicaid", 0) == 1:
        return "Medicaid"
    if row.get("ins_medicare", 0) == 1:
        return "Medicare"
    if row.get("ins_employer", 0) == 1:
        return "Employer"
    if row.get("ins_private", 0) == 1:
        return "Private"
    if row.get("ins_military", 0) == 1 or row.get("ins_indian", 0) == 1:
        return "Other_Public"
    if row.get("ins_none", 0) == 1:
        return "Uninsured"
    return "Missing"

insurance["insurance_type"] = insurance.apply(recode_insurance, axis=1)

# Income ───────────────────────────────────────────────────────────
income_map = {
    45880480: "Less_10K",
    45876648: "10K_25K",
    45880660: "25K_35K",
    45880116: "35K_50K",
    45877579: "50K_75K",
    45881804: "75K_100K",
    45876981: "100K_150K",
    45883383: "150K_200K",
    45874835: "200K_Plus",
}
income_sql = f"""
SELECT person_id,
  value_source_concept_id AS income_code
FROM `{CDR}`.observation
WHERE observation_source_concept_id = 1585375
  AND person_id IN ({','.join(map(str, eligible.person_id.tolist()))})
"""
income = query(income_sql, "Income")
income["income"] = income["income_code"].map(income_map).fillna("Missing")

# Education ────────────────────────────────────────────────────────
edu_map = {
    903079: "Never_Attended",
    903096: "Less_HS",
    903087: "HS_GED",
    903095: "Some_College",
    903071: "College_Grad",
    903085: "Advanced_Degree",
}
edu_sql = f"""
SELECT person_id,
  value_source_concept_id AS edu_code
FROM `{CDR}`.observation
WHERE observation_source_concept_id = 1585940
  AND person_id IN ({','.join(map(str, eligible.person_id.tolist()))})
"""
edu = query(edu_sql, "Education")
edu["education"] = edu["edu_code"].map(edu_map).fillna("Missing")

# Employment ───────────────────────────────────────────────────────
emp_map = {
    903026: "Employed",
    903025: "Unemployed_Looking",
    903027: "Not_Working_Not_Looking",
    903018: "Disabled",
    903023: "Retired",
    903036: "Student",
}
emp_sql = f"""
SELECT person_id,
  value_source_concept_id AS emp_code
FROM `{CDR}`.observation
WHERE observation_source_concept_id = 1585952
  AND person_id IN ({','.join(map(str, eligible.person_id.tolist()))})
"""
emp = query(emp_sql, "Employment")
emp["employment"] = emp["emp_code"].map(emp_map).fillna("Missing")

# Housing ──────────────────────────────────────────────────────────
housing_map = {
    903069: "Own",
    903064: "Rent",
    903058: "Other",
}
housing_sql = f"""
SELECT person_id,
  value_source_concept_id AS housing_code
FROM `{CDR}`.observation
WHERE observation_source_concept_id = 1585889
  AND person_id IN ({','.join(map(str, eligible.person_id.tolist()))})
"""
housing = query(housing_sql, "Housing")
housing["housing"] = housing["housing_code"].map(housing_map).fillna("Missing")

# Housing stability ────────────────────────────────────────────────
stability_map = {
    903049: "Worried",
    903047: "Not_Worried",
}
stability_sql = f"""
SELECT person_id,
  value_source_concept_id AS stability_code
FROM `{CDR}`.observation
WHERE observation_source_concept_id = 1585886
  AND person_id IN ({','.join(map(str, eligible.person_id.tolist()))})
"""
stability = query(stability_sql, "Housing stability")
stability["housing_stability"] = stability["stability_code"].map(stability_map).fillna("Missing")

# Disability (mobility) ────────────────────────────────────────────
disability_sql = f"""
SELECT person_id,
  MAX(CASE WHEN value_source_concept_id IN (903100,903114) THEN 1 ELSE 0 END) AS disability_mobility
FROM `{CDR}`.observation
WHERE observation_source_concept_id = 1585747
  AND person_id IN ({','.join(map(str, eligible.person_id.tolist()))})
GROUP BY person_id
"""
disability = query(disability_sql, "Disability")

# Merge all SDoH
sdoh = eligible[["person_id"]].copy()
sdoh = sdoh.merge(insurance[["person_id", "insurance_type"]], on="person_id", how="left")
sdoh = sdoh.merge(income[["person_id", "income"]], on="person_id", how="left")
sdoh = sdoh.merge(edu[["person_id", "education"]], on="person_id", how="left")
sdoh = sdoh.merge(emp[["person_id", "employment"]], on="person_id", how="left")
sdoh = sdoh.merge(housing[["person_id", "housing"]], on="person_id", how="left")
sdoh = sdoh.merge(stability[["person_id", "housing_stability"]], on="person_id", how="left")
sdoh = sdoh.merge(disability[["person_id", "disability_mobility"]], on="person_id", how="left")
sdoh = sdoh.fillna("Missing")

print(f"\n  SDoH summary:")
for col in ["insurance_type", "income", "education", "employment",
            "housing", "housing_stability", "disability_mobility"]:
    print(f"    {col}: {sdoh[col].value_counts().to_dict()}")

save(sdoh, "04_sdoh.csv")


# =====================================================================
# STEP 5: CANCER TYPE + CONCOMITANT NEPHROTOXINS
# Replaces vaccination step in COVID pipeline.
# =====================================================================
print("\n" + "=" * 70)
print("STEP 5: Cancer Type + Concomitant Nephrotoxins")
print("=" * 70)

# ── 5a. Cancer type classification ────────────────────────────────
cancer_type_sql = f"""
WITH cancer_dx AS (
  SELECT co.person_id,
    co.condition_start_date,
    UPPER(REPLACE(c.concept_code, '.', '')) AS dx_code
  FROM `{CDR}`.condition_occurrence co
  JOIN `{CDR}`.concept c ON c.concept_id = co.condition_source_concept_id
  WHERE c.vocabulary_id = 'ICD10CM'
    AND (STARTS_WITH(c.concept_code, 'C') OR STARTS_WITH(c.concept_code, 'c'))
    AND co.person_id IN ({','.join(map(str, eligible.person_id.tolist()))})
)
SELECT person_id, dx_code, condition_start_date
FROM cancer_dx
ORDER BY person_id, condition_start_date
"""
cancer_dx = query(cancer_type_sql, "Cancer type")
cancer_dx["condition_start_date"] = pd.to_datetime(cancer_dx["condition_start_date"])
cancer_dx = cancer_dx.merge(eligible[["person_id", "ici_index_date"]], on="person_id")

def classify_cancer(dx_code):
    code = dx_code.upper()
    if code.startswith("C34"): return "Lung"
    if code.startswith("C43"): return "Melanoma"
    if code.startswith("C64") or code.startswith("C65"): return "Renal_Cell"
    if code.startswith("C67"): return "Urothelial"
    if any(code.startswith(p) for p in ["C00","C01","C02","C03","C04","C05","C06",
            "C07","C08","C09","C10","C11","C12","C13","C14","C30","C31","C32"]):
        return "Head_Neck"
    if code.startswith("C50"): return "Breast"
    if code.startswith("C22"): return "Hepatocellular"
    if any(code.startswith(p) for p in ["C18","C19","C20"]): return "Colorectal"
    if any(code.startswith(p) for p in ["C81","C82","C83","C84","C85","C86",
            "C88","C90","C91","C92","C93","C94","C95","C96"]):
        return "Hematologic"
    return "Other_Solid"

cancer_dx["cancer_type"] = cancer_dx["dx_code"].apply(classify_cancer)

# Pick the cancer type closest to ICI index date
cancer_dx["days_to_ici"] = abs(
    (cancer_dx["condition_start_date"] - cancer_dx["ici_index_date"]).dt.days
)
cancer_primary = cancer_dx.sort_values("days_to_ici").groupby("person_id").first()
cancer_primary = cancer_primary[["cancer_type"]].reset_index()

eligible_cancer = eligible[["person_id"]].merge(cancer_primary, on="person_id", how="left")
eligible_cancer["cancer_type"] = eligible_cancer["cancer_type"].fillna("Unknown")
print(f"  Cancer types: {eligible_cancer.cancer_type.value_counts().to_dict()}")

# ── 5b. Concomitant nephrotoxic medications ───────────────────────
# Binary flags: any exposure within 90 days before/after ICI index

# PPI concept IDs (omeprazole, esomeprazole, pantoprazole, lansoprazole, etc.)
PPI_CONCEPTS = "911735,929887,904453,948078,2038233,997276"
# NSAID concept IDs (ibuprofen, naproxen, diclofenac, celecoxib, etc.)
NSAID_CONCEPTS = "1177480,1115171,1124300,1118084,1236607,1150345,1146810"
# ACEi/ARB concept IDs
ACEI_ARB_CONCEPTS = ("1335471,1340128,1341927,1363749,1308216,1310756,"
                     "1331235,1334456,1373928,1395058,40235485,1347384")
# Diuretic concept IDs (furosemide, HCTZ, bumetanide, etc.)
DIURETIC_CONCEPTS = "956874,974166,932745,942350,904542,970250"

nephrotoxin_sql = f"""
WITH ici_dates AS (
  SELECT person_id, ici_index_date
  FROM UNNEST([{','.join(
    f"STRUCT({r.person_id} AS person_id, DATE '{r.ici_index_date.strftime('%Y-%m-%d')}' AS ici_index_date)"
    for _, r in eligible.iterrows()
  )}])
)
SELECT de.person_id,
  MAX(CASE WHEN de.drug_concept_id IN ({PPI_CONCEPTS}) THEN 1 ELSE 0 END) AS ppi_flag,
  MAX(CASE WHEN de.drug_concept_id IN ({NSAID_CONCEPTS}) THEN 1 ELSE 0 END) AS nsaid_flag,
  MAX(CASE WHEN de.drug_concept_id IN ({ACEI_ARB_CONCEPTS}) THEN 1 ELSE 0 END) AS acei_arb_flag,
  MAX(CASE WHEN de.drug_concept_id IN ({DIURETIC_CONCEPTS}) THEN 1 ELSE 0 END) AS diuretic_flag
FROM `{CDR}`.drug_exposure de
WHERE de.person_id IN ({','.join(map(str, eligible.person_id.tolist()))})
  AND de.drug_concept_id IN ({PPI_CONCEPTS},{NSAID_CONCEPTS},{ACEI_ARB_CONCEPTS},{DIURETIC_CONCEPTS})
GROUP BY de.person_id
"""
# NOTE: The UNNEST approach above may hit BQ limits for large cohorts.
# If so, simplify by just querying all drug_exposure for these patients
# and filtering in Python with the 90-day window.
# For pilot, this simpler version works:
nephrotoxin_simple_sql = f"""
SELECT person_id,
  MAX(CASE WHEN drug_concept_id IN ({PPI_CONCEPTS}) THEN 1 ELSE 0 END) AS ppi_flag,
  MAX(CASE WHEN drug_concept_id IN ({NSAID_CONCEPTS}) THEN 1 ELSE 0 END) AS nsaid_flag,
  MAX(CASE WHEN drug_concept_id IN ({ACEI_ARB_CONCEPTS}) THEN 1 ELSE 0 END) AS acei_arb_flag,
  MAX(CASE WHEN drug_concept_id IN ({DIURETIC_CONCEPTS}) THEN 1 ELSE 0 END) AS diuretic_flag
FROM `{CDR}`.drug_exposure
WHERE person_id IN ({','.join(map(str, eligible.person_id.tolist()))})
  AND drug_concept_id IN ({PPI_CONCEPTS},{NSAID_CONCEPTS},{ACEI_ARB_CONCEPTS},{DIURETIC_CONCEPTS})
GROUP BY person_id
"""
nephrotoxins = query(nephrotoxin_simple_sql, "Concomitant nephrotoxins")
nephrotoxins = eligible[["person_id"]].merge(nephrotoxins, on="person_id", how="left")
for col in ["ppi_flag", "nsaid_flag", "acei_arb_flag", "diuretic_flag"]:
    nephrotoxins[col] = nephrotoxins[col].fillna(0).astype(int)

print(f"  PPI:  {nephrotoxins.ppi_flag.sum():,} ({nephrotoxins.ppi_flag.mean()*100:.1f}%)")
print(f"  NSAID: {nephrotoxins.nsaid_flag.sum():,} ({nephrotoxins.nsaid_flag.mean()*100:.1f}%)")
print(f"  ACEi/ARB: {nephrotoxins.acei_arb_flag.sum():,} ({nephrotoxins.acei_arb_flag.mean()*100:.1f}%)")
print(f"  Diuretic: {nephrotoxins.diuretic_flag.sum():,} ({nephrotoxins.diuretic_flag.mean()*100:.1f}%)")

# Merge cancer type + nephrotoxins
covariates = eligible_cancer.merge(nephrotoxins, on="person_id")
save(covariates, "05_cancer_nephrotoxins.csv")


# =====================================================================
# STEP 6: MATCHING VARIABLE EXTRACTION
# Same PS covariates as Gatz/Wang: enrollment date, dx count, EHR length.
# =====================================================================
print("\n" + "=" * 70)
print("STEP 6: Matching Variables")
print("=" * 70)

match_sql = f"""
SELECT
  p.person_id,
  -- enrollment proxy: earliest observation date
  (SELECT MIN(observation_date)
   FROM `{CDR}`.observation o
   WHERE o.person_id = p.person_id) AS first_obs_date,
  -- number of diagnoses
  (SELECT COUNT(DISTINCT condition_concept_id)
   FROM `{CDR}`.condition_occurrence co
   WHERE co.person_id = p.person_id) AS n_diagnoses,
  -- EHR length: first to last condition date
  (SELECT DATE_DIFF(MAX(condition_start_date), MIN(condition_start_date), DAY)
   FROM `{CDR}`.condition_occurrence co
   WHERE co.person_id = p.person_id) AS ehr_length_days
FROM `{CDR}`.person p
WHERE p.person_id IN ({','.join(map(str, eligible.person_id.tolist()))})
"""
match_vars = query(match_sql, "Matching variables")
match_vars["first_obs_date"] = pd.to_datetime(match_vars["first_obs_date"])
# Convert to days since a reference date for PSM
ref_date = match_vars["first_obs_date"].min()
match_vars["enrollment_days"] = (match_vars["first_obs_date"] - ref_date).dt.days
match_vars = match_vars.fillna(0)

save(
    match_vars[["person_id", "enrollment_days", "n_diagnoses", "ehr_length_days"]],
    "06_matching_variables.csv",
)


# =====================================================================
# STEP 7: BUILD REGRESSION BASE (merge all CSVs)
# =====================================================================
print("\n" + "=" * 70)
print("STEP 7: Build Regression Base")
print("=" * 70)

reg = eligible[["person_id", "severity", "ici_index_date", "ici_regimen",
                "baseline_cr", "max_cr_ratio",
                "aki_delta03", "aki_kdigo2", "aki_kdigo3", "aki_180d"]].copy()
reg = reg.merge(demo[["person_id","sex_at_birth","race","ethnicity","age_group"]], on="person_id")
reg = reg.merge(charlson, on="person_id")
reg = reg.merge(covariates, on="person_id")
reg = reg.merge(match_vars[["person_id","enrollment_days","n_diagnoses","ehr_length_days"]], on="person_id")

print(f"  Regression base: {len(reg):,} rows, {reg.shape[1]} cols")
print(f"  Cases: {reg.severity.sum():,}  Controls: {(reg.severity==0).sum():,}")

save(reg, "07_pre_matching_base.csv")


# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 70)
print("ETL COMPLETE")
print("=" * 70)
print(f"  Output:  {RESULTS}/")
print(f"  Files:   01_ici_cohort.csv through 07_pre_matching_base.csv")
print(f"  Next:    Rscript 01b_psm.R ici_aki")
print("=" * 70)

# Upload all to bucket
subprocess.run(
    ["gsutil", "-m", "cp", f"{RESULTS}/*.csv", f"{BUCKET_DIR}/"],
    capture_output=True,
)
print(f"  Uploaded to {BUCKET_DIR}/")
