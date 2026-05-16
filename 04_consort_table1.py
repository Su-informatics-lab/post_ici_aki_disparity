#!/usr/bin/env python3
"""
Post-ICI AKI × SDoH — CONSORT Numbers + Table 1
Reads results CSVs and produces publication-ready outputs.

Usage: python 04_consort_table1.py [cohort]
  cohort: ici_aki (default) or inpc

Inputs:  results/{cohort}/00_consort_numbers.csv  (from ETL)
         results/{cohort}/09_regression_base.csv   (from PSM)
         results/{cohort}/08c_smd_balance.csv      (from PSM)
         results/{cohort}/all_model_coefficients.csv
         results/{cohort}/all_sensitivity_coefficients.csv

Outputs: results/{cohort}/consort_flowchart.csv
         results/{cohort}/table1_characteristics.csv
         results/{cohort}/table2_model_summary.csv
"""

import os
import sys

import numpy as np
import pandas as pd

args = sys.argv[1:]
COHORT = args[0] if args else "ici_aki"
RESULTS = os.path.join("results", COHORT)

print("=" * 70)
print(f"CONSORT + TABLE 1 — {COHORT.upper()}")
print("=" * 70)


def save(df, filename):
    path = os.path.join(RESULTS, filename)
    df.to_csv(path, index=False)
    print(f"  Saved: {path} ({len(df)} rows)")


# ═══════════════════════════════════════════════════════════════════
# A. CONSORT FLOWCHART NUMBERS
# ═══════════════════════════════════════════════════════════════════
print("\n--- CONSORT FLOWCHART ---")

consort_file = os.path.join(RESULTS, "00_consort_numbers.csv")
if os.path.exists(consort_file):
    consort = pd.read_csv(consort_file)
    print(consort.to_string(index=False))
else:
    print("  00_consort_numbers.csv not found — run ETL first.")
    print("  (INPC ETL does not produce CONSORT; extract from ETL stdout)")


# ═══════════════════════════════════════════════════════════════════
# B. TABLE 1: COHORT CHARACTERISTICS (matched cohort)
# ═══════════════════════════════════════════════════════════════════
print("\n--- TABLE 1: COHORT CHARACTERISTICS ---")

reg_file = os.path.join(RESULTS, "09_regression_base.csv")
if not os.path.exists(reg_file):
    reg_file = os.path.join(RESULTS, "07_pre_matching_base.csv")
if not os.path.exists(reg_file):
    print("  ERROR: No regression base found.")
    sys.exit(1)

df = pd.read_csv(reg_file, low_memory=False)
print(f"  Loaded: {reg_file} ({len(df)} rows)")

# Outcome column
outcome_col = "severity" if "severity" in df.columns else "Treatment"
if outcome_col not in df.columns:
    print(f"  ERROR: No outcome column found")
    sys.exit(1)

cases = df[df[outcome_col] == 1]
controls = df[df[outcome_col] == 0]
print(f"  Cases: {len(cases)}  Controls: {len(controls)}")


def compute_smd(case_vals, ctrl_vals):
    """Standardized mean difference for a binary or continuous variable."""
    p1, p0 = case_vals.mean(), ctrl_vals.mean()
    # For binary: Cohen's h approximation
    pooled_sd = np.sqrt((case_vals.var() + ctrl_vals.var()) / 2)
    if pooled_sd == 0:
        return 0.0
    return (p1 - p0) / pooled_sd


def categorical_row(col, level, label=None):
    """One row of Table 1 for a categorical level."""
    if label is None:
        label = f"  {level}"
    c_n = (cases[col] == level).sum() if col in cases.columns else 0
    c_pct = c_n / len(cases) * 100 if len(cases) > 0 else 0
    t_n = (controls[col] == level).sum() if col in controls.columns else 0
    t_pct = t_n / len(controls) * 100 if len(controls) > 0 else 0
    all_n = c_n + t_n
    all_pct = all_n / len(df) * 100 if len(df) > 0 else 0

    # SMD for binary indicator
    case_ind = (
        (cases[col] == level).astype(float) if col in cases.columns else pd.Series([0])
    )
    ctrl_ind = (
        (controls[col] == level).astype(float)
        if col in controls.columns
        else pd.Series([0])
    )
    smd = compute_smd(case_ind, ctrl_ind)

    return {
        "Variable": label,
        "Cases_N": c_n,
        "Cases_Pct": round(c_pct, 1),
        "Controls_N": t_n,
        "Controls_Pct": round(t_pct, 1),
        "All_N": all_n,
        "All_Pct": round(all_pct, 1),
        "SMD": round(abs(smd), 3),
    }


def continuous_row(col, label):
    """One row for a continuous variable (median [IQR])."""
    c_vals = cases[col].dropna() if col in cases.columns else pd.Series(dtype=float)
    t_vals = (
        controls[col].dropna() if col in controls.columns else pd.Series(dtype=float)
    )
    a_vals = df[col].dropna() if col in df.columns else pd.Series(dtype=float)

    smd = compute_smd(c_vals, t_vals) if len(c_vals) > 0 and len(t_vals) > 0 else 0

    def fmt(s):
        if len(s) == 0:
            return "—"
        return f"{s.median():.1f} [{s.quantile(0.25):.1f}–{s.quantile(0.75):.1f}]"

    return {
        "Variable": label,
        "Cases_N": fmt(c_vals),
        "Cases_Pct": "",
        "Controls_N": fmt(t_vals),
        "Controls_Pct": "",
        "All_N": fmt(a_vals),
        "All_Pct": "",
        "SMD": round(abs(smd), 3),
    }


rows = []

# ── Sample size header ────────────────────────────────────────────
rows.append(
    {
        "Variable": "N",
        "Cases_N": len(cases),
        "Controls_N": len(controls),
        "All_N": len(df),
        "SMD": "",
    }
)

# ── Demographics ──────────────────────────────────────────────────
rows.append({"Variable": "Demographics", "SMD": ""})

if "sex_at_birth" in df.columns:
    rows.append({"Variable": "Sex at birth", "SMD": ""})
    for level in ["Male", "Female", "Other"]:
        if (df.sex_at_birth == level).sum() > 0:
            rows.append(categorical_row("sex_at_birth", level))

if "race" in df.columns:
    rows.append({"Variable": "Race", "SMD": ""})
    for level in [
        "White",
        "Black",
        "Asian",
        "Hispanic",
        "AIAN",
        "Native_Hawaiian_PI",
        "Other",
    ]:
        if (df.race == level).sum() > 0:
            rows.append(categorical_row("race", level))

if "ethnicity" in df.columns:
    rows.append({"Variable": "Ethnicity", "SMD": ""})
    for level in ["Not_Hispanic", "Hispanic", "Unknown"]:
        if (df.ethnicity == level).sum() > 0:
            rows.append(categorical_row("ethnicity", level))

if "age_group" in df.columns:
    rows.append({"Variable": "Age group", "SMD": ""})
    for level in ["18-44", "45-54", "55-64", "65-74", "75+"]:
        if (df.age_group == level).sum() > 0:
            rows.append(categorical_row("age_group", level))

if "age_at_ici" in df.columns:
    rows.append(continuous_row("age_at_ici", "Age at ICI, median [IQR]"))

# ── NCI-CCI ───────────────────────────────────────────────────────
rows.append({"Variable": "Comorbidities (NCI-CCI)", "SMD": ""})

if "charlson_score" in df.columns:
    rows.append(continuous_row("charlson_score", "Charlson score, median [IQR]"))
elif "nci_cci_score" in df.columns:
    rows.append(continuous_row("nci_cci_score", "NCI-CCI score, median [IQR]"))

if "nci_index" in df.columns:
    rows.append(continuous_row("nci_index", "NCI index, median [IQR]"))

# Individual NCI-CCI conditions
nci_conditions = [
    ("Acute_MI", "Acute MI"),
    ("History_MI", "History of MI"),
    ("Congestive_Heart_Failure", "CHF"),
    ("Peripheral_Vascular_Disease", "PVD"),
    ("Cerebrovascular_Disease", "CVD"),
    ("Chronic_Pulmonary_Disease", "COPD"),
    ("Dementia", "Dementia"),
    ("Paralysis", "Paralysis"),
    ("Diabetes", "Diabetes (any)"),
    ("Diabetes_Complicated", "Diabetes (complicated)"),
    ("Renal_Disease", "Renal disease"),
    ("Liver_Disease_Mild", "Liver disease (mild)"),
    ("Liver_Disease_Moderate_Severe", "Liver disease (mod/severe)"),
    ("Peptic_Ulcer_Disease", "PUD"),
    ("Rheumatic_Disease", "Rheumatic disease"),
    ("AIDS", "AIDS"),
]
for col, label in nci_conditions:
    if col in df.columns and df[col].sum() > 0:
        rows.append(categorical_row(col, 1, f"  {label}"))

# ── Cancer type ───────────────────────────────────────────────────
if "cancer_type_collapsed" in df.columns:
    rows.append({"Variable": "Cancer type (collapsed)", "SMD": ""})
    for level in ["Lung", "Melanoma", "Other"]:
        if (df.cancer_type_collapsed == level).sum() > 0:
            rows.append(categorical_row("cancer_type_collapsed", level))

if "cancer_type" in df.columns:
    rows.append({"Variable": "Cancer type (detailed)", "SMD": ""})
    for level in df.cancer_type.value_counts().index:
        rows.append(categorical_row("cancer_type", level))

# ── ICI regimen ───────────────────────────────────────────────────
if "ici_collapsed" in df.columns:
    rows.append({"Variable": "ICI regimen (collapsed)", "SMD": ""})
    for level in ["anti_pd1", "other_combo"]:
        if (df.ici_collapsed == level).sum() > 0:
            rows.append(categorical_row("ici_collapsed", level))

if "ici_regimen" in df.columns:
    rows.append({"Variable": "ICI regimen (detailed)", "SMD": ""})
    for level in df.ici_regimen.value_counts().index:
        rows.append(categorical_row("ici_regimen", level))

# ── Nephrotoxins ──────────────────────────────────────────────────
rows.append({"Variable": "Concomitant nephrotoxins", "SMD": ""})
for col, label in [
    ("ppi", "PPI"),
    ("nsaid", "NSAID"),
    ("acei_arb", "ACEi/ARB"),
    ("diuretic", "Diuretic"),
]:
    if col in df.columns:
        rows.append(categorical_row(col, 1, f"  {label}"))

# ── SDoH (AoU only) ──────────────────────────────────────────────
sdoh_domains = {
    "insurance_type": {
        "label": "Insurance type",
        "levels": [
            "Private",
            "Medicare",
            "Medicaid",
            "VA_Military",
            "Uninsured",
            "Other",
            "Unknown",
        ],
    },
    "income": {
        "label": "Household income",
        "levels": [
            "gt100k",
            "75k_100k",
            "50k_75k",
            "25k_50k",
            "10k_25k",
            "lt10k",
            "Unknown",
        ],
    },
    "education": {
        "label": "Education",
        "levels": ["Graduate", "College", "Some_College", "HS_GED", "lt_HS", "Unknown"],
    },
    "employment": {
        "label": "Employment",
        "levels": [
            "Employed",
            "Self_Employed",
            "Retired",
            "Unable_to_Work",
            "Unemployed",
            "Student",
            "Homemaker",
            "Other",
            "Unknown",
        ],
    },
    "housing": {
        "label": "Housing",
        "levels": ["Own", "Rent", "Other_Arrangement", "Unknown"],
    },
    "housing_stability": {
        "label": "Housing stability concern",
        "levels": ["Stable", "Unstable", "Unknown"],
    },
}

has_any_sdoh = any(col in df.columns for col in sdoh_domains)
if has_any_sdoh:
    rows.append({"Variable": "Social determinants of health", "SMD": ""})
    for col, info in sdoh_domains.items():
        if col in df.columns:
            rows.append({"Variable": info["label"], "SMD": ""})
            for level in info["levels"]:
                if (df[col] == level).sum() > 0:
                    rows.append(categorical_row(col, level))

# ── Baseline creatinine ───────────────────────────────────────────
if "baseline_cr" in df.columns:
    rows.append({"Variable": "Baseline characteristics", "SMD": ""})
    rows.append(continuous_row("baseline_cr", "Baseline Cr (mg/dL), median [IQR]"))

if "max_cr_ratio" in df.columns:
    rows.append(continuous_row("max_cr_ratio", "Max Cr ratio, median [IQR]"))

# ── EHR utilization ───────────────────────────────────────────────
if "n_diagnoses" in df.columns:
    rows.append({"Variable": "EHR utilization", "SMD": ""})
    rows.append(continuous_row("n_diagnoses", "Unique diagnoses, median [IQR]"))
if "ehr_length_days" in df.columns:
    rows.append(continuous_row("ehr_length_days", "EHR length (days), median [IQR]"))

# Build Table 1
table1 = pd.DataFrame(rows)
# Fill NaN
for col in table1.columns:
    table1[col] = table1[col].fillna("")

save(table1, "table1_characteristics.csv")

# Print summary
print("\n  Table 1 preview (first 30 rows):")
print(table1.head(30).to_string(index=False))


# ═══════════════════════════════════════════════════════════════════
# C. TABLE 2: MODEL RESULTS SUMMARY
# ═══════════════════════════════════════════════════════════════════
print("\n--- TABLE 2: MODEL RESULTS SUMMARY ---")

coef_file = os.path.join(RESULTS, "all_model_coefficients.csv")
if os.path.exists(coef_file):
    all_coefs = pd.read_csv(coef_file)

    # Key predictors to highlight
    key_vars = [
        "nci_cci_score",
        "nci_index",
        "f.raceBlack",
        "f.raceOther",
        "f.cancerMelanoma",
        "f.cancerOther",
        "f.iciother_combo",
        "ppi",
        "nsaid",
        "acei_arb",
        "diuretic",
        "f.insuranceMedicaid",
        "f.insuranceMedicare",
        "f.incomelt10k",
        "f.income10k_25k",
        "f.educationlt_HS",
        "f.educationHS_GED",
        "f.employmentUnable_to_Work",
        "f.employmentUnemployed",
        "f.housingRent",
        "f.housingOther_Arrangement",
        "f.stabilityUnstable",
    ]

    summary_rows = []
    for _, row in all_coefs.iterrows():
        if row["variable"] in key_vars or row.get("p", 1) < 0.05:
            summary_rows.append(
                {
                    "Model": row.get("model", ""),
                    "Variable": row["variable"],
                    "AOR": round(row["exp_coef"], 2),
                    "CI_Lower": round(row["lower95"], 2),
                    "CI_Upper": round(row["upper95"], 2),
                    "P": f"{row['p']:.4f}" if row["p"] >= 0.001 else f"{row['p']:.2e}",
                }
            )

    table2 = pd.DataFrame(summary_rows)
    save(table2, "table2_model_summary.csv")
    print(f"  Key results: {len(table2)} rows")
else:
    print("  all_model_coefficients.csv not found — run 02_models.R first")


# ═══════════════════════════════════════════════════════════════════
# D. RACE ATTENUATION SUMMARY
# ═══════════════════════════════════════════════════════════════════
atten_file = os.path.join(RESULTS, "race_attenuation.csv")
if os.path.exists(atten_file):
    print("\n--- RACE ATTENUATION ---")
    atten = pd.read_csv(atten_file)
    print(atten.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════
# E. SENSITIVITY SUMMARY
# ═══════════════════════════════════════════════════════════════════
sens_file = os.path.join(RESULTS, "sensitivity_summary_comparison.csv")
if os.path.exists(sens_file):
    print("\n--- SENSITIVITY: BLACK AOR ACROSS THRESHOLDS ---")
    sens = pd.read_csv(sens_file)
    print(sens.to_string(index=False))


print("\n" + "=" * 70)
print(f"TABLE EXTRACTION COMPLETE — {COHORT.upper()}")
print("=" * 70)
