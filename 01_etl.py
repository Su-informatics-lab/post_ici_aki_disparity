#!/usr/bin/env python3
"""
Post-ICI AKI x SDoH -- Consolidated ETL (v6)
Usage:  python 01_etl.py aou      # AoU Workbench (BigQuery)
        python 01_etl.py inpc     # Quartz HPC (local CSV)

v6 changes vs v5:
  - AKI phenotype aligned with dualr-graph (01_etl_quartz.py):
    Cr >=1.5x baseline  OR  AKI ICD (N17/584) + severe/incident visit
  - Primary window: 180 days (was 365)
  - 365-day kept as sensitivity column (aki_365d / severity_12m)
  - Washout: Cr-based OR ICD-based (union), 90d pre-ICI
  - Cancer collapse: Lung [ref], Melanoma, Other  (Renal_Cell into Other)
  - ICI collapse:    anti_pd1 [ref], anti_pdl1, ctla4_containing
  - baseline_egfr:   kept in CSV for Table 1; excluded from base model in R

Phenotype:
  - Baseline Cr: median [-365,-7], fallback last [-365,-1]
  - AKI: Cr >=1.5x baseline OR AKI ICD (N17/584) + severe/incident visit
  - Primary window: 180 days; 365-day as sensitivity
  - Pre-ICI washout: Cr >=1.5x OR AKI ICD in [-90, 0] excluded
  - Cr plausibility: >= 0.1 mg/dL
  - Nephrotoxins: [-90, 0] days pre-ICI only
  - Baseline eGFR: CKD-EPI 2021 race-free
  - NCI-CCI: corrected MI/paralysis/hierarchy scoring
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from nci_cci_scoring import (
    NCI_CCI_CONDITIONS,
    NCI_CODESETS,
    compute_charlson_score,
    compute_nci_index,
)

# =====================================================================
# CLI + CONFIG
# =====================================================================
if len(sys.argv) < 2 or sys.argv[1] not in ("aou", "inpc"):
    print("Usage: python 01_etl.py [aou|inpc]")
    sys.exit(1)

MODE = sys.argv[1]
RESULTS = f"results/{'ici_aki' if MODE == 'aou' else 'inpc'}"
os.makedirs(RESULTS, exist_ok=True)

# INPC data path (ignored in aou mode)
INPC_DATA = "/N/project/depot/hw56/irAKI_data/structured_data"

# AoU CDR (ignored in inpc mode)
CDR = os.environ.get("WORKSPACE_CDR", "")
if MODE == "aou" and not CDR:
    print("ERROR: WORKSPACE_CDR not set. Run on AoU Workbench.")
    sys.exit(1)

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

NEPHROTOXIN_CLASSES = {
    "ppi": [
        "omeprazole",
        "pantoprazole",
        "lansoprazole",
        "esomeprazole",
        "rabeprazole",
        "dexlansoprazole",
    ],
    "nsaid": [
        "ibuprofen",
        "naproxen",
        "diclofenac",
        "meloxicam",
        "celecoxib",
        "indomethacin",
        "ketorolac",
        "piroxicam",
    ],
    "acei_arb": [
        "lisinopril",
        "enalapril",
        "ramipril",
        "benazepril",
        "captopril",
        "losartan",
        "valsartan",
        "irbesartan",
        "olmesartan",
        "candesartan",
        "telmisartan",
    ],
    "diuretic": [
        "furosemide",
        "hydrochlorothiazide",
        "chlorthalidone",
        "bumetanide",
        "torsemide",
        "spironolactone",
        "amiloride",
    ],
}

SDOH_CONCEPTS = {
    "insurance_type": 43528428,
    "income": 1585375,
    "education": 1585940,
    "employment": 1585952,
    "housing": 1585370,
    "housing_stability": 1585886,
}

# ── v6: AKI ICD codes + severe visit types + window config ────────
AKI_ICD_PREFIXES = {"9": ["584"], "10": ["N17"]}
AKI_ICD_NORM = [
    p.upper().replace(".", "") for codes in AKI_ICD_PREFIXES.values() for p in codes
]
SEVERE_VISIT_CONCEPTS = {9201, 9203, 262, 8717, 8782}
PRIMARY_WINDOW_DAYS = 180
SENSITIVITY_WINDOW_DAYS = 365


# =====================================================================
# SHARED HELPERS
# =====================================================================
def save(df, filename):
    path = os.path.join(RESULTS, filename)
    df.to_csv(path, index=False)
    print(f"  Saved: {path} ({len(df):,} rows, {df.shape[1]} cols)")


def parse_date(s):
    return pd.to_datetime(s, format="mixed", dayfirst=False, errors="coerce")


def q(sql):
    """BigQuery helper (AoU only)."""
    return pd.read_gbq(sql, dialect="standard")


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


def classify_cancer(code):
    """ICD-10-CM C-code -> cancer type (shared AoU/INPC)."""
    if pd.isna(code):
        return "Other_Solid"
    code = str(code).upper().replace(".", "")
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
    if any(code.startswith(p) for p in ["C10", "C11", "C12", "C13", "C14", "C32"]):
        return "Head_Neck"
    if code.startswith("C22"):
        return "Hepatocellular"
    if any(code.startswith(p) for p in ["C18", "C19", "C20"]):
        return "Colorectal"
    if any(
        code.startswith(p) for p in ["C81", "C82", "C83", "C84", "C85", "C91", "C92"]
    ):
        return "Hematologic"
    return "Other_Solid"


def classify_ici(drugs_list):
    """Drug name list -> ICI regimen class (shared AoU/INPC)."""
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


def compute_egfr(baseline_cr, age, is_female):
    """CKD-EPI 2021 race-free equation."""
    scr = np.asarray(baseline_cr, dtype=np.float64)
    age_arr = np.asarray(age, dtype=np.float64)
    fem = np.asarray(is_female, dtype=bool)
    kappa = np.where(fem, 0.7, 0.9)
    alpha = np.where(fem, -0.241, -0.302)
    ratio = scr / kappa
    return (
        142
        * np.power(np.minimum(ratio, 1.0), alpha)
        * np.power(np.maximum(ratio, 1.0), -1.200)
        * np.power(0.9938, age_arr)
        * np.where(fem, 1.012, 1.0)
    ).astype(np.float32)


# ── v5: Cancer and ICI collapse functions ────────────────────────
def collapse_cancer(cancer_type):
    """3-level: Lung [ref], Melanoma, Other.
    v5.1/v6: Renal_Cell folded back into Other (saves 1 df at EPV ~10).
    Renal_Cell (N=53 AoU) not significant in base model."""
    if cancer_type in ("Lung", "Melanoma"):
        return cancer_type
    return "Other"


def collapse_ici(ici_regimen):
    """3-level: anti_pd1 [ref], anti_pdl1, ctla4_containing.
    CTLA-4 containing regimens (mono + combo) carry highest AKI risk
    (Liu 2023 meta-analysis OR ~2.5). PD-L1 separated from PD-1
    per dualr-graph convention."""
    if ici_regimen == "anti_pd1":
        return "anti_pd1"
    if ici_regimen == "anti_pdl1":
        return "anti_pdl1"
    # combo_pd1_ctla4, anti_ctla4, other -> ctla4_containing
    return "ctla4_containing"


# ── Baseline Cr logic (shared) ───────────────────────────────────
def compute_baseline_followup(cr_merged, eligible_pids):
    """Compute baseline Cr (median [-365,-7] + fallback) and follow-up max."""
    # Primary window
    bl_main = cr_merged[
        (cr_merged.days_from_ici >= -365) & (cr_merged.days_from_ici <= -7)
    ]
    bl_primary = (
        bl_main.groupby("person_id")
        .agg(
            baseline_cr=("value_as_number", "median"),
            n_baseline=("value_as_number", "count"),
        )
        .reset_index()
    )
    # Fallback: last Cr in [-365, -1]
    bl_fb_pool = cr_merged[
        (cr_merged.days_from_ici >= -365) & (cr_merged.days_from_ici <= -1)
    ].sort_values(["person_id", "days_from_ici"])
    bl_fb = (
        bl_fb_pool.groupby("person_id")
        .tail(1)[["person_id", "value_as_number"]]
        .rename(columns={"value_as_number": "baseline_cr"})
    )
    bl_fb["n_baseline"] = 1
    pids_primary = set(bl_primary.person_id)
    bl_fb_only = bl_fb[~bl_fb.person_id.isin(pids_primary)]
    baseline = pd.concat([bl_primary, bl_fb_only], ignore_index=True)

    # Follow-up: max Cr in [1, 365]
    fu = cr_merged[(cr_merged.days_from_ici >= 1) & (cr_merged.days_from_ici <= 365)]
    followup = (
        fu.groupby("person_id")
        .agg(
            max_followup_cr=("value_as_number", "max"),
            n_followup=("value_as_number", "count"),
        )
        .reset_index()
    )

    return baseline, followup, len(bl_primary), len(bl_fb_only)


def apply_aki_phenotype(eligible, cr_merged, icd_first=None):
    """Apply AKI definitions: Cr >=1.5x OR ICD+severe (v6 dual phenotype).

    Args:
        eligible: DataFrame with person_id, baseline_cr, ici_index_date, etc.
        cr_merged: Creatinine measurements merged with ICI dates (has days_from_ici).
        icd_first: DataFrame with [person_id, icd_evt_date] — first ICD-based AKI
                   event per patient (already filtered for severe/incident). None if
                   no ICD data available (shouldn't happen in v6).

    Returns:
        eligible with severity (180d primary), sensitivity columns, evt_source.
    """
    eligible = eligible.copy()
    eligible["max_cr_ratio"] = eligible.max_followup_cr / eligible.baseline_cr
    eligible["max_delta_cr"] = eligible.max_followup_cr - eligible.baseline_cr

    # ── Cr-based event dates (first Cr >=1.5x within 365d) ────────
    fu = cr_merged[
        (cr_merged.days_from_ici >= 1)
        & (cr_merged.days_from_ici <= SENSITIVITY_WINDOW_DAYS)
    ].copy()
    fu = fu.merge(eligible[["person_id", "baseline_cr"]], on="person_id")
    fu["cr_ratio"] = fu.value_as_number / fu.baseline_cr
    cr_events = fu[fu.cr_ratio >= 1.5].copy()
    if len(cr_events) > 0:
        cr_first = (
            cr_events.sort_values("measurement_date")
            .groupby("person_id")
            .first()
            .reset_index()[["person_id", "measurement_date"]]
            .rename(columns={"measurement_date": "cr_evt_date"})
        )
    else:
        cr_first = pd.DataFrame(columns=["person_id", "cr_evt_date"])

    # ── Union: Cr + ICD (earliest event wins) ─────────────────────
    eligible = eligible.merge(cr_first, on="person_id", how="left")
    if icd_first is not None and len(icd_first) > 0:
        eligible = eligible.merge(icd_first, on="person_id", how="left")
    else:
        eligible["icd_evt_date"] = pd.NaT

    cok = eligible.cr_evt_date.notna()
    iok = eligible.icd_evt_date.notna()
    cr_wins = cok & (~iok | (eligible.cr_evt_date <= eligible.icd_evt_date))
    icd_wins = iok & (~cok | (eligible.icd_evt_date < eligible.cr_evt_date))

    eligible["evt_date"] = pd.NaT
    eligible.loc[cr_wins, "evt_date"] = eligible.loc[cr_wins, "cr_evt_date"]
    eligible.loc[icd_wins, "evt_date"] = eligible.loc[icd_wins, "icd_evt_date"]
    eligible["evt_source"] = "none"
    eligible.loc[cr_wins, "evt_source"] = "cr"
    eligible.loc[icd_wins, "evt_source"] = "icd"
    eligible["evt_days"] = (eligible.evt_date - eligible.ici_index_date).dt.days

    # ── Primary: 180-day window ───────────────────────────────────
    eligible["severity"] = 0
    eligible.loc[
        (eligible.evt_source != "none") & (eligible.evt_days <= PRIMARY_WINDOW_DAYS),
        "severity",
    ] = 1

    # ── Sensitivity: 365-day window ───────────────────────────────
    eligible["severity_12m"] = (eligible.evt_source != "none").astype(int)
    eligible["aki_365d"] = eligible["severity_12m"]

    # ── Cr-only sensitivity thresholds (within primary window) ────
    # These use Cr ratios only; ICD events don't have lab values
    fu_primary = cr_merged[
        (cr_merged.days_from_ici >= 1)
        & (cr_merged.days_from_ici <= PRIMARY_WINDOW_DAYS)
    ]
    max_primary = (
        fu_primary.groupby("person_id")
        .value_as_number.max()
        .reset_index()
        .rename(columns={"value_as_number": "max_cr_primary"})
    )
    eligible = eligible.merge(max_primary, on="person_id", how="left")
    eligible["max_ratio_primary"] = eligible.max_cr_primary / eligible.baseline_cr
    eligible["max_delta_primary"] = eligible.max_cr_primary - eligible.baseline_cr

    eligible["aki_delta03"] = (
        (eligible.max_delta_primary >= 0.3).fillna(False)
        | (eligible.evt_source == "icd")  # ICD-coded AKI counts as >=0.3
    ).astype(int)
    # Only count within primary window for delta03
    eligible.loc[
        (eligible.evt_source == "icd") & (eligible.evt_days > PRIMARY_WINDOW_DAYS),
        "aki_delta03",
    ] = 0

    eligible["aki_kdigo2"] = (
        (eligible.max_ratio_primary >= 2.0).fillna(False).astype(int)
    )
    eligible["aki_kdigo3"] = (
        (eligible.max_ratio_primary >= 3.0).fillna(False).astype(int)
    )

    # Print summary
    n_cr = int((eligible.evt_source == "cr").sum())
    n_icd = int((eligible.evt_source == "icd").sum())
    n_primary = int(eligible.severity.sum())
    n_12m = int(eligible.severity_12m.sum())
    print(f"  AKI events (union, <={SENSITIVITY_WINDOW_DAYS}d): {n_12m}")
    print(f"    Cr-first: {n_cr}, ICD-first: {n_icd}")
    print(
        f"  PRIMARY events (<={PRIMARY_WINDOW_DAYS}d): {n_primary} ({n_primary/len(eligible)*100:.1f}%)"
    )
    print(f"  12-month events (sensitivity): {n_12m} ({n_12m/len(eligible)*100:.1f}%)")

    return eligible


# ── NCI-CCI builder (shared) ────────────────────────────────────
def build_nci_cci(person_ids, dx_df, icd_col="icd_code"):
    """Build NCI-CCI flags + scores from diagnosis DataFrame."""
    charlson = pd.DataFrame({"person_id": person_ids})
    for c in NCI_CCI_CONDITIONS:
        charlson[c] = 0
    for condition, codes in NCI_CODESETS.items():
        all_pfx = []
        for ver_codes in codes.values():
            all_pfx.extend(ver_codes)
        mask = dx_df[icd_col].apply(
            lambda x: (
                any(str(x).startswith(p) for p in all_pfx) if pd.notna(x) else False
            )
        )
        flagged = dx_df[mask].person_id.unique()
        charlson.loc[charlson.person_id.isin(flagged), condition] = 1
    for c in NCI_CCI_CONDITIONS:
        charlson[c] = charlson[c].astype("int8")
    charlson["charlson_score"] = compute_charlson_score(charlson)
    charlson["nci_index"] = compute_nci_index(charlson)
    charlson["nci_cci_score"] = charlson["charlson_score"]  # R compat
    return charlson


# ── v6: ICD-based AKI event computation (shared) ─────────────────
def compute_icd_aki(aki_icd_df, eligible, severe_visits=None):
    """Compute ICD-based AKI events and washout from AKI ICD records.

    Args:
        aki_icd_df: DataFrame [person_id, condition_start_date, visit_occurrence_id]
        eligible: DataFrame with [person_id, ici_index_date]
        severe_visits: set of visit_occurrence_ids that are ED/inpatient, or None
                       (if None, all visits accepted)

    Returns:
        icd_first: DataFrame [person_id, icd_evt_date] — first post-ICI ICD event
        washout_icd_pids: set of person_ids with pre-ICI AKI ICD codes
    """
    if len(aki_icd_df) == 0:
        return (
            pd.DataFrame(columns=["person_id", "icd_evt_date"]),
            set(),
        )

    aki = aki_icd_df.copy()
    aki["condition_start_date"] = parse_date(aki.condition_start_date)
    aki = aki.merge(eligible[["person_id", "ici_index_date"]], on="person_id")
    aki["days"] = (aki.condition_start_date - aki.ici_index_date).dt.days

    # Filter by severe visit type (if visit table available)
    if severe_visits is not None:
        aki["visit_occurrence_id"] = pd.to_numeric(
            aki.visit_occurrence_id, errors="coerce"
        )
        is_severe = aki.visit_occurrence_id.isin(severe_visits)
    else:
        is_severe = pd.Series(True, index=aki.index)

    # Pre-ICI ICD AKI → washout
    pre_icd = aki[(aki.days >= -90) & (aki.days <= 0)]
    washout_icd_pids = set(int(p) for p in pre_icd.person_id.unique())

    # Incident = no pre-ICI AKI ICD
    is_incident = ~aki.person_id.isin(set(pre_icd.person_id))

    # Keep: severe visit OR incident (same logic as dualr-graph)
    aki = aki[is_severe | is_incident].copy()

    # Post-ICI ICD events (1–365 days)
    aki = aki[(aki.days >= 1) & (aki.days <= SENSITIVITY_WINDOW_DAYS)]

    if len(aki) > 0:
        icd_first = (
            aki.sort_values("condition_start_date")
            .groupby("person_id")
            .first()
            .reset_index()[["person_id", "condition_start_date"]]
            .rename(columns={"condition_start_date": "icd_evt_date"})
        )
    else:
        icd_first = pd.DataFrame(columns=["person_id", "icd_evt_date"])

    return icd_first, washout_icd_pids


# =====================================================================
# AoU-SPECIFIC FUNCTIONS
# =====================================================================
def classify_insurance(name):
    if pd.isna(name):
        return "Unknown"
    n = str(name).lower()
    if "medicaid" in n:
        return "Medicaid"
    if "medicare" in n:
        return "Medicare"
    if any(
        k in n
        for k in [
            "employer",
            "union",
            "private",
            "purchased",
            "exchange",
            "marketplace",
        ]
    ):
        return "Private"
    if any(k in n for k in ["military", "va ", "tricare", "champva"]):
        return "VA_Military"
    if any(k in n for k in ["indian", "ihs"]):
        return "IHS"
    if any(k in n for k in ["no", "uninsured", "none"]):
        return "Uninsured"
    return "Other"


def classify_income(name):
    if pd.isna(name):
        return "Unknown"
    n = str(name).lower()
    if "less 10k" in n:
        return "lt10k"
    if "10k 25k" in n:
        return "10k_25k"
    if "25k 35k" in n or "35k 50k" in n:
        return "25k_50k"
    if "50k 75k" in n:
        return "50k_75k"
    if "75k 100k" in n:
        return "75k_100k"
    if any(k in n for k in ["100k 150k", "150k 200k", "more 200k"]):
        return "gt100k"
    if "skip" in n or "prefer not" in n:
        return "Unknown"
    return "Unknown"


def classify_education(name):
    if pd.isna(name):
        return "Unknown"
    n = str(name).lower()
    if any(
        k in n
        for k in [
            "never",
            "one through four",
            "five through eight",
            "nine through eleven",
        ]
    ):
        return "lt_HS"
    if "twelve" in n or "ged" in n:
        return "HS_GED"
    if "college one" in n or "one to three" in n:
        return "Some_College"
    if "college graduate" in n:
        return "College"
    if "advanced" in n:
        return "Graduate"
    if "skip" in n or "prefer not" in n:
        return "Unknown"
    return "Unknown"


def classify_employment(name):
    if pd.isna(name):
        return "Unknown"
    n = str(name).lower()
    if "employed" in n and "not" not in n and "self" not in n:
        return "Employed"
    if "self" in n:
        return "Self_Employed"
    if any(k in n for k in ["unemployed", "out of work", "looking"]):
        return "Unemployed"
    if "retired" in n:
        return "Retired"
    if any(k in n for k in ["unable", "disabled", "disability"]):
        return "Unable_to_Work"
    if "student" in n:
        return "Student"
    if "homemaker" in n:
        return "Homemaker"
    if "skip" in n or "prefer not" in n:
        return "Unknown"
    return "Other"


def classify_housing(name):
    if pd.isna(name):
        return "Unknown"
    n = str(name).lower()
    if ": rent" in n:
        return "Rent"
    if ": other" in n or "arrangement" in n:
        return "Other_Arrangement"
    if ": own" in n:
        return "Own"
    if "skip" in n or "prefer not" in n or "dont know" in n:
        return "Unknown"
    return "Unknown"


def classify_stability(name):
    if pd.isna(name):
        return "Unknown"
    n = str(name).lower()
    if "concern: no" in n:
        return "Stable"
    if "concern: yes" in n:
        return "Unstable"
    if "skip" in n or "prefer not" in n:
        return "Unknown"
    return "Unknown"


# =====================================================================
# MAIN: AoU PIPELINE
# =====================================================================
def run_aou():
    consort = {}

    print("=" * 70)
    print("POST-ICI AKI x SDoH -- AoU ETL (v6)")
    print("=" * 70)
    print(f"  CDR: {CDR}")
    print(f"  Output: {RESULTS}/")

    total_sql = f"SELECT COUNT(DISTINCT person_id) AS n FROM `{CDR}.person`"
    consort["total_aou"] = q(total_sql).n.iloc[0]
    print(f"  Total AoU participants: {consort['total_aou']:,}")

    # ── STEP 1: ICI Cohort + AKI ────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 1: ICI Cohort + AKI Phenotyping")
    print("=" * 70)

    agent_likes = " OR ".join(
        [f"LOWER(c.concept_name) LIKE '%{a}%'" for a in ICI_AGENTS]
    )
    ici_concepts = q(f"""
        SELECT DISTINCT c.concept_id, c.concept_name
        FROM `{CDR}.concept` c
        WHERE ({agent_likes}) AND c.domain_id = 'Drug' AND c.standard_concept = 'S'
    """)
    ici_ids = tuple(ici_concepts.concept_id.tolist())
    print(f"  Found {len(ici_ids)} ICI drug concept IDs")
    if not ici_ids:
        print("FATAL: No ICI concepts found.")
        sys.exit(1)

    ici_patients = q(f"""
        SELECT de.person_id,
               MIN(de.drug_exposure_start_date) AS ici_index_date,
               ARRAY_AGG(DISTINCT LOWER(c.concept_name) ORDER BY LOWER(c.concept_name)) AS ici_drugs
        FROM `{CDR}.drug_exposure` de
        JOIN `{CDR}.concept` c ON c.concept_id = de.drug_concept_id
        WHERE de.drug_concept_id IN ({','.join(str(x) for x in ici_ids)})
        GROUP BY de.person_id
    """)
    ici_patients["ici_index_date"] = parse_date(ici_patients["ici_index_date"])
    consort["ici_treated"] = len(ici_patients)
    print(f"  ICI-treated patients: {len(ici_patients):,}")

    # Cancer filter
    cancer_pts = q(f"""
        SELECT DISTINCT co.person_id FROM `{CDR}.condition_occurrence` co
        JOIN `{CDR}.concept` c ON c.concept_id = co.condition_source_concept_id
        WHERE c.vocabulary_id = 'ICD10CM'
          AND (c.concept_code LIKE 'C%' OR c.concept_code LIKE 'D0%'
               OR c.concept_code LIKE 'D1%' OR c.concept_code LIKE 'D2%'
               OR c.concept_code LIKE 'D3%' OR c.concept_code LIKE 'D4%')
    """)
    ici_cancer = ici_patients[ici_patients.person_id.isin(cancer_pts.person_id)]
    consort["cancer_pts_total"] = len(cancer_pts)
    consort["ici_cancer"] = len(ici_cancer)
    print(f"  ICI + cancer: {len(ici_cancer):,}")

    # Basics Survey filter
    basics_pts = q(f"""
        SELECT DISTINCT person_id FROM `{CDR}.observation`
        WHERE observation_source_concept_id = 1585845
    """)
    ici_cb = ici_cancer[ici_cancer.person_id.isin(basics_pts.person_id)]
    consort["ici_cancer_basics"] = len(ici_cb)
    consort["excluded_no_basics"] = len(ici_cancer) - len(ici_cb)
    print(f"  ICI + cancer + Basics Survey: {len(ici_cb):,}")

    # Creatinine
    pids_str = ",".join(str(x) for x in ici_cb.person_id.tolist())
    cr_all = q(f"""
        SELECT m.person_id, m.measurement_date, m.value_as_number
        FROM `{CDR}.measurement` m
        WHERE m.measurement_concept_id = 3016723
          AND m.value_as_number IS NOT NULL AND m.value_as_number >= 0.1
          AND m.value_as_number < 30
    """)
    cr_all["measurement_date"] = parse_date(cr_all["measurement_date"])
    cr_all = cr_all[cr_all.person_id.isin(ici_cb.person_id)]
    print(f"  Creatinine measurements (ICI patients): {len(cr_all):,}")

    cr_merged = cr_all.merge(ici_cb[["person_id", "ici_index_date"]], on="person_id")
    cr_merged["days_from_ici"] = (
        cr_merged.measurement_date - cr_merged.ici_index_date
    ).dt.days

    baseline, followup, n_pri, n_fb = compute_baseline_followup(
        cr_merged, ici_cb.person_id
    )
    print(f"  Patients with baseline Cr: {len(baseline):,}")
    print(f"    Primary [-365, -7] median: {n_pri:,}")
    print(f"    Fallback [-365, -1] last:  {n_fb:,}")
    consort["has_baseline_cr"] = len(baseline)
    consort["baseline_cr_primary"] = n_pri
    consort["baseline_cr_fallback"] = n_fb
    print(f"  Patients with follow-up Cr: {len(followup):,}")
    consort["has_followup_cr"] = len(followup)

    eligible = ici_cb.merge(baseline, on="person_id").merge(followup, on="person_id")

    # ESKD exclusion
    eskd_pts = q(f"""
        SELECT DISTINCT co.person_id FROM `{CDR}.condition_occurrence` co
        JOIN `{CDR}.concept` c ON c.concept_id = co.condition_source_concept_id
        WHERE c.vocabulary_id = 'ICD10CM'
          AND (c.concept_code IN ('N186')
               OR c.concept_code LIKE 'Z992%' OR c.concept_code LIKE 'Z490%'
               OR c.concept_code LIKE 'Z491%' OR c.concept_code LIKE 'Z492%'
               OR c.concept_code LIKE 'Z940%')
    """)
    pre = len(eligible)
    eligible = eligible[~eligible.person_id.isin(eskd_pts.person_id)]
    eligible = eligible[eligible.baseline_cr < 4.0]
    consort["pre_eskd_exclusion"] = pre
    consort["excluded_eskd"] = pre - len(eligible)
    print(f"  Excluded ESKD/transplant/baseline>=4: {pre - len(eligible)}")

    # Pre-ICI Cr washout
    pre_cr = cr_merged[
        (cr_merged.days_from_ici >= -90) & (cr_merged.days_from_ici <= 0)
    ]
    pre_cr = pre_cr.merge(eligible[["person_id", "baseline_cr"]], on="person_id")
    washout_cr = set(
        pre_cr[pre_cr.value_as_number / pre_cr.baseline_cr >= 1.5].person_id
    )
    print(f"  Washout Cr-based (Cr >=1.5x in 90d pre-ICI): {len(washout_cr)}")

    # v6: AKI ICD codes + severe visits
    pids_elig_str = ",".join(str(x) for x in eligible.person_id.tolist())
    aki_icd_prefixes_sql = " OR ".join(
        [f"UPPER(REPLACE(c.concept_code, '.', '')) LIKE '{p}%'" for p in AKI_ICD_NORM]
    )
    aki_icd_df = q(f"""
        SELECT co.person_id, co.condition_start_date, co.visit_occurrence_id
        FROM `{CDR}.condition_occurrence` co
        JOIN `{CDR}.concept` c ON c.concept_id = co.condition_source_concept_id
        WHERE c.vocabulary_id IN ('ICD9CM','ICD10CM')
          AND ({aki_icd_prefixes_sql})
          AND co.person_id IN ({pids_elig_str})
    """)
    print(
        f"  AKI ICD records: {len(aki_icd_df):,}, patients: {aki_icd_df.person_id.nunique():,}"
    )

    severe_concepts_sql = ",".join(str(c) for c in SEVERE_VISIT_CONCEPTS)
    severe_visits_df = q(f"""
        SELECT DISTINCT visit_occurrence_id
        FROM `{CDR}.visit_occurrence`
        WHERE person_id IN ({pids_elig_str})
          AND visit_concept_id IN ({severe_concepts_sql})
    """)
    severe_visits = set(severe_visits_df.visit_occurrence_id)
    print(f"  Severe visits (ED/inpatient): {len(severe_visits):,}")

    icd_first, washout_icd = compute_icd_aki(aki_icd_df, eligible, severe_visits)
    print(f"  Washout ICD-based (AKI ICD in 90d pre-ICI): {len(washout_icd)}")
    print(f"  ICD AKI events (post-ICI): {len(icd_first):,}")

    # v6: Union washout
    washout_all = washout_cr | washout_icd
    eligible = eligible[~eligible.person_id.isin(washout_all)].copy()
    consort["excluded_washout"] = len(washout_all)
    consort["excluded_washout_cr"] = len(washout_cr)
    consort["excluded_washout_icd"] = len(washout_icd)
    consort["excluded_washout_overlap"] = len(washout_cr & washout_icd)
    consort["eligible"] = len(eligible)
    print(
        f"  Washout total: {len(washout_all)} excluded "
        f"({len(washout_cr)} Cr, {len(washout_icd)} ICD, "
        f"{len(washout_cr & washout_icd)} overlap)"
    )
    print(f"  Eligible cohort: {len(eligible):,}")

    # AKI phenotype (v6: Cr OR ICD, 180d primary)
    eligible = apply_aki_phenotype(eligible, cr_merged, icd_first)
    cases = eligible.severity.sum()
    consort["cases"] = int(cases)
    consort["controls"] = int(len(eligible) - cases)
    consort["cases_12m"] = int(eligible.severity_12m.sum())
    consort["primary_window_days"] = PRIMARY_WINDOW_DAYS
    consort["excluded_no_baseline"] = (
        consort["ici_cancer_basics"] - consort["has_baseline_cr"]
    )
    consort["excluded_no_followup"] = (
        consort["has_baseline_cr"] - consort["pre_eskd_exclusion"]
    )
    print(
        f"  Cases ({PRIMARY_WINDOW_DAYS}d primary): {cases:,} ({cases/len(eligible)*100:.1f}%)"
    )
    print(f"  Controls:          {len(eligible)-cases:,}")

    consort_df = pd.DataFrame([consort]).T.reset_index()
    consort_df.columns = ["step", "n"]
    save(consort_df, "00_consort_numbers.csv")
    print("\n  CONSORT flowchart:")
    for step, n in consort.items():
        print(f"    {step:35s} {int(n):>10,}")
    save(eligible, "01_eligible_cohort.csv")

    # ── STEP 2: Demographics ────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: Demographics")
    print("=" * 70)
    pids_str = ",".join(str(x) for x in eligible.person_id.tolist())
    demo = q(f"""
        SELECT p.person_id, p.year_of_birth, p.sex_at_birth_concept_id,
               p.race_concept_id, p.ethnicity_concept_id
        FROM `{CDR}.person` p
        WHERE p.person_id IN ({pids_str})
    """)
    sex_map = {45880669: "Male", 45878463: "Female"}
    demo["sex_at_birth"] = demo.sex_at_birth_concept_id.map(sex_map).fillna("Other")

    # Race from survey
    try:
        race_df = q(f"""
            SELECT person_id, answer.concept_name AS race_name
            FROM `{CDR}.ds_survey` WHERE survey = 'The Basics'
              AND question_concept_id = 1586140 AND person_id IN ({pids_str})
        """)
    except:
        race_df = pd.DataFrame()

    if len(race_df) > 0:

        def _race(n):
            if pd.isna(n):
                return "Other"
            n = str(n).lower()
            if "black" in n or "african" in n:
                return "Black"
            if "white" in n:
                return "White"
            if "asian" in n:
                return "Asian"
            if "hawaiian" in n or "pacific" in n:
                return "Native_Hawaiian_PI"
            if "american indian" in n or "alaska" in n:
                return "AIAN"
            if "hispanic" in n or "latino" in n:
                return "Hispanic"
            return "Other"

        race_df["race"] = race_df.race_name.apply(_race)
        demo = demo.merge(
            race_df.groupby("person_id").race.first().reset_index(),
            on="person_id",
            how="left",
        )
    else:
        race_map = {
            8516: "Black",
            8515: "Asian",
            8527: "White",
            8557: "Native_Hawaiian_PI",
            8657: "AIAN",
        }
        demo["race"] = demo.race_concept_id.map(race_map).fillna("Other")
    demo.race = demo.race.fillna("Other")

    eth_map = {38003563: "Hispanic", 38003564: "Not_Hispanic"}
    demo["ethnicity"] = demo.ethnicity_concept_id.map(eth_map).fillna("Unknown")
    demo = demo.merge(eligible[["person_id", "ici_index_date"]], on="person_id")
    demo["age_at_ici"] = demo.ici_index_date.dt.year - demo.year_of_birth
    demo["age_group"] = demo.age_at_ici.apply(age_group)

    # eGFR
    print("  Computing baseline eGFR (CKD-EPI 2021 race-free)...")
    egfr_df = demo[["person_id", "sex_at_birth", "age_at_ici"]].merge(
        eligible[["person_id", "baseline_cr"]], on="person_id"
    )
    egfr_df["baseline_egfr"] = compute_egfr(
        egfr_df.baseline_cr, egfr_df.age_at_ici, egfr_df.sex_at_birth == "Female"
    )
    n_ckd3 = (egfr_df.baseline_egfr < 60).sum()
    print(
        f"    eGFR: median={egfr_df.baseline_egfr.median():.1f}, "
        f"IQR=[{egfr_df.baseline_egfr.quantile(.25):.1f}, "
        f"{egfr_df.baseline_egfr.quantile(.75):.1f}], "
        f"<60 (CKD >=3): {n_ckd3} ({n_ckd3/len(egfr_df)*100:.1f}%)"
    )
    demo = demo.merge(
        egfr_df[["person_id", "baseline_egfr"]], on="person_id", how="left"
    )

    print(f"  Demographics: {len(demo):,}")
    for col in ["sex_at_birth", "race", "ethnicity", "age_group"]:
        print(f"    {col}:")
        for val, cnt in demo[col].value_counts().items():
            print(f"      {val:30s} {cnt:>5,}  ({cnt/len(demo)*100:.1f}%)")
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

    # ── STEP 3: NCI-CCI ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3: NCI Charlson Comorbidity Index (v2 -- corrected scoring)")
    print("=" * 70)
    dx_all = q(f"""
        SELECT co.person_id, UPPER(REPLACE(c.concept_code,'.','')) AS icd_code
        FROM `{CDR}.condition_occurrence` co
        JOIN `{CDR}.concept` c ON c.concept_id = co.condition_source_concept_id
        WHERE c.vocabulary_id IN ('ICD9CM','ICD10CM') AND co.person_id IN ({pids_str})
    """)
    print(f"  Diagnosis records: {len(dx_all):,}")
    charlson = build_nci_cci(eligible.person_id.values, dx_all, "icd_code")
    # QC
    print(
        f"  QC: {((charlson.Diabetes==1)&(charlson.Diabetes_Complicated==1)).sum()} "
        f"pts w/ both diabetes flags (raw, NOT zeroed)"
    )
    print(
        f"  QC: {((charlson.Acute_MI==1)&(charlson.History_MI==1)).sum()} "
        f"pts w/ both MI types -> Charlson MI = 1pt (OR, not 2)"
    )
    print(
        f"  QC: {((charlson.Paralysis==1)&(charlson.Cerebrovascular_Disease==1)).sum()} "
        f"pts w/ paralysis+CVD -> both score independently"
    )
    print(
        f"  Charlson score: median {charlson.charlson_score.median():.0f}, "
        f"IQR {charlson.charlson_score.quantile(.25):.0f}-"
        f"{charlson.charlson_score.quantile(.75):.0f}, "
        f"max {charlson.charlson_score.max():.0f}"
    )
    print(
        f"  NCI index: median {charlson.nci_index.median():.3f}, "
        f"max {charlson.nci_index.max():.3f}"
    )
    for c in NCI_CCI_CONDITIONS:
        print(f"    {c:40s} {charlson[c].sum():>6,}  ({charlson[c].mean()*100:.1f}%)")
    save(charlson, "03_nci_charlson.csv")

    # ── STEP 4: SDoH ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: SDoH (Basics Survey)")
    print("=" * 70)
    sdoh_ids = list(SDOH_CONCEPTS.values())
    sdoh_raw = q(f"""
        SELECT o.person_id, o.observation_source_concept_id,
               o.value_source_concept_id,
               vc.concept_name AS value_name
        FROM `{CDR}.observation` o
        LEFT JOIN `{CDR}.concept` vc ON vc.concept_id = o.value_source_concept_id
        WHERE o.observation_source_concept_id IN ({','.join(str(x) for x in sdoh_ids)})
          AND o.person_id IN ({pids_str})
    """)
    print(f"  SDoH observations: {len(sdoh_raw):,}")

    classifiers = {
        "insurance_type": (43528428, classify_insurance),
        "income": (1585375, classify_income),
        "education": (1585940, classify_education),
        "employment": (1585952, classify_employment),
        "housing": (1585370, classify_housing),
        "housing_stability": (1585886, classify_stability),
    }
    sdoh = pd.DataFrame({"person_id": eligible.person_id.values})
    for col, (cid, fn) in classifiers.items():
        sub = sdoh_raw[sdoh_raw.observation_source_concept_id == cid].copy()
        sub[col] = sub.value_name.apply(fn)
        sub_final = sub.groupby("person_id")[col].first().reset_index()
        sdoh = sdoh.merge(sub_final, on="person_id", how="left")
        sdoh[col] = sdoh[col].astype(object).fillna("Unknown")

    print(f"  SDoH assembled: {len(sdoh):,}")
    for col in SDOH_CONCEPTS:
        non_unk = (sdoh[col] != "Unknown").sum()
        print(f"    {col:25s} known: {non_unk:>5,} ({non_unk/len(sdoh)*100:.1f}%)")
    save(sdoh, "04_sdoh.csv")

    # ── STEP 5: Cancer + ICI + Nephrotoxins ──────────────────────
    print("\n" + "=" * 70)
    print("STEP 5: Cancer Type + ICI Class + Nephrotoxins")
    print("=" * 70)
    cancer_dx = q(f"""
        SELECT co.person_id, c.concept_code
        FROM `{CDR}.condition_occurrence` co
        JOIN `{CDR}.concept` c ON c.concept_id = co.condition_source_concept_id
        WHERE c.vocabulary_id = 'ICD10CM' AND c.concept_code LIKE 'C%'
          AND co.person_id IN ({pids_str})
    """)
    cancer_dx["cancer_type"] = cancer_dx.concept_code.apply(classify_cancer)
    cancer_primary = (
        cancer_dx.groupby("person_id")
        .cancer_type.agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )

    ici_reg = ici_cb[["person_id", "ici_drugs"]].copy()
    ici_reg["ici_regimen"] = ici_reg.ici_drugs.apply(classify_ici)

    cov = eligible[["person_id"]].merge(cancer_primary, on="person_id", how="left")
    cov = cov.merge(ici_reg[["person_id", "ici_regimen"]], on="person_id", how="left")
    cov.cancer_type = cov.cancer_type.fillna("Unknown")
    cov.ici_regimen = cov.ici_regimen.fillna("unknown")

    # Nephrotoxins ([-90, 0] pre-ICI)
    ici_id_str = ",".join(str(x) for x in ici_ids)
    for drug_class, agents in NEPHROTOXIN_CLASSES.items():
        alikes = " OR ".join([f"LOWER(c.concept_name) LIKE '%{a}%'" for a in agents])
        neph_pts = q(f"""
            WITH ici_dates AS (
              SELECT person_id, MIN(drug_exposure_start_date) AS ici_date
              FROM `{CDR}.drug_exposure`
              WHERE drug_concept_id IN ({ici_id_str}) GROUP BY person_id)
            SELECT DISTINCT de.person_id
            FROM `{CDR}.drug_exposure` de
            JOIN `{CDR}.concept` c ON c.concept_id = de.drug_concept_id
            JOIN ici_dates i ON i.person_id = de.person_id
            WHERE ({alikes}) AND de.person_id IN ({pids_str})
              AND de.drug_exposure_start_date
                  BETWEEN DATE_SUB(i.ici_date, INTERVAL 90 DAY) AND i.ici_date
        """)
        cov[drug_class] = cov.person_id.isin(neph_pts.person_id).astype(int)
        print(
            f"  {drug_class}: {cov[drug_class].sum():,} ({cov[drug_class].mean()*100:.1f}%)"
        )

    # v5 collapse
    cov["cancer_type_collapsed"] = cov.cancer_type.apply(collapse_cancer)
    cov["ici_collapsed"] = cov.ici_regimen.apply(collapse_ici)

    print(f"  Cancer types: {cov.cancer_type.value_counts().to_dict()}")
    print(f"  ICI regimens: {cov.ici_regimen.value_counts().to_dict()}")
    save(cov, "05_covariates.csv")

    # ── STEP 6: Matching vars ────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 6: Matching Variables")
    print("=" * 70)
    match_vars = q(f"""
        SELECT person_id,
               COUNT(DISTINCT condition_concept_id) AS n_diagnoses,
               MIN(condition_start_date) AS first_dx,
               MAX(condition_start_date) AS last_dx
        FROM `{CDR}.condition_occurrence`
        WHERE person_id IN ({pids_str}) GROUP BY person_id
    """)
    match_vars["first_dx"] = parse_date(match_vars["first_dx"])
    match_vars["last_dx"] = parse_date(match_vars["last_dx"])
    match_vars["ehr_length_days"] = (match_vars.last_dx - match_vars.first_dx).dt.days
    ref = match_vars.first_dx.min()
    match_vars["enrollment_days"] = (match_vars.first_dx - ref).dt.days
    save(
        match_vars[["person_id", "n_diagnoses", "ehr_length_days", "enrollment_days"]],
        "06_matching_variables.csv",
    )

    # ── STEP 7: Regression base ──────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 7: Regression Base Assembly")
    print("=" * 70)
    reg_cols = [
        "person_id",
        "severity",
        "severity_12m",
        "aki_365d",
        "ici_index_date",
        "baseline_cr",
        "max_cr_ratio",
        "max_delta_cr",
        "aki_delta03",
        "aki_kdigo2",
        "aki_kdigo3",
        "evt_source",
    ]
    reg = eligible[[c for c in reg_cols if c in eligible.columns]].copy()
    reg = reg.merge(
        demo[
            [
                "person_id",
                "sex_at_birth",
                "race",
                "ethnicity",
                "age_group",
                "baseline_egfr",
            ]
        ],
        on="person_id",
    )
    reg = reg.merge(charlson, on="person_id")
    reg = reg.merge(cov, on="person_id")
    reg = reg.merge(sdoh, on="person_id")
    reg = reg.merge(
        match_vars[["person_id", "enrollment_days", "n_diagnoses", "ehr_length_days"]],
        on="person_id",
    )
    print(f"  Regression base: {len(reg):,} rows, {reg.shape[1]} cols")
    print(f"  Cases: {reg.severity.sum():,}  Controls: {(reg.severity==0).sum():,}")
    print(
        f"  Charlson score: median {reg.charlson_score.median():.0f}, "
        f"IQR {reg.charlson_score.quantile(.25):.0f}-{reg.charlson_score.quantile(.75):.0f}"
    )
    print(f"  NCI index: median {reg.nci_index.median():.3f}")
    print(
        f"  eGFR: median {reg.baseline_egfr.median():.1f}, "
        f"<60: {(reg.baseline_egfr<60).sum()} ({(reg.baseline_egfr<60).mean()*100:.1f}%)"
    )
    save(reg, "07_pre_matching_base.csv")

    print("\n" + "=" * 70)
    print("AoU ETL COMPLETE (v6)")
    print("=" * 70)
    print(f"  Next: Rscript 01b_psm.R aou && Rscript 02_models.R aou")


# =====================================================================
# MAIN: INPC PIPELINE
# =====================================================================
def run_inpc():
    consort = {}

    print("=" * 70)
    print("POST-ICI AKI -- INPC ETL (v6)")
    print("=" * 70)
    print(f"  Data: {INPC_DATA}")
    print(f"  Output: {RESULTS}/")

    # Load tables
    print("\n  Loading core tables...")
    person = pd.read_csv(f"{INPC_DATA}/r6335_person.csv", low_memory=False)
    consort["total_inpc"] = len(person)
    print(f"  person: {len(person):,}")

    drug_raw = pd.read_csv(
        f"{INPC_DATA}/r6335_drug_exposure.csv",
        low_memory=False,
        usecols=[
            "person_id",
            "drug_concept_id",
            "drug_source_concept_id",
            "drug_exposure_start_date",
            "drug_source_value",
        ],
    )
    print(f"  drug_exposure: {len(drug_raw):,}")

    concept_tbl = pd.read_csv(
        f"{INPC_DATA}/r6335_concept.csv",
        encoding="cp1252",
        low_memory=False,
        usecols=["concept_id", "concept_name", "vocabulary_id"],
    )
    print(f"  concept: {len(concept_tbl):,}")

    cond = pd.read_csv(
        f"{INPC_DATA}/r6335_condition_occurrence.csv",
        low_memory=False,
        usecols=[
            "person_id",
            "condition_concept_id",
            "condition_start_date",
            "condition_source_value",
            "visit_occurrence_id",
        ],
    )
    print(f"  condition_occurrence: {len(cond):,}")

    meas = pd.read_csv(f"{INPC_DATA}/r6335_measurement.csv", low_memory=False)
    meas.columns = [c.lower() for c in meas.columns]
    print(f"  measurement: {len(meas):,}")

    # v6: load visit_occurrence for severe visit filtering
    visit_path = f"{INPC_DATA}/r6335_visit_occurrence.csv"
    if os.path.exists(visit_path):
        visit_raw = pd.read_csv(
            visit_path,
            low_memory=False,
            usecols=["visit_occurrence_id", "person_id", "visit_concept_id"],
        )
        visit_raw["visit_concept_id"] = pd.to_numeric(
            visit_raw.visit_concept_id, errors="coerce"
        )
        severe_visits = set(
            visit_raw[
                visit_raw.visit_concept_id.isin(SEVERE_VISIT_CONCEPTS)
            ].visit_occurrence_id
        )
        print(
            f"  visit_occurrence: {len(visit_raw):,} (severe: {len(severe_visits):,})"
        )
        del visit_raw
    else:
        severe_visits = None
        print("  visit_occurrence: NOT FOUND — accepting all AKI ICD codes")

    # ── STEP 1: ICI Cohort ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 1: ICI Cohort + AKI Phenotyping")
    print("=" * 70)

    ici_concept = concept_tbl[
        concept_tbl.concept_name.str.lower().apply(
            lambda x: any(a in str(x) for a in ICI_AGENTS) if pd.notna(x) else False
        )
    ]
    ici_cids = set(ici_concept.concept_id.tolist())
    print(f"  ICI concept IDs from concept table: {len(ici_cids)}")

    ici_by_cid = drug_raw[drug_raw.drug_concept_id.isin(ici_cids)]
    drug_raw["drug_source_concept_id"] = pd.to_numeric(
        drug_raw["drug_source_concept_id"], errors="coerce"
    )
    ici_by_src = drug_raw[drug_raw.drug_source_concept_id.isin(ici_cids)]
    ici_by_sv = drug_raw[
        drug_raw.drug_source_value.str.lower().apply(
            lambda x: any(a in str(x) for a in ICI_AGENTS) if pd.notna(x) else False
        )
    ]
    print(
        f"    drug_concept_id matches: {len(ici_by_cid):,} records, {ici_by_cid.person_id.nunique():,} patients"
    )
    print(
        f"    drug_source_concept_id matches: {len(ici_by_src):,} records, {ici_by_src.person_id.nunique():,} patients"
    )
    print(
        f"    drug_source_value keyword matches: {len(ici_by_sv):,} records, {ici_by_sv.person_id.nunique():,} patients"
    )

    ici_drug = pd.concat([ici_by_cid, ici_by_src, ici_by_sv]).drop_duplicates()
    print(f"  ICI drug records (union): {len(ici_drug):,}")
    ici_drug["drug_exposure_start_date"] = parse_date(
        ici_drug["drug_exposure_start_date"]
    )
    ici_patients = (
        ici_drug.groupby("person_id")
        .agg(ici_index_date=("drug_exposure_start_date", "min"))
        .reset_index()
    )

    # Resolve drug names
    dl = ici_drug.merge(
        concept_tbl[["concept_id", "concept_name"]].rename(
            columns={"concept_name": "n1"}
        ),
        left_on="drug_concept_id",
        right_on="concept_id",
        how="left",
    ).drop(columns=["concept_id"], errors="ignore")
    dl = dl.merge(
        concept_tbl[["concept_id", "concept_name"]].rename(
            columns={"concept_name": "n2"}
        ),
        left_on="drug_source_concept_id",
        right_on="concept_id",
        how="left",
    ).drop(columns=["concept_id"], errors="ignore")
    dl["resolved"] = dl.n1.fillna(dl.n2).fillna(dl.drug_source_value)
    drug_names = (
        dl.groupby("person_id")
        .resolved.apply(lambda x: list(x.dropna().str.lower().unique()))
        .reset_index()
    )
    drug_names.columns = ["person_id", "ici_drugs"]
    ici_patients = ici_patients.merge(drug_names, on="person_id", how="left")
    consort["ici_treated"] = len(ici_patients)
    print(f"  ICI-treated patients: {len(ici_patients):,}")

    # ICI class QC
    for label, agents in [
        ("anti-PD-1", ["nivolumab", "pembrolizumab", "cemiplimab", "dostarlimab"]),
        ("anti-PD-L1", ["atezolizumab", "durvalumab", "avelumab"]),
        ("anti-CTLA-4", ["ipilimumab", "tremelimumab"]),
    ]:
        n = ici_patients.ici_drugs.apply(
            lambda x: (
                any(a in " ".join(str(d) for d in x) for a in agents)
                if isinstance(x, list)
                else False
            )
        ).sum()
        print(f"    {label}:  {n:,}")

    # Cancer filter
    cond["icd_raw"] = cond.condition_source_value.str.extract(
        r"\^\^(.+)$", expand=False
    )
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

    # Creatinine
    cr = meas[meas.measurement_concept_id == 3016723].copy()
    cr = cr[
        cr.value_as_number.notna()
        & (cr.value_as_number >= 0.1)
        & (cr.value_as_number < 30)
    ]
    cr["measurement_date"] = parse_date(cr["measurement_date"])
    cr = cr[cr.person_id.isin(ici_cancer.person_id)]
    print(f"  Creatinine measurements (ICI+cancer): {len(cr):,}")

    cr_merged = cr.merge(ici_cancer[["person_id", "ici_index_date"]], on="person_id")
    cr_merged["days_from_ici"] = (
        cr_merged.measurement_date - cr_merged.ici_index_date
    ).dt.days

    baseline, followup, n_pri, n_fb = compute_baseline_followup(
        cr_merged, ici_cancer.person_id
    )
    print(f"  Patients with baseline Cr: {len(baseline):,}")
    print(f"    Primary [-365, -7] median: {n_pri:,}")
    print(f"    Fallback [-365, -1] last:  {n_fb:,}")
    consort["has_baseline_cr"] = len(baseline)
    consort["baseline_cr_primary"] = n_pri
    consort["baseline_cr_fallback"] = n_fb
    print(f"  Patients with follow-up Cr: {len(followup):,}")
    consort["has_followup_cr"] = len(followup)

    eligible = ici_cancer.merge(baseline, on="person_id").merge(
        followup, on="person_id"
    )

    # ESKD
    eskd_codes = ["N186", "Z992", "Z490", "Z491", "Z492", "Z940"]
    eskd_mask = cond.icd_clean.apply(
        lambda x: (
            any(str(x).startswith(c) for c in eskd_codes) if pd.notna(x) else False
        )
    )
    eskd_pts = cond[eskd_mask].person_id.unique()
    pre = len(eligible)
    eligible = eligible[~eligible.person_id.isin(eskd_pts)]
    eligible = eligible[eligible.baseline_cr < 4.0]
    consort["pre_eskd_exclusion"] = pre
    consort["excluded_eskd"] = pre - len(eligible)
    consort["excluded_no_baseline"] = consort["ici_cancer"] - consort["has_baseline_cr"]
    print(f"  Excluded ESKD/transplant/baseline>=4: {pre - len(eligible)}")

    # Cr-based washout
    pre_cr = cr_merged[
        (cr_merged.days_from_ici >= -90) & (cr_merged.days_from_ici <= 0)
    ]
    pre_cr = pre_cr.merge(eligible[["person_id", "baseline_cr"]], on="person_id")
    washout_cr = set(
        pre_cr[pre_cr.value_as_number / pre_cr.baseline_cr >= 1.5].person_id
    )
    print(f"  Washout Cr-based: {len(washout_cr)}")

    # v6: AKI ICD codes from condition_occurrence
    elig_pids = set(eligible.person_id)
    aki_icd_mask = cond.icd_clean.apply(
        lambda x: (
            any(str(x).startswith(p) for p in AKI_ICD_NORM) if pd.notna(x) else False
        )
    )
    aki_icd_df = cond[aki_icd_mask & cond.person_id.isin(elig_pids)][
        ["person_id", "condition_start_date", "visit_occurrence_id"]
    ].copy()
    print(
        f"  AKI ICD records: {len(aki_icd_df):,}, patients: {aki_icd_df.person_id.nunique():,}"
    )

    icd_first, washout_icd = compute_icd_aki(aki_icd_df, eligible, severe_visits)
    print(f"  Washout ICD-based: {len(washout_icd)}")
    print(f"  ICD AKI events (post-ICI): {len(icd_first):,}")

    # v6: Union washout
    washout_all = washout_cr | washout_icd
    eligible = eligible[~eligible.person_id.isin(washout_all)].copy()
    consort["excluded_washout"] = len(washout_all)
    consort["excluded_washout_cr"] = len(washout_cr)
    consort["excluded_washout_icd"] = len(washout_icd)
    consort["eligible"] = len(eligible)
    print(
        f"  Washout total: {len(washout_all)} excluded "
        f"({len(washout_cr)} Cr, {len(washout_icd)} ICD, "
        f"{len(washout_cr & washout_icd)} overlap)"
    )
    print(f"  Eligible cohort: {len(eligible):,}")

    eligible = apply_aki_phenotype(eligible, cr_merged, icd_first)
    cases = eligible.severity.sum()
    consort["cases"] = int(cases)
    consort["controls"] = int(len(eligible) - cases)
    consort["cases_12m"] = int(eligible.severity_12m.sum())
    consort["primary_window_days"] = PRIMARY_WINDOW_DAYS
    print(
        f"  Cases ({PRIMARY_WINDOW_DAYS}d primary): {cases:,} ({cases/len(eligible)*100:.1f}%)"
    )
    print(f"  Controls:          {len(eligible)-cases:,}")

    consort_df = pd.DataFrame([consort]).T.reset_index()
    consort_df.columns = ["step", "n"]
    save(consort_df, "00_consort_numbers.csv")
    print("\n  CONSORT flowchart:")
    for step, n in consort.items():
        print(f"    {step:35s} {int(n):>10,}")
    save(eligible, "01_eligible_cohort.csv")

    # ── STEP 2: Demographics ────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: Demographics")
    print("=" * 70)
    demo = person[person.person_id.isin(eligible.person_id)].copy()
    demo["sex_at_birth"] = demo.gender_concept_id.map(
        {8507: "Male", 8532: "Female"}
    ).fillna("Other")
    demo["race"] = demo.race_concept_id.map(
        {
            8516: "Black",
            8515: "Asian",
            8527: "White",
            8557: "Native_Hawaiian_PI",
            8657: "AIAN",
        }
    ).fillna("Other")
    demo["ethnicity"] = demo.ethnicity_concept_id.map(
        {38003563: "Hispanic", 38003564: "Not_Hispanic"}
    ).fillna("Unknown")
    demo = demo.merge(eligible[["person_id", "ici_index_date"]], on="person_id")
    demo["age_at_ici"] = demo.ici_index_date.dt.year - demo.year_of_birth
    demo["age_group"] = demo.age_at_ici.apply(age_group)

    print("  Computing baseline eGFR (CKD-EPI 2021 race-free)...")
    egfr_df = demo[["person_id", "sex_at_birth", "age_at_ici"]].merge(
        eligible[["person_id", "baseline_cr"]], on="person_id"
    )
    egfr_df["baseline_egfr"] = compute_egfr(
        egfr_df.baseline_cr, egfr_df.age_at_ici, egfr_df.sex_at_birth == "Female"
    )
    n_ckd3 = (egfr_df.baseline_egfr < 60).sum()
    print(
        f"    eGFR: median={egfr_df.baseline_egfr.median():.1f}, "
        f"IQR=[{egfr_df.baseline_egfr.quantile(.25):.1f}, "
        f"{egfr_df.baseline_egfr.quantile(.75):.1f}], "
        f"<60 (CKD >=3): {n_ckd3} ({n_ckd3/len(egfr_df)*100:.1f}%)"
    )
    demo = demo.merge(
        egfr_df[["person_id", "baseline_egfr"]], on="person_id", how="left"
    )

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

    # ── STEP 3: NCI-CCI ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3: NCI Charlson Comorbidity Index (v2 -- corrected scoring)")
    print("=" * 70)
    elig_cond = cond[cond.person_id.isin(eligible.person_id)].copy()
    print(f"  Diagnosis records: {len(elig_cond):,}")
    charlson = build_nci_cci(eligible.person_id.values, elig_cond, "icd_clean")
    print(
        f"  QC: {((charlson.Diabetes==1)&(charlson.Diabetes_Complicated==1)).sum()} "
        f"pts w/ both diabetes flags (raw, NOT zeroed)"
    )
    print(
        f"  QC: {((charlson.Acute_MI==1)&(charlson.History_MI==1)).sum()} "
        f"pts w/ both MI types -> Charlson MI = 1pt (OR, not 2)"
    )
    print(
        f"  QC: {((charlson.Paralysis==1)&(charlson.Cerebrovascular_Disease==1)).sum()} "
        f"pts w/ paralysis+CVD -> both score independently"
    )
    print(
        f"  Charlson score: median {charlson.charlson_score.median():.0f}, "
        f"IQR {charlson.charlson_score.quantile(.25):.0f}-"
        f"{charlson.charlson_score.quantile(.75):.0f}, "
        f"max {charlson.charlson_score.max():.0f}"
    )
    print(
        f"  NCI index: median {charlson.nci_index.median():.3f}, "
        f"max {charlson.nci_index.max():.3f}"
    )
    for c in NCI_CCI_CONDITIONS:
        print(f"    {c:40s} {charlson[c].sum():>6,}  ({charlson[c].mean()*100:.1f}%)")
    save(charlson, "03_nci_charlson.csv")

    # ── STEP 4: Cancer + ICI + Nephrotoxins ──────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: Cancer Type + Nephrotoxins")
    print("=" * 70)
    cancer_cond = cond[cond.person_id.isin(eligible.person_id) & cancer_mask].copy()
    cancer_cond["cancer_type"] = cancer_cond.icd_clean.apply(classify_cancer)
    cancer_primary = (
        cancer_cond.groupby("person_id")
        .cancer_type.agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )

    ici_reg = ici_cancer[["person_id", "ici_drugs"]].copy()
    ici_reg["ici_regimen"] = ici_reg.ici_drugs.apply(classify_ici)

    cov = eligible[["person_id"]].merge(cancer_primary, on="person_id", how="left")
    cov = cov.merge(ici_reg[["person_id", "ici_regimen"]], on="person_id", how="left")
    cov.cancer_type = cov.cancer_type.fillna("Unknown")
    cov.ici_regimen = cov.ici_regimen.fillna("unknown")

    # Nephrotoxins
    elig_drug = drug_raw[drug_raw.person_id.isin(eligible.person_id)].copy()
    elig_drug["drug_exposure_start_date"] = parse_date(
        elig_drug["drug_exposure_start_date"]
    )
    elig_drug = elig_drug.merge(
        eligible[["person_id", "ici_index_date"]], on="person_id"
    )
    elig_drug["days_from_ici"] = (
        elig_drug.drug_exposure_start_date - elig_drug.ici_index_date
    ).dt.days
    elig_drug_w = elig_drug[
        (elig_drug.days_from_ici >= -90) & (elig_drug.days_from_ici <= 0)
    ].copy()
    elig_drug_w = elig_drug_w.merge(
        concept_tbl[["concept_id", "concept_name"]],
        left_on="drug_concept_id",
        right_on="concept_id",
        how="left",
    ).drop(columns=["concept_id"], errors="ignore")

    for drug_class, agents in NEPHROTOXIN_CLASSES.items():
        by_name = elig_drug_w.concept_name.str.lower().apply(
            lambda x: any(a in str(x) for a in agents) if pd.notna(x) else False
        )
        by_src = elig_drug_w.drug_source_value.str.lower().apply(
            lambda x: any(a in str(x) for a in agents) if pd.notna(x) else False
        )
        flagged = elig_drug_w[by_name | by_src].person_id.unique()
        cov[drug_class] = cov.person_id.isin(flagged).astype(int)
        print(
            f"  {drug_class}: {cov[drug_class].sum():,} ({cov[drug_class].mean()*100:.1f}%)"
        )

    # v5 collapse
    cov["cancer_type_collapsed"] = cov.cancer_type.apply(collapse_cancer)
    cov["ici_collapsed"] = cov.ici_regimen.apply(collapse_ici)
    save(cov, "05_covariates.csv")

    # ── STEP 5: Matching vars ────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5: Matching Variables")
    print("=" * 70)
    dx_counts = (
        elig_cond.groupby("person_id")
        .agg(n_diagnoses=("condition_concept_id", "nunique"))
        .reset_index()
    )
    cond_dates = cond[cond.person_id.isin(eligible.person_id)].copy()
    cond_dates["condition_start_date"] = parse_date(cond_dates["condition_start_date"])
    ehr = (
        cond_dates.groupby("person_id")
        .agg(
            first_dx=("condition_start_date", "min"),
            last_dx=("condition_start_date", "max"),
        )
        .reset_index()
    )
    ehr["ehr_length_days"] = (ehr.last_dx - ehr.first_dx).dt.days
    ref = ehr.first_dx.min()
    ehr["enrollment_days"] = (ehr.first_dx - ref).dt.days
    match_vars = dx_counts.merge(
        ehr[["person_id", "enrollment_days", "ehr_length_days"]], on="person_id"
    )
    save(match_vars, "06_matching_variables.csv")

    # ── STEP 6: Regression base ──────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 6: Regression Base Assembly")
    print("=" * 70)
    reg_cols = [
        "person_id",
        "severity",
        "severity_12m",
        "aki_365d",
        "ici_index_date",
        "baseline_cr",
        "max_cr_ratio",
        "max_delta_cr",
        "aki_delta03",
        "aki_kdigo2",
        "aki_kdigo3",
        "evt_source",
    ]
    reg = eligible[[c for c in reg_cols if c in eligible.columns]].copy()
    reg = reg.merge(
        demo[
            [
                "person_id",
                "sex_at_birth",
                "race",
                "ethnicity",
                "age_group",
                "baseline_egfr",
            ]
        ],
        on="person_id",
    )
    reg = reg.merge(charlson, on="person_id")
    reg = reg.merge(cov, on="person_id")
    reg = reg.merge(match_vars, on="person_id")
    print(f"  Regression base: {len(reg):,} rows, {reg.shape[1]} cols")
    print(f"  Cases: {reg.severity.sum():,}  Controls: {(reg.severity==0).sum():,}")
    print(f"  Charlson score: median {reg.charlson_score.median():.0f}")
    print(f"  NCI index: median {reg.nci_index.median():.3f}")
    print(
        f"  eGFR: median {reg.baseline_egfr.median():.1f}, "
        f"<60: {(reg.baseline_egfr<60).sum()} ({(reg.baseline_egfr<60).mean()*100:.1f}%)"
    )
    save(reg, "07_pre_matching_base.csv")

    print("\n" + "=" * 70)
    print("INPC ETL COMPLETE (v6)")
    print("=" * 70)
    print(f"  Next: Rscript 01b_psm.R inpc && Rscript 02_models.R inpc")


# =====================================================================
if __name__ == "__main__":
    if MODE == "aou":
        run_aou()
    else:
        run_inpc()
