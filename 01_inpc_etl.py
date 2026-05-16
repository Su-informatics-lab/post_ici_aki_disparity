#!/usr/bin/env python3
"""
Post-ICI AKI × SDoH — INPC ETL (Clinical Model Transportability)
Reads from: /N/project/depot/hw56/irAKI_data/structured_data/

Purpose-built OMOP CSV dump of post-ICI patients from INPC.
Replicates AoU base model (demographics + NCI-CCI + cancer type +
ICI class + nephrotoxins → AKI) without SDoH (no surveys in INPC).

Output: results/inpc/*.csv  (same schema as AoU for 01b_psm.R + 02_models.R)
Usage:  python 01_inpc_etl.py
Then:   Rscript 01b_psm.R inpc
        Rscript 02_models.R inpc
"""

import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

DATA = "/N/project/depot/hw56/irAKI_data/structured_data"
RESULTS = "results/inpc"
os.makedirs(RESULTS, exist_ok=True)

print("=" * 70)
print("POST-ICI AKI — INPC ETL (Clinical Transportability)")
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
print(f"  person: {len(person):,}")

drug = pd.read_csv(
    f"{DATA}/r6335_drug_exposure.csv",
    low_memory=False,
    usecols=[
        "person_id",
        "drug_concept_id",
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


# ═══════════════════════════════════════════════════════════════════
# STEP 1: ICI COHORT + AKI PHENOTYPING
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1: ICI Cohort + AKI Phenotyping")
print("=" * 70)

# ── 1a. ICI patients ──────────────────────────────────────────────
ici_terms = [
    "nivolumab",
    "pembrolizumab",
    "cemiplimab",
    "dostarlimab",
    "atezolizumab",
    "durvalumab",
    "avelumab",
    "ipilimumab",
    "tremelimumab",
    "relatlimab",
]

drug_named = drug.merge(
    concept[["concept_id", "concept_name"]].rename(
        columns={"concept_id": "drug_concept_id"}
    ),
    on="drug_concept_id",
    how="left",
)
ici_mask = drug_named.concept_name.str.lower().str.contains(
    "|".join(ici_terms), na=False
)
ici_drugs = drug_named[ici_mask].copy()
print(f"  ICI records: {len(ici_drugs):,}")
print(f"  ICI patients: {ici_drugs.person_id.nunique():,}")

# Index date = first ICI
ici_drugs["ici_date"] = parse_date(ici_drugs["drug_exposure_start_date"])
ici_index = (
    ici_drugs.groupby("person_id").agg(ici_index_date=("ici_date", "min")).reset_index()
)
print(f"  Patients with ICI index: {len(ici_index):,}")

# ── 1b. ICI class classification ─────────────────────────────────
name_to_class = {}
for d in ["nivolumab", "pembrolizumab", "cemiplimab", "dostarlimab"]:
    name_to_class[d] = "anti_pd1"
for d in ["atezolizumab", "durvalumab", "avelumab"]:
    name_to_class[d] = "anti_pdl1"
for d in ["ipilimumab", "tremelimumab"]:
    name_to_class[d] = "anti_ctla4"
name_to_class["relatlimab"] = "anti_lag3"


def get_class(name):
    if pd.isna(name):
        return "unknown"
    name = str(name).lower()
    for key, cls in name_to_class.items():
        if key in name:
            return cls
    return "unknown"


ici_drugs["ici_class"] = ici_drugs["concept_name"].apply(get_class)
ici_drugs = ici_drugs.merge(ici_index, on="person_id")
ici_drugs["days_from_index"] = (
    ici_drugs["ici_date"] - ici_drugs["ici_index_date"]
).dt.days
ici_window = ici_drugs[ici_drugs["days_from_index"] <= 30]


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

# ── 1c. Cancer diagnosis ─────────────────────────────────────────
# Extract ICD code from condition_source_value (format: "1284^^H35.341")
cond["icd_code"] = (
    cond["condition_source_value"].astype(str).str.extract(r"\^\^(.+)$")[0]
)
cond["icd_code"] = cond["icd_code"].str.replace(".", "", regex=False).str.upper()

cancer_mask = cond["icd_code"].str.match(r"^C\d", na=False)
cancer_pts = cond[
    cancer_mask & cond.person_id.isin(ici_index.person_id)
].person_id.unique()
print(f"  ICI + cancer: {len(cancer_pts):,}")

cohort = ici_index[ici_index.person_id.isin(cancer_pts)].copy()
print(f"  Cohort (ICI + cancer): {len(cohort):,}")

# ── 1d. Creatinine (chunked read) ────────────────────────────────
print("\n  Reading measurement table (chunked, Cr only)...")
cr_rows = []
for chunk in pd.read_csv(
    f"{DATA}/r6335_measurement.csv", chunksize=5_000_000, low_memory=False
):
    chunk.columns = [c.upper() for c in chunk.columns]
    cr = chunk[chunk["MEASUREMENT_CONCEPT_ID"] == 3016723].copy()
    if len(cr) > 0:
        cr["cr_value"] = pd.to_numeric(cr["VALUE_AS_NUMBER"], errors="coerce")
        cr["measurement_date"] = parse_date(cr["MEASUREMENT_DATETIME"])
        cr = cr[cr.cr_value > 0][["PERSON_ID", "measurement_date", "cr_value"]]
        cr = cr.rename(columns={"PERSON_ID": "person_id"})
        cr = cr[cr.person_id.isin(cohort.person_id)]
        if len(cr) > 0:
            cr_rows.append(cr)

cr_all = pd.concat(cr_rows, ignore_index=True) if cr_rows else pd.DataFrame()
print(
    f"  Cr records (ICI+cancer): {len(cr_all):,}, patients: {cr_all.person_id.nunique():,}"
)

# Plausibility + unit check (all mg/dL per audit)
cr_all = cr_all[(cr_all.cr_value > 0.1) & (cr_all.cr_value < 30)].copy()

# Merge index date
cr_all = cr_all.merge(cohort[["person_id", "ici_index_date"]], on="person_id")
cr_all["days_from_index"] = (
    cr_all["measurement_date"] - cr_all["ici_index_date"]
).dt.days

# Baseline: median outpatient Cr [-365d, -7d]
# (INPC dump doesn't have visit_type easily, use all Cr as baseline)
cr_base_window = cr_all[
    (cr_all.days_from_index >= -365) & (cr_all.days_from_index <= -7)
]
baseline = (
    cr_base_window.groupby("person_id")
    .agg(
        baseline_cr=("cr_value", "median"),
        n_baseline_cr=("cr_value", "count"),
    )
    .reset_index()
)

# Fallback: most recent in [-365, -1] if no [-365, -7] value
cr_base_recent = cr_all[
    (cr_all.days_from_index >= -365) & (cr_all.days_from_index <= -1)
]
baseline_recent = (
    cr_base_recent.sort_values("measurement_date").groupby("person_id").tail(1)
)
baseline_recent = baseline_recent[["person_id", "cr_value"]].rename(
    columns={"cr_value": "baseline_cr_recent"}
)
baseline = baseline.merge(baseline_recent, on="person_id", how="outer")
baseline["baseline_cr"] = baseline["baseline_cr"].fillna(baseline["baseline_cr_recent"])

# Follow-up: [+1d, +365d]
cr_fu = cr_all[(cr_all.days_from_index >= 1) & (cr_all.days_from_index <= 365)]

print(f"  Baseline Cr: {len(baseline):,} patients")
print(f"  Follow-up Cr: {cr_fu.person_id.nunique():,} patients")

# AKI phenotyping
cr_fu = cr_fu.merge(baseline[["person_id", "baseline_cr"]], on="person_id")
cr_fu["cr_ratio"] = cr_fu["cr_value"] / cr_fu["baseline_cr"]
cr_fu["delta_cr"] = cr_fu["cr_value"] - cr_fu["baseline_cr"]

aki_pts = cr_fu[cr_fu.cr_ratio >= 1.5].person_id.unique()

max_stats = (
    cr_fu.groupby("person_id")
    .agg(
        max_cr_ratio=("cr_ratio", "max"),
        max_delta_cr=("delta_cr", "max"),
    )
    .reset_index()
)

# Build eligible cohort
eligible = cohort[
    cohort.person_id.isin(baseline.person_id)
    & cohort.person_id.isin(cr_fu.person_id.unique())
].copy()
eligible = eligible.merge(baseline[["person_id", "baseline_cr"]], on="person_id")
eligible = eligible[eligible.baseline_cr < 4.0].copy()
eligible["severity"] = eligible.person_id.isin(aki_pts).astype(int)
eligible = eligible.merge(max_stats, on="person_id", how="left")
eligible = eligible.merge(ici_regimen, on="person_id", how="left")

# Sensitivity flags
eligible["aki_delta03"] = (eligible["max_delta_cr"] >= 0.3).astype(int)
eligible["aki_kdigo2"] = eligible.person_id.isin(
    cr_fu[cr_fu.cr_ratio >= 2.0].person_id.unique()
).astype(int)
eligible["aki_kdigo3"] = eligible.person_id.isin(
    cr_fu[cr_fu.cr_ratio >= 3.0].person_id.unique()
).astype(int)
cr_180 = cr_fu[cr_fu.days_from_index <= 180]
eligible["aki_180d"] = eligible.person_id.isin(
    cr_180[cr_180.cr_ratio >= 1.5].person_id.unique()
).astype(int)

n_cases = eligible.severity.sum()
n_controls = len(eligible) - n_cases
print(f"\n  ┌─────────────────────────────────────────┐")
print(f"  │  INPC COHORT: {len(eligible):,} patients         │")
print(f"  │  AKI Cases (Cr ≥1.5×): {n_cases:,}              │")
print(f"  │  Controls: {n_controls:,}                         │")
print(f"  │  AKI rate: {n_cases/len(eligible)*100:.1f}%                       │")
print(f"  │  Regimen: {eligible.ici_regimen.value_counts().to_dict()} │")
print(f"  └─────────────────────────────────────────┘")
print(
    f"  Sensitivity: ΔCr≥0.3={eligible.aki_delta03.sum()}, ≥2.0×={eligible.aki_kdigo2.sum()}, ≥3.0×={eligible.aki_kdigo3.sum()}"
)

save(eligible, "01_ici_cohort.csv")


# ═══════════════════════════════════════════════════════════════════
# STEP 2: DEMOGRAPHICS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Demographics")
print("=" * 70)

demo = person[person.person_id.isin(eligible.person_id)].copy()
demo["sex_at_birth"] = demo.gender_concept_id.map(
    {8507: "Male", 8532: "Female"}
).fillna("Other")
demo["race"] = demo.race_concept_id.map(
    {8527: "White", 8516: "Black", 8515: "Asian"}
).fillna("Other")
demo["ethnicity"] = demo.ethnicity_concept_id.map(
    {38003564: "Not Hispanic", 38003563: "Hispanic"}
).fillna("Other")
demo = demo.merge(eligible[["person_id", "ici_index_date"]], on="person_id")
demo["age_at_ici"] = demo.ici_index_date.dt.year - demo.year_of_birth
demo["age_group"] = pd.cut(
    demo.age_at_ici,
    bins=[0, 45, 55, 65, 200],
    labels=["<45", "45-54", "55-64", "65+"],
    right=False,
)

print(f"  Sex: {demo.sex_at_birth.value_counts().to_dict()}")
print(f"  Race: {demo.race.value_counts().to_dict()}")
print(f"  Age: {demo.age_group.value_counts().to_dict()}")

save(
    demo[["person_id", "sex_at_birth", "race", "ethnicity", "age_at_ici", "age_group"]],
    "02_demographics.csv",
)


# ═══════════════════════════════════════════════════════════════════
# STEP 3: NCI CHARLSON (14 conditions)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: NCI Charlson Comorbidity Index")
print("=" * 70)

# NCI 2021 ICD code sets (same as AoU ETL)
NCI_CCI = {
    "Acute_MI": {"9": ["410"], "10": ["I21", "I22"]},
    "History_MI": {"9": ["412"], "10": ["I252"]},
    "Congestive_Heart_Failure": {
        "9": [
            "39891",
            "40201",
            "40211",
            "40291",
            "40401",
            "40403",
            "40411",
            "40413",
            "40491",
            "40493",
            "4254",
            "4255",
            "4256",
            "4257",
            "4258",
            "4259",
            "428",
        ],
        "10": [
            "I099",
            "I110",
            "I130",
            "I132",
            "I255",
            "I420",
            "I425",
            "I426",
            "I427",
            "I428",
            "I429",
            "I43",
            "I50",
            "P290",
        ],
    },
    "Peripheral_Vascular_Disease": {
        "9": [
            "0930",
            "440",
            "441",
            "4431",
            "4432",
            "4433",
            "4434",
            "4435",
            "4436",
            "4437",
            "4438",
            "4439",
            "4471",
            "5571",
            "5579",
            "V434",
        ],
        "10": [
            "I70",
            "I71",
            "I731",
            "I738",
            "I739",
            "I771",
            "I790",
            "I792",
            "K551",
            "K558",
            "K559",
            "Z958",
            "Z959",
        ],
    },
    "Cerebrovascular_Disease": {
        "9": ["36234", "430", "431", "432", "433", "434", "435", "436", "437", "438"],
        "10": [
            "G45",
            "G46",
            "H340",
            "I60",
            "I61",
            "I62",
            "I63",
            "I64",
            "I65",
            "I66",
            "I67",
            "I68",
            "I69",
        ],
    },
    "Chronic_Pulmonary_Disease": {
        "9": [
            "4168",
            "4169",
            "490",
            "491",
            "492",
            "493",
            "494",
            "495",
            "496",
            "497",
            "498",
            "499",
            "500",
            "501",
            "502",
            "503",
            "504",
            "505",
            "5064",
            "5081",
            "5088",
        ],
        "10": [
            "I278",
            "I279",
            "J40",
            "J41",
            "J42",
            "J43",
            "J44",
            "J45",
            "J46",
            "J47",
            "J60",
            "J61",
            "J62",
            "J63",
            "J64",
            "J65",
            "J66",
            "J67",
            "J684",
            "J701",
            "J703",
        ],
    },
    "Dementia": {
        "9": ["290", "2941", "3312"],
        "10": ["F00", "F01", "F02", "F03", "F051", "G30", "G311"],
    },
    "Paralysis": {
        "9": [
            "3341",
            "342",
            "343",
            "3440",
            "3441",
            "3442",
            "3443",
            "3444",
            "3445",
            "3446",
            "3449",
        ],
        "10": [
            "G041",
            "G114",
            "G801",
            "G802",
            "G81",
            "G82",
            "G830",
            "G831",
            "G832",
            "G833",
            "G834",
            "G839",
        ],
    },
    "Diabetes": {
        "9": ["2500", "2501", "2502", "2503", "2508", "2509"],
        "10": [
            "E100",
            "E101",
            "E106",
            "E108",
            "E109",
            "E110",
            "E111",
            "E116",
            "E118",
            "E119",
            "E130",
            "E131",
            "E136",
            "E138",
            "E139",
        ],
    },
    "Diabetes_Complicated": {
        "9": ["2504", "2505", "2506", "2507"],
        "10": [
            "E102",
            "E103",
            "E104",
            "E105",
            "E107",
            "E112",
            "E113",
            "E114",
            "E115",
            "E117",
            "E132",
            "E133",
            "E134",
            "E135",
            "E137",
        ],
    },
    "Renal_Disease": {
        "9": [
            "40301",
            "40311",
            "40391",
            "40402",
            "40403",
            "40412",
            "40413",
            "40492",
            "40493",
            "582",
            "5830",
            "5831",
            "5832",
            "5833",
            "5834",
            "5835",
            "5836",
            "5837",
            "585",
            "586",
            "5880",
            "V420",
            "V451",
            "V56",
        ],
        "10": [
            "I120",
            "I131",
            "N032",
            "N033",
            "N034",
            "N035",
            "N036",
            "N037",
            "N052",
            "N053",
            "N054",
            "N055",
            "N056",
            "N057",
            "N18",
            "N19",
            "N250",
            "Z490",
            "Z491",
            "Z492",
            "Z940",
            "Z992",
        ],
    },
    "Liver_Disease_Mild": {
        "9": [
            "07022",
            "07023",
            "07032",
            "07033",
            "07044",
            "07054",
            "0706",
            "0709",
            "570",
            "571",
            "5733",
            "5734",
            "5738",
            "5739",
            "V427",
        ],
        "10": [
            "B18",
            "K700",
            "K701",
            "K702",
            "K703",
            "K709",
            "K713",
            "K714",
            "K715",
            "K717",
            "K73",
            "K74",
            "K760",
            "K762",
            "K763",
            "K764",
            "K768",
            "K769",
            "Z944",
        ],
    },
    "Liver_Disease_Moderate_Severe": {
        "9": [
            "4560",
            "4561",
            "4562",
            "5722",
            "5723",
            "5724",
            "5725",
            "5726",
            "5727",
            "5728",
        ],
        "10": [
            "I850",
            "I859",
            "I864",
            "I982",
            "K704",
            "K711",
            "K721",
            "K729",
            "K765",
            "K766",
            "K767",
        ],
    },
    "Peptic_Ulcer_Disease": {
        "9": ["531", "532", "533", "534"],
        "10": ["K25", "K26", "K27", "K28"],
    },
    "Rheumatic_Disease": {
        "9": [
            "4465",
            "7100",
            "7101",
            "7102",
            "7103",
            "7104",
            "7140",
            "7141",
            "7142",
            "7148",
            "725",
        ],
        "10": ["M05", "M06", "M315", "M32", "M33", "M34", "M351", "M353", "M360"],
    },
    "AIDS": {"9": ["042"], "10": ["B20"]},
}

# Build condition flags from ICD codes in condition_occurrence
eligible_cond = cond[cond.person_id.isin(eligible.person_id)].copy()

charlson = eligible[["person_id"]].copy()
for condition, codes in NCI_CCI.items():
    all_prefixes = []
    for ver_codes in codes.values():
        all_prefixes.extend(ver_codes)
    mask = eligible_cond.icd_code.apply(
        lambda x: (
            any(str(x).startswith(p) for p in all_prefixes) if pd.notna(x) else False
        )
    )
    flagged = eligible_cond[mask].person_id.unique()
    charlson[condition] = charlson.person_id.isin(flagged).astype(int)

# Hierarchies
charlson.loc[charlson.Liver_Disease_Moderate_Severe == 1, "Liver_Disease_Mild"] = 0
charlson.loc[charlson.Diabetes_Complicated == 1, "Diabetes"] = 0

# NCI-CCI score
NCI_WEIGHTS = {
    "Acute_MI": 1,
    "History_MI": 1,
    "Congestive_Heart_Failure": 1,
    "Peripheral_Vascular_Disease": 1,
    "Cerebrovascular_Disease": 1,
    "Chronic_Pulmonary_Disease": 1,
    "Dementia": 1,
    "Paralysis": 2,
    "Diabetes": 1,
    "Diabetes_Complicated": 2,
    "Renal_Disease": 2,
    "Liver_Disease_Mild": 1,
    "Liver_Disease_Moderate_Severe": 3,
    "Peptic_Ulcer_Disease": 1,
    "Rheumatic_Disease": 1,
    "AIDS": 6,
}
charlson["nci_cci_score"] = sum(
    charlson[c] * w for c, w in NCI_WEIGHTS.items() if c in charlson.columns
)

print(
    f"  NCI-CCI score: median {charlson.nci_cci_score.median():.0f}, "
    f"IQR {charlson.nci_cci_score.quantile(0.25):.0f}–{charlson.nci_cci_score.quantile(0.75):.0f}"
)
for c in NCI_CCI:
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
cancer_cond["days_to_ici"] = abs(
    (cancer_cond.condition_start_date - cancer_cond.ici_index_date).dt.days
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
    if code.startswith("C22"):
        return "Hepatocellular"
    if any(code.startswith(p) for p in ["C18", "C19", "C20"]):
        return "Colorectal"
    if any(
        code.startswith(p)
        for p in [
            "C00",
            "C01",
            "C02",
            "C03",
            "C04",
            "C05",
            "C06",
            "C07",
            "C08",
            "C09",
            "C10",
            "C11",
            "C12",
            "C13",
            "C14",
            "C30",
            "C31",
            "C32",
        ]
    ):
        return "Head_Neck"
    if code >= "C81" and code < "C97":
        return "Hematologic"
    return "Other_Solid"


cancer_cond["cancer_type"] = cancer_cond.icd_code.apply(classify_cancer)
cancer_primary = cancer_cond.sort_values("days_to_ici").groupby("person_id").first()
cancer_primary = cancer_primary[["cancer_type"]].reset_index()
cancer_primary["cancer_type_collapsed"] = cancer_primary.cancer_type.apply(
    lambda x: x if x in ["Lung", "Melanoma"] else "Other"
)

print(f"  Cancer types: {cancer_primary.cancer_type.value_counts().to_dict()}")

# Nephrotoxins
nephrotoxin_terms = {
    "ppi_flag": [
        "omeprazole",
        "pantoprazole",
        "lansoprazole",
        "esomeprazole",
        "rabeprazole",
    ],
    "nsaid_flag": [
        "ibuprofen",
        "naproxen",
        "diclofenac",
        "celecoxib",
        "meloxicam",
        "indomethacin",
        "ketorolac",
    ],
    "acei_arb_flag": [
        "lisinopril",
        "enalapril",
        "ramipril",
        "losartan",
        "valsartan",
        "irbesartan",
        "olmesartan",
        "candesartan",
        "benazepril",
    ],
    "diuretic_flag": [
        "furosemide",
        "hydrochlorothiazide",
        "bumetanide",
        "torsemide",
        "spironolactone",
        "chlorthalidone",
    ],
}

drug_elig = drug_named[drug_named.person_id.isin(eligible.person_id)]
nephro = eligible[["person_id"]].copy()
for flag, terms in nephrotoxin_terms.items():
    pattern = "|".join(terms)
    mask = drug_elig.concept_name.str.lower().str.contains(pattern, na=False)
    flagged = drug_elig[mask].person_id.unique()
    nephro[flag] = nephro.person_id.isin(flagged).astype(int)
    print(f"  {flag}: {nephro[flag].sum():,} ({nephro[flag].mean()*100:.1f}%)")

covariates = cancer_primary.merge(nephro, on="person_id")
save(covariates, "05_cancer_nephrotoxins.csv")


# ═══════════════════════════════════════════════════════════════════
# STEP 5: MATCHING VARIABLES + REGRESSION BASE
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: Matching Variables + Regression Base")
print("=" * 70)

# Matching vars: enrollment proxy, dx count, EHR length
dx_counts = (
    cond[cond.person_id.isin(eligible.person_id)]
    .groupby("person_id")
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

# Build regression base
reg = eligible[
    [
        "person_id",
        "severity",
        "ici_index_date",
        "ici_regimen",
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
    demo[["person_id", "sex_at_birth", "race", "ethnicity", "age_group"]],
    on="person_id",
)
reg = reg.merge(charlson, on="person_id")
reg = reg.merge(covariates, on="person_id")
reg = reg.merge(match_vars, on="person_id")

# Collapsed factors
reg["ici_collapsed"] = reg.ici_regimen.apply(
    lambda x: "anti_pd1" if x == "anti_pd1" else "other_combo"
)

print(f"  Regression base: {len(reg):,} rows, {reg.shape[1]} cols")
print(f"  Cases: {reg.severity.sum():,}  Controls: {(reg.severity==0).sum():,}")
print(f"  NCI-CCI score: median {reg.nci_cci_score.median():.0f}")
save(reg, "07_pre_matching_base.csv")


# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("INPC ETL COMPLETE")
print("=" * 70)
print(f"  Output: {RESULTS}/")
print(f"  Next:   Rscript 01b_psm.R inpc")
print(f"          Rscript 02_models.R inpc")
