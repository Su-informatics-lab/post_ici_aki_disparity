#!/usr/bin/env python3
"""
Post-ICI AKI — Concept ID Discovery & Feasibility Check  (v2)
Run on AoU Researcher Workbench FIRST, before the full ETL.

v2 changes:
  - Full 13-agent ICI list (added retifanlimab, toripalimab, tislelizumab)
  - Creatinine UNIT audit (mg/dL vs umol/L detection)

Copy-paste the FULL output back to Claude.
"""

import os
import pandas as pd

CDR = os.environ["WORKSPACE_CDR"]
print(f"CDR: {CDR}")


def q(sql, label=""):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    df = pd.read_gbq(sql, dialect="standard")
    print(df.to_string(index=False))
    print(f"  [{len(df)} rows]")
    return df


# Full 13-agent ICI list
ICI_DRUGS = [
    "nivolumab", "pembrolizumab", "cemiplimab", "dostarlimab",
    "retifanlimab", "toripalimab", "tislelizumab",
    "atezolizumab", "durvalumab", "avelumab",
    "ipilimumab", "tremelimumab",
    "relatlimab",
]

ICI_LIKE = " OR ".join([f"LOWER(c.concept_name) LIKE '%{d}%'" for d in ICI_DRUGS])


# ===============================================================
# 1. ICI DRUG CONCEPT IDS (all 13 agents)
# ===============================================================
print("\n" + "#"*70)
print("# SECTION 1: ICI DRUG CONCEPT IDS")
print("#"*70)

for drug in ICI_DRUGS:
    q(f"""
    SELECT de.drug_concept_id, c.concept_name, c.vocabulary_id,
           c.concept_class_id, COUNT(*) AS n_records,
           COUNT(DISTINCT de.person_id) AS n_patients
    FROM `{CDR}`.drug_exposure de
    JOIN `{CDR}`.concept c ON c.concept_id = de.drug_concept_id
    WHERE LOWER(c.concept_name) LIKE '%{drug}%'
    GROUP BY 1,2,3,4
    ORDER BY n_patients DESC
    LIMIT 20
    """, f"ICI: {drug}")

print("\n--- drug_source_concept_id check ---")
for drug in ICI_DRUGS:
    q(f"""
    SELECT de.drug_source_concept_id, c.concept_name, c.vocabulary_id,
           COUNT(DISTINCT de.person_id) AS n_patients
    FROM `{CDR}`.drug_exposure de
    JOIN `{CDR}`.concept c ON c.concept_id = de.drug_source_concept_id
    WHERE LOWER(c.concept_name) LIKE '%{drug}%'
    GROUP BY 1,2,3
    ORDER BY n_patients DESC
    LIMIT 10
    """, f"ICI source: {drug}")


# ===============================================================
# 2. NEPHROTOXIN CONCEPT IDS
# ===============================================================
print("\n" + "#"*70)
print("# SECTION 2: NEPHROTOXIN CONCEPT IDS")
print("#"*70)

q(f"""
SELECT de.drug_concept_id, c.concept_name, c.concept_class_id,
       COUNT(DISTINCT de.person_id) AS n_patients
FROM `{CDR}`.drug_exposure de
JOIN `{CDR}`.concept c ON c.concept_id = de.drug_concept_id
WHERE LOWER(c.concept_name) LIKE '%omeprazole%'
   OR LOWER(c.concept_name) LIKE '%pantoprazole%'
   OR LOWER(c.concept_name) LIKE '%lansoprazole%'
   OR LOWER(c.concept_name) LIKE '%esomeprazole%'
   OR LOWER(c.concept_name) LIKE '%rabeprazole%'
   OR LOWER(c.concept_name) LIKE '%dexlansoprazole%'
GROUP BY 1,2,3 ORDER BY n_patients DESC LIMIT 30
""", "PPIs")

q(f"""
SELECT de.drug_concept_id, c.concept_name, c.concept_class_id,
       COUNT(DISTINCT de.person_id) AS n_patients
FROM `{CDR}`.drug_exposure de
JOIN `{CDR}`.concept c ON c.concept_id = de.drug_concept_id
WHERE LOWER(c.concept_name) LIKE '%ibuprofen%'
   OR LOWER(c.concept_name) LIKE '%naproxen%'
   OR LOWER(c.concept_name) LIKE '%diclofenac%'
   OR LOWER(c.concept_name) LIKE '%celecoxib%'
   OR LOWER(c.concept_name) LIKE '%meloxicam%'
   OR LOWER(c.concept_name) LIKE '%indomethacin%'
   OR LOWER(c.concept_name) LIKE '%ketorolac%'
GROUP BY 1,2,3 ORDER BY n_patients DESC LIMIT 30
""", "NSAIDs")

q(f"""
SELECT de.drug_concept_id, c.concept_name, c.concept_class_id,
       COUNT(DISTINCT de.person_id) AS n_patients
FROM `{CDR}`.drug_exposure de
JOIN `{CDR}`.concept c ON c.concept_id = de.drug_concept_id
WHERE LOWER(c.concept_name) LIKE '%lisinopril%'
   OR LOWER(c.concept_name) LIKE '%enalapril%'
   OR LOWER(c.concept_name) LIKE '%ramipril%'
   OR LOWER(c.concept_name) LIKE '%losartan%'
   OR LOWER(c.concept_name) LIKE '%valsartan%'
   OR LOWER(c.concept_name) LIKE '%irbesartan%'
   OR LOWER(c.concept_name) LIKE '%olmesartan%'
   OR LOWER(c.concept_name) LIKE '%candesartan%'
   OR LOWER(c.concept_name) LIKE '%telmisartan%'
   OR LOWER(c.concept_name) LIKE '%benazepril%'
GROUP BY 1,2,3 HAVING n_patients >= 50 ORDER BY n_patients DESC LIMIT 30
""", "ACEi/ARBs")

q(f"""
SELECT de.drug_concept_id, c.concept_name, c.concept_class_id,
       COUNT(DISTINCT de.person_id) AS n_patients
FROM `{CDR}`.drug_exposure de
JOIN `{CDR}`.concept c ON c.concept_id = de.drug_concept_id
WHERE LOWER(c.concept_name) LIKE '%furosemide%'
   OR LOWER(c.concept_name) LIKE '%hydrochlorothiazide%'
   OR LOWER(c.concept_name) LIKE '%bumetanide%'
   OR LOWER(c.concept_name) LIKE '%torsemide%'
   OR LOWER(c.concept_name) LIKE '%spironolactone%'
   OR LOWER(c.concept_name) LIKE '%chlorthalidone%'
GROUP BY 1,2,3 HAVING n_patients >= 50 ORDER BY n_patients DESC LIMIT 30
""", "Diuretics")


# ===============================================================
# 3. SERUM CREATININE — CONCEPTS + UNIT AUDIT
# ===============================================================
print("\n" + "#"*70)
print("# SECTION 3: CREATININE CONCEPTS + UNIT AUDIT")
print("#"*70)

q(f"""
SELECT m.measurement_concept_id, c.concept_name, c.concept_code,
       c.vocabulary_id, COUNT(*) AS n_records,
       COUNT(DISTINCT m.person_id) AS n_patients
FROM `{CDR}`.measurement m
JOIN `{CDR}`.concept c ON c.concept_id = m.measurement_concept_id
WHERE (LOWER(c.concept_name) LIKE '%creatinine%'
       AND LOWER(c.concept_name) NOT LIKE '%clearance%'
       AND LOWER(c.concept_name) NOT LIKE '%ratio%'
       AND LOWER(c.concept_name) NOT LIKE '%urine%')
   OR c.concept_code = '2160-0'
GROUP BY 1,2,3,4
ORDER BY n_patients DESC LIMIT 15
""", "3a. Creatinine measurement concepts")

q(f"""
SELECT
  u.concept_name AS unit_concept_name,
  m.unit_source_value,
  m.unit_concept_id,
  COUNT(*) AS n_records,
  COUNT(DISTINCT m.person_id) AS n_patients,
  MIN(m.value_as_number) AS min_val,
  APPROX_QUANTILES(m.value_as_number, 4)[OFFSET(1)] AS q25,
  APPROX_QUANTILES(m.value_as_number, 4)[OFFSET(2)] AS median_val,
  APPROX_QUANTILES(m.value_as_number, 4)[OFFSET(3)] AS q75,
  MAX(m.value_as_number) AS max_val
FROM `{CDR}`.measurement m
LEFT JOIN `{CDR}`.concept u ON u.concept_id = m.unit_concept_id
WHERE m.measurement_concept_id IN (
    SELECT concept_id FROM `{CDR}`.concept
    WHERE concept_code = '2160-0' AND vocabulary_id = 'LOINC'
) AND m.value_as_number IS NOT NULL AND m.value_as_number > 0
GROUP BY 1,2,3
ORDER BY n_records DESC LIMIT 20
""", "3b. Creatinine UNIT distribution (CRITICAL for conversion)")

q(f"""
SELECT
  CASE
    WHEN m.value_as_number < 30 THEN 'likely_mg_dL'
    WHEN m.value_as_number >= 30 AND m.value_as_number < 2000 THEN 'likely_umol_L'
    ELSE 'implausible'
  END AS unit_guess,
  COUNT(*) AS n_records,
  COUNT(DISTINCT m.person_id) AS n_patients,
  MIN(m.value_as_number) AS min_val,
  APPROX_QUANTILES(m.value_as_number, 4)[OFFSET(2)] AS median_val,
  MAX(m.value_as_number) AS max_val
FROM `{CDR}`.measurement m
WHERE m.measurement_concept_id IN (
    SELECT concept_id FROM `{CDR}`.concept
    WHERE concept_code = '2160-0' AND vocabulary_id = 'LOINC'
) AND m.value_as_number IS NOT NULL AND m.value_as_number > 0
GROUP BY 1 ORDER BY n_records DESC
""", "3c. Cr value range (unit mixing detection)")

q(f"""
WITH ici_pts AS (
  SELECT DISTINCT de.person_id
  FROM `{CDR}`.drug_exposure de
  JOIN `{CDR}`.concept c ON c.concept_id = de.drug_concept_id
  WHERE {ICI_LIKE}
)
SELECT
  u.concept_name AS unit_name,
  m.unit_source_value,
  COUNT(*) AS n_records,
  COUNT(DISTINCT m.person_id) AS n_ici_patients,
  APPROX_QUANTILES(m.value_as_number, 4)[OFFSET(2)] AS median_val
FROM `{CDR}`.measurement m
JOIN ici_pts i ON m.person_id = i.person_id
LEFT JOIN `{CDR}`.concept u ON u.concept_id = m.unit_concept_id
WHERE m.measurement_concept_id IN (
    SELECT concept_id FROM `{CDR}`.concept
    WHERE concept_code = '2160-0' AND vocabulary_id = 'LOINC'
) AND m.value_as_number IS NOT NULL AND m.value_as_number > 0
GROUP BY 1,2 ORDER BY n_records DESC
""", "3d. Cr units among ICI patients specifically")


# ===============================================================
# 4. FEASIBILITY COUNTS
# ===============================================================
print("\n" + "#"*70)
print("# SECTION 4: FEASIBILITY COUNTS")
print("#"*70)

q(f"SELECT COUNT(DISTINCT person_id) AS n FROM `{CDR}`.person",
  "4a. Total AoU")
q(f"SELECT COUNT(DISTINCT person_id) AS n FROM `{CDR}`.condition_occurrence",
  "4b. With dx")
q(f"""
SELECT COUNT(DISTINCT co.person_id) AS n
FROM `{CDR}`.condition_occurrence co
JOIN `{CDR}`.concept c ON c.concept_id = co.condition_source_concept_id
WHERE c.vocabulary_id = 'ICD10CM' AND c.concept_code LIKE 'C%'
""", "4c. Cancer patients")
q(f"""
SELECT COUNT(DISTINCT de.person_id) AS n
FROM `{CDR}`.drug_exposure de
JOIN `{CDR}`.concept c ON c.concept_id = de.drug_concept_id
WHERE {ICI_LIKE}
""", "4d. ICI patients")
q(f"""
WITH ici AS (
  SELECT DISTINCT de.person_id FROM `{CDR}`.drug_exposure de
  JOIN `{CDR}`.concept c ON c.concept_id = de.drug_concept_id WHERE {ICI_LIKE}
),
cancer AS (
  SELECT DISTINCT co.person_id FROM `{CDR}`.condition_occurrence co
  JOIN `{CDR}`.concept c ON c.concept_id = co.condition_source_concept_id
  WHERE c.vocabulary_id = 'ICD10CM' AND c.concept_code LIKE 'C%'
)
SELECT COUNT(DISTINCT i.person_id) AS n
FROM ici i JOIN cancer cp ON i.person_id = cp.person_id
""", "4e. ICI + cancer")
q(f"""
WITH ici AS (
  SELECT DISTINCT de.person_id FROM `{CDR}`.drug_exposure de
  JOIN `{CDR}`.concept c ON c.concept_id = de.drug_concept_id WHERE {ICI_LIKE}
),
cancer AS (
  SELECT DISTINCT co.person_id FROM `{CDR}`.condition_occurrence co
  JOIN `{CDR}`.concept c ON c.concept_id = co.condition_source_concept_id
  WHERE c.vocabulary_id = 'ICD10CM' AND c.concept_code LIKE 'C%'
),
survey AS (
  SELECT DISTINCT person_id FROM `{CDR}`.observation
  WHERE observation_source_concept_id = 1585845
)
SELECT COUNT(DISTINCT i.person_id) AS n
FROM ici i JOIN cancer cp ON i.person_id = cp.person_id
JOIN survey s ON i.person_id = s.person_id
""", "4f. ICI + cancer + survey")
q(f"""
WITH ici AS (
  SELECT DISTINCT de.person_id FROM `{CDR}`.drug_exposure de
  JOIN `{CDR}`.concept c ON c.concept_id = de.drug_concept_id WHERE {ICI_LIKE}
),
cr AS (
  SELECT DISTINCT person_id FROM `{CDR}`.measurement
  WHERE measurement_concept_id IN (
    SELECT concept_id FROM `{CDR}`.concept
    WHERE concept_code = '2160-0' AND vocabulary_id = 'LOINC'
  ) AND value_as_number IS NOT NULL AND value_as_number > 0
)
SELECT COUNT(DISTINCT i.person_id) AS n
FROM ici i JOIN cr ON i.person_id = cr.person_id
""", "4g. ICI + creatinine")

q(f"""
SELECT c.concept_name, COUNT(DISTINCT de.person_id) AS n_patients
FROM `{CDR}`.drug_exposure de
JOIN `{CDR}`.concept c ON c.concept_id = de.drug_concept_id
WHERE {ICI_LIKE}
GROUP BY 1 ORDER BY n_patients DESC
""", "4h. ICI drug breakdown")

q(f"""
WITH ici AS (
  SELECT DISTINCT de.person_id FROM `{CDR}`.drug_exposure de
  JOIN `{CDR}`.concept c ON c.concept_id = de.drug_concept_id WHERE {ICI_LIKE}
)
SELECT
  CASE
    WHEN c.concept_code LIKE 'C34%%' THEN 'Lung'
    WHEN c.concept_code LIKE 'C43%%' THEN 'Melanoma'
    WHEN c.concept_code LIKE 'C64%%' OR c.concept_code LIKE 'C65%%' THEN 'Renal_Cell'
    WHEN c.concept_code LIKE 'C67%%' THEN 'Urothelial'
    WHEN c.concept_code LIKE 'C50%%' THEN 'Breast'
    WHEN c.concept_code LIKE 'C22%%' THEN 'Hepatocellular'
    WHEN c.concept_code LIKE 'C18%%' OR c.concept_code LIKE 'C19%%'
         OR c.concept_code LIKE 'C20%%' THEN 'Colorectal'
    WHEN c.concept_code >= 'C81' AND c.concept_code < 'C97' THEN 'Hematologic'
    ELSE 'Other'
  END AS cancer_group,
  COUNT(DISTINCT co.person_id) AS n_patients
FROM `{CDR}`.condition_occurrence co
JOIN `{CDR}`.concept c ON c.concept_id = co.condition_source_concept_id
JOIN ici ON co.person_id = ici.person_id
WHERE c.vocabulary_id = 'ICD10CM' AND c.concept_code LIKE 'C%%'
GROUP BY 1 ORDER BY n_patients DESC
""", "4i. Cancer types among ICI patients")


print("\n" + "#"*70)
print("# DONE — Copy EVERYTHING above and paste back to Claude.")
print("# Claude will: (1) set exact concept IDs in ETL,")
print("#              (2) add unit conversion if needed,")
print("#              (3) confirm sample size is sufficient.")
print("#"*70)
