#!/usr/bin/env python3
"""
INPC Recon for Post-ICI AKI — Run on Quartz HPC.
Discovers: ICI drugs in med/pharmacy, lab schema, feasibility counts.
Paste FULL output back to Claude.
"""

import os

import duckdb

RAW = "/N/project/depot/raw/inpc_12_14_2023_parquet"
con = duckdb.connect(":memory:")
con.execute("PRAGMA threads=7")
con.execute("SET memory_limit='60GB'")


def q(sql, label=""):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    df = con.sql(sql).df()
    print(df.to_string(index=False))
    print(f"  [{len(df)} rows]")
    return df


# ═══════════════════════════════════════════════════════════
# 1. LAB SCHEMA DISCOVERY
# ═══════════════════════════════════════════════════════════
print("#" * 60)
print("# SECTION 1: LAB FILE SCHEMA")
print("#" * 60)

lab_dir = f"{RAW}/lab/labparquetfiles"
lab_files = [f for f in os.listdir(lab_dir) if f.endswith(".parquet")]
print(f"\n  Lab files: {len(lab_files)}")
for f in sorted(lab_files)[:5]:
    print(f"    {f}")

# Show columns from first lab file
first_lab = os.path.join(lab_dir, sorted(lab_files)[0])
q(
    f"SELECT * FROM read_parquet('{first_lab}') LIMIT 3",
    "Lab file columns (first 3 rows)",
)

# Show column names and types
q(
    f"""
SELECT column_name, column_type
FROM (DESCRIBE SELECT * FROM read_parquet('{first_lab}'))
""",
    "Lab column schema",
)


# ═══════════════════════════════════════════════════════════
# 2. CREATININE IN LABS
# ═══════════════════════════════════════════════════════════
print("\n" + "#" * 60)
print("# SECTION 2: CREATININE DISCOVERY")
print("#" * 60)

# Build a union of all lab files
lab_glob = f"{lab_dir}/*.parquet"

# What test names contain "creatinine"?
q(
    f"""
SELECT DISTINCT UPPER(TRIM(LAB_TEST_NAME)) AS test_name, COUNT(*) AS n
FROM read_parquet('{lab_glob}')
WHERE LOWER(LAB_TEST_NAME) LIKE '%creatinine%'
  AND LOWER(LAB_TEST_NAME) NOT LIKE '%clearance%'
  AND LOWER(LAB_TEST_NAME) NOT LIKE '%urine%'
  AND LOWER(LAB_TEST_NAME) NOT LIKE '%ratio%'
GROUP BY 1 ORDER BY n DESC LIMIT 20
""",
    "Creatinine test names",
)

# Also check LOINC codes if available
q(
    f"""
SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{first_lab}'))
WHERE LOWER(column_name) LIKE '%loinc%' OR LOWER(column_name) LIKE '%code%'
""",
    "LOINC/code columns in lab",
)

# Value distribution for creatinine
q(
    f"""
SELECT
  COUNT(*) AS n_records,
  COUNT(DISTINCT STUDY_ID) AS n_patients,
  MIN(TRY_CAST(LAB_VALUE AS DOUBLE)) AS min_val,
  APPROX_QUANTILE(TRY_CAST(LAB_VALUE AS DOUBLE), 0.5) AS median_val,
  MAX(TRY_CAST(LAB_VALUE AS DOUBLE)) AS max_val
FROM read_parquet('{lab_glob}')
WHERE LOWER(LAB_TEST_NAME) LIKE '%creatinine%'
  AND LOWER(LAB_TEST_NAME) NOT LIKE '%clearance%'
  AND LOWER(LAB_TEST_NAME) NOT LIKE '%urine%'
  AND TRY_CAST(LAB_VALUE AS DOUBLE) IS NOT NULL
  AND TRY_CAST(LAB_VALUE AS DOUBLE) > 0
""",
    "Creatinine value distribution",
)

# Unit column?
q(
    f"""
SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{first_lab}'))
WHERE LOWER(column_name) LIKE '%unit%'
""",
    "Unit columns in lab",
)


# ═══════════════════════════════════════════════════════════
# 3. ICI DRUGS IN MED/PHARMACY
# ═══════════════════════════════════════════════════════════
print("\n" + "#" * 60)
print("# SECTION 3: ICI DRUGS IN MED ORDERS + PHARMACY")
print("#" * 60)

ICI_NAMES = [
    "nivolumab",
    "pembrolizumab",
    "cemiplimab",
    "dostarlimab",
    "retifanlimab",
    "toripalimab",
    "tislelizumab",
    "atezolizumab",
    "durvalumab",
    "avelumab",
    "ipilimumab",
    "tremelimumab",
    "relatlimab",
    "opdivo",
    "keytruda",
    "yervoy",
    "tecentriq",
    "imfinzi",
    "bavencio",
    "libtayo",
    "opdualag",
]
ICI_WHERE = " OR ".join([f"LOWER(MEDICATION_NAME) LIKE '%{d}%'" for d in ICI_NAMES])

# Med orders
med_file = f"{RAW}/med/inpc_med_orders.parquet"
if os.path.exists(med_file):
    q(
        f"""
    SELECT MEDICATION_NAME, COUNT(DISTINCT STUDY_ID) AS n_patients, COUNT(*) AS n_records
    FROM read_parquet('{med_file}')
    WHERE {ICI_WHERE}
    GROUP BY 1 ORDER BY n_patients DESC LIMIT 30
    """,
        "ICI in med_orders",
    )
else:
    print("  med_orders not found")

# Pharmacy
pharm_dir = f"{RAW}/pharmacy"
pharm_files = [
    os.path.join(pharm_dir, f) for f in os.listdir(pharm_dir) if f.endswith(".parquet")
]
if pharm_files:
    pharm_glob = f"{pharm_dir}/*.parquet"
    q(
        f"""
    SELECT MEDICATION_NAME, COUNT(DISTINCT STUDY_ID) AS n_patients, COUNT(*) AS n_records
    FROM read_parquet('{pharm_glob}')
    WHERE {ICI_WHERE}
    GROUP BY 1 ORDER BY n_patients DESC LIMIT 30
    """,
        "ICI in pharmacy",
    )
else:
    print("  pharmacy files not found")


# ═══════════════════════════════════════════════════════════
# 4. FEASIBILITY COUNTS
# ═══════════════════════════════════════════════════════════
print("\n" + "#" * 60)
print("# SECTION 4: FEASIBILITY COUNTS")
print("#" * 60)

# Total patients
q(
    f"""
SELECT COUNT(DISTINCT STUDY_ID) AS n
FROM read_parquet('{RAW}/demography/rdrp5292_demo_de.parquet')
""",
    "4a. Total INPC patients",
)

# Cancer patients
q(
    f"""
SELECT COUNT(DISTINCT STUDY_ID) AS n
FROM read_parquet('{RAW}/diagnosis/inpc_diagnosis.parquet')
WHERE UPPER(REPLACE(TRIM(DX_CODE), '.', '')) LIKE 'C%'
""",
    "4b. Cancer patients",
)

# ICI patients (from med orders)
if os.path.exists(med_file):
    q(
        f"""
    SELECT COUNT(DISTINCT STUDY_ID) AS n
    FROM read_parquet('{med_file}')
    WHERE {ICI_WHERE}
    """,
        "4c. ICI patients (med_orders)",
    )

# ICI + cancer
if os.path.exists(med_file):
    q(
        f"""
    WITH ici AS (
      SELECT DISTINCT STUDY_ID FROM read_parquet('{med_file}')
      WHERE {ICI_WHERE}
    ),
    cancer AS (
      SELECT DISTINCT STUDY_ID
      FROM read_parquet('{RAW}/diagnosis/inpc_diagnosis.parquet')
      WHERE UPPER(REPLACE(TRIM(DX_CODE), '.', '')) LIKE 'C%'
    )
    SELECT COUNT(DISTINCT i.STUDY_ID) AS n
    FROM ici i JOIN cancer c ON i.STUDY_ID = c.STUDY_ID
    """,
        "4d. ICI + cancer",
    )

# ICI + creatinine
if os.path.exists(med_file):
    q(
        f"""
    WITH ici AS (
      SELECT DISTINCT STUDY_ID FROM read_parquet('{med_file}')
      WHERE {ICI_WHERE}
    ),
    cr AS (
      SELECT DISTINCT STUDY_ID FROM read_parquet('{lab_glob}')
      WHERE LOWER(LAB_TEST_NAME) LIKE '%creatinine%'
        AND LOWER(LAB_TEST_NAME) NOT LIKE '%clearance%'
        AND LOWER(LAB_TEST_NAME) NOT LIKE '%urine%'
        AND TRY_CAST(LAB_VALUE AS DOUBLE) > 0
    )
    SELECT COUNT(DISTINCT i.STUDY_ID) AS n
    FROM ici i JOIN cr ON i.STUDY_ID = cr.STUDY_ID
    """,
        "4e. ICI + creatinine",
    )

# Demographics of ICI patients
if os.path.exists(med_file):
    q(
        f"""
    WITH ici AS (
      SELECT DISTINCT STUDY_ID FROM read_parquet('{med_file}')
      WHERE {ICI_WHERE}
    )
    SELECT d.RACE, COUNT(DISTINCT d.STUDY_ID) AS n
    FROM read_parquet('{RAW}/demography/rdrp5292_demo_de.parquet') d
    JOIN ici ON d.STUDY_ID = ici.STUDY_ID
    GROUP BY 1 ORDER BY n DESC
    """,
        "4f. Race distribution of ICI patients",
    )

con.close()
print("\n" + "#" * 60)
print("# DONE — paste back to Claude")
print("#" * 60)
