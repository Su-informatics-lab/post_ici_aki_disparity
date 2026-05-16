#!/usr/bin/env python3
"""INPC Recon for Post-ICI AKI — Run on Quartz. Paste output back to Claude."""

import os

import duckdb

RAW = "/N/project/depot/raw/inpc_12_14_2023_parquet"
con = duckdb.connect(":memory:")
con.execute("PRAGMA threads=7; SET memory_limit='60GB'")
lab_glob = f"{RAW}/lab/labparquetfiles/*.parquet"
med_file = f"{RAW}/med/inpc_med_orders.parquet"
dx_file = f"{RAW}/diagnosis/inpc_diagnosis.parquet"
demo_file = f"{RAW}/demography/rdrp5292_demo_de.parquet"


def q(sql, label=""):
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    df = con.sql(sql).df()
    print(df.to_string(index=False))
    print(f"  [{len(df)} rows]")
    return df


ICI = [
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
ICI_W = " OR ".join([f"LOWER(MEDICATION_NAME) LIKE '%{d}%'" for d in ICI])

print("#" * 60 + "\n# SECTION 1: CREATININE\n" + "#" * 60)

q(
    f"""
SELECT LOINC, LAB_NAME, COUNT(*) AS n, COUNT(DISTINCT STUDY_ID) AS n_pts,
  APPROX_QUANTILE(TRY_CAST(LAB_RESULT AS DOUBLE), 0.5) AS median_val
FROM read_parquet('{lab_glob}')
WHERE LOINC = '2160-0' AND TRY_CAST(LAB_RESULT AS DOUBLE) > 0
GROUP BY 1,2 ORDER BY n DESC LIMIT 10
""",
    "Cr via LOINC 2160-0",
)

q(
    f"""
SELECT LAB_UNITS, COUNT(*) AS n,
  APPROX_QUANTILE(TRY_CAST(LAB_RESULT AS DOUBLE), 0.5) AS median_val
FROM read_parquet('{lab_glob}')
WHERE LOINC = '2160-0' AND TRY_CAST(LAB_RESULT AS DOUBLE) > 0
GROUP BY 1 ORDER BY n DESC LIMIT 10
""",
    "Cr unit distribution",
)

print("\n" + "#" * 60 + "\n# SECTION 2: ICI DRUGS\n" + "#" * 60)

q(
    f"""
SELECT MEDICATION_NAME, COUNT(DISTINCT STUDY_ID) AS n_pts, COUNT(*) AS n_rec
FROM read_parquet('{med_file}')
WHERE {ICI_W}
GROUP BY 1 ORDER BY n_pts DESC LIMIT 30
""",
    "ICI in med_orders",
)

pharm_dir = f"{RAW}/pharmacy"
if os.path.exists(pharm_dir) and any(
    f.endswith(".parquet") for f in os.listdir(pharm_dir)
):
    q(
        f"""
    SELECT MEDICATION_NAME, COUNT(DISTINCT STUDY_ID) AS n_pts
    FROM read_parquet('{pharm_dir}/*.parquet')
    WHERE {ICI_W}
    GROUP BY 1 ORDER BY n_pts DESC LIMIT 20
    """,
        "ICI in pharmacy",
    )

print("\n" + "#" * 60 + "\n# SECTION 3: FEASIBILITY\n" + "#" * 60)

q(
    f"SELECT COUNT(DISTINCT STUDY_ID) AS n FROM read_parquet('{demo_file}')",
    "Total patients",
)
q(
    f"""
SELECT COUNT(DISTINCT STUDY_ID) AS n FROM read_parquet('{dx_file}')
WHERE UPPER(REPLACE(TRIM(DX_CODE),'.','')) LIKE 'C%'
""",
    "Cancer patients",
)
q(
    f"SELECT COUNT(DISTINCT STUDY_ID) AS n FROM read_parquet('{med_file}') WHERE {ICI_W}",
    "ICI patients",
)
q(
    f"""
WITH ici AS (SELECT DISTINCT STUDY_ID FROM read_parquet('{med_file}') WHERE {ICI_W}),
cancer AS (SELECT DISTINCT STUDY_ID FROM read_parquet('{dx_file}')
  WHERE UPPER(REPLACE(TRIM(DX_CODE),'.','')) LIKE 'C%')
SELECT COUNT(DISTINCT i.STUDY_ID) AS n FROM ici i JOIN cancer c ON i.STUDY_ID=c.STUDY_ID
""",
    "ICI + cancer",
)
q(
    f"""
WITH ici AS (SELECT DISTINCT STUDY_ID FROM read_parquet('{med_file}') WHERE {ICI_W}),
cr AS (SELECT DISTINCT STUDY_ID FROM read_parquet('{lab_glob}')
  WHERE LOINC='2160-0' AND TRY_CAST(LAB_RESULT AS DOUBLE) > 0)
SELECT COUNT(DISTINCT i.STUDY_ID) AS n FROM ici i JOIN cr ON i.STUDY_ID=cr.STUDY_ID
""",
    "ICI + creatinine",
)
q(
    f"""
WITH ici AS (SELECT DISTINCT STUDY_ID FROM read_parquet('{med_file}') WHERE {ICI_W})
SELECT d.RACE, COUNT(DISTINCT d.STUDY_ID) AS n
FROM read_parquet('{demo_file}') d JOIN ici ON d.STUDY_ID=ici.STUDY_ID
GROUP BY 1 ORDER BY n DESC
""",
    "ICI patient race distribution",
)

con.close()
print("\n" + "#" * 60 + "\n# DONE — paste back to Claude\n" + "#" * 60)
