#!/usr/bin/env python3
"""Part 2: audit measurement + condition (columns differ from OMOP standard)."""

import warnings

import pandas as pd

warnings.filterwarnings("ignore")
DATA = "/N/project/depot/hw56/irAKI_data/structured_data"

# Just read first few rows to get columns
meas_sample = pd.read_csv(f"{DATA}/r6335_measurement.csv", nrows=5)
print("MEASUREMENT columns:", list(meas_sample.columns))
print(meas_sample.head(2).to_string(index=False))

cond_sample = pd.read_csv(f"{DATA}/r6335_condition_occurrence.csv", nrows=5)
print("\nCONDITION columns:", list(cond_sample.columns))
print(cond_sample.head(2).to_string(index=False))

# Now read measurement with chunking (140M rows is too big for one shot)
print("\nReading measurement in chunks (Cr only)...")
cr_rows = []
for chunk in pd.read_csv(
    f"{DATA}/r6335_measurement.csv", chunksize=5_000_000, low_memory=False
):
    # Standardize column names to lowercase
    chunk.columns = [c.lower() for c in chunk.columns]
    pid_col = [c for c in chunk.columns if "person" in c][0]
    cid_col = [c for c in chunk.columns if "concept_id" in c and "measurement" in c][0]
    val_col = [c for c in chunk.columns if "value_as_number" in c]

    # Filter for creatinine concept 3016723
    cr = chunk[chunk[cid_col] == 3016723]
    if len(cr) > 0:
        cr_rows.append(cr)
    print(f"  chunk: {len(chunk):,} rows, cr: {len(cr):,}")

if cr_rows:
    cr_all = pd.concat(cr_rows, ignore_index=True)
    pid_col = [c for c in cr_all.columns if "person" in c][0]
    val_col = [c for c in cr_all.columns if "value_as_number" in c]
    print(
        f"\nCr total: {len(cr_all):,} records, {cr_all[pid_col].nunique():,} patients"
    )
    if val_col:
        v = pd.to_numeric(cr_all[val_col[0]], errors="coerce").dropna()
        v = v[v > 0]
        print(
            f"Cr values: median={v.median():.2f}, IQR={v.quantile(0.25):.2f}-{v.quantile(0.75):.2f}"
        )
    unit_col = [c for c in cr_all.columns if "unit" in c and "source" in c]
    if unit_col:
        print(f"Units: {cr_all[unit_col[0]].value_counts().head(5).to_dict()}")

    # Cross with ICI patients (3,259 from part 1)
    drug = pd.read_csv(
        f"{DATA}/r6335_drug_exposure.csv",
        low_memory=False,
        usecols=["person_id", "drug_concept_id"],
    )
    concept = pd.read_csv(
        f"{DATA}/r6335_concept.csv",
        encoding="cp1252",
        low_memory=False,
        usecols=["concept_id", "concept_name"],
    )
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
    dm = drug.merge(
        concept.rename(columns={"concept_id": "drug_concept_id"}),
        on="drug_concept_id",
        how="left",
    )
    ici_pids = set(
        dm[
            dm.concept_name.str.lower().str.contains("|".join(ici_terms), na=False)
        ].person_id
    )
    cr_pids = set(cr_all[pid_col])
    print(f"\nICI patients: {len(ici_pids):,}")
    print(f"Cr patients: {len(cr_pids):,}")
    print(f"ICI + Cr: {len(ici_pids & cr_pids):,}")
else:
    print("No creatinine found with concept 3016723")
    print("Try searching by name in concept table instead")

print("\n# DONE — paste back to Claude")
