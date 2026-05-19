# Post-ICI AKI × Social Determinants of Health

Two-cohort propensity-matched case-control study of social determinants
of health and acute kidney injury in immune checkpoint inhibitor–treated
cancer patients.

- **All of Us** (primary) — ICI-treated cancer patients with 6 SDoH domains from the Basics Survey
- **INPC** (transportability) — ICI-treated cancer patients from the Indiana Network for Patient Care

## Reproduction

```bash
# ── AoU (Researcher Workbench, CDR C2024Q3R9) ──────────────
python 01_aou_etl.py                 # ETL → results/ici_aki/01–07
Rscript 01b_psm.R ici_aki           # 1:4 NN PSM with replacement
Rscript 02_models.R ici_aki         # base + 6 SDoH domain + joint + race attenuation
Rscript 03_sensitivity.R ici_aki    # S1–S5

# ── INPC (Quartz HPC, /N/project/depot/hw56/irAKI_data/) ───
python 01_inpc_etl.py                # ETL → results/inpc/01–07
Rscript 01b_psm.R inpc              # 1:4 NN PSM with replacement
Rscript 02_models.R inpc            # base model (no SDoH in INPC)
Rscript 03_sensitivity.R inpc       # S1–S5

# ── Tables & Figures ───────────────────────────────────────
python 04_tables.py                  # Table 1, Table 2
python 05_figures.py                 # Figures 1–3
python 06_supplement.py              # eTables, eFigures
```

## Pipeline

| Step | Script | Environment | Output |
|------|--------|-------------|--------|
| ETL | `01_aou_etl.py` | AoU Workbench | `results/ici_aki/01–07_*.csv` |
| ETL | `01_inpc_etl.py` | Quartz HPC | `results/inpc/01–07_*.csv` |
| PSM | `01b_psm.R` | R 4.5+ | `08_matched_cohort.csv`, `08c_smd_balance.csv` |
| Models | `02_models.R` | R 4.5+ | `*_coefficients.csv`, `race_attenuation.csv` |
| Sensitivity | `03_sensitivity.R` | R 4.5+ | `all_sensitivity_coefficients.csv` |
| Shared | `nci_cci_scoring.py` | Python 3.10+ | NCI 14-condition CCI (both cohorts) |

## Key Methods

- **AKI phenotype:** Serum creatinine ≥1.5× baseline within 365 days of first ICI, with 90-day pre-ICI washout, baseline Cr fallback [-365,−7] → [-365,−1], and CKD-EPI 2021 eGFR
- **Matching:** 1:4 nearest-neighbor PSM with replacement, 0.2 SD caliper
- **Analysis:** Exact conditional logistic regression (clogit)
- **Comorbidity:** NCI 14-condition Charlson (excludes Malignancy/Metastatic)
- **SDoH (AoU only):** Insurance, income, education, employment, housing, housing stability

## Data Access

- **All of Us:** Controlled Tier via [Researcher Workbench](https://workbench.researchallofus.org)
- **INPC:** De-identified extract available from corresponding author under DUA with Indiana University and Regenstrief Institute
- Person-level CSVs are `.gitignore`d. Only aggregate model outputs (`*_coefficients.csv`, `*_balance.csv`) are committed.

## Requirements

```
Python 3.10+    pandas, numpy
R 4.5+          survival, MatchIt, cobalt, dplyr, readr, sandwich, lmtest
```

## Contact

[Haining Wang](mailto:hw56@iu.edu) · [Jing Su](mailto:su1@iu.edu)
Su Lab — Biostatistics & Health Data Science, Indiana University School of Medicine
