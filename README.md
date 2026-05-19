# Post-ICI AKI × Social Determinants of Health

Two-cohort propensity-matched case-control study of social determinants
of health and acute kidney injury in immune checkpoint inhibitor–treated
cancer patients.

- **All of Us** (primary) — ICI-treated cancer patients with 6 SDoH domains from the Basics Survey
- **INPC** (transportability) — ICI-treated cancer patients from the Indiana Network for Patient Care

## Reproduction

```bash
# ── AoU (Researcher Workbench, CDR C2024Q3R9) ──────────────
python 01_etl.py aou              # ETL → results/ici_aki/
Rscript 01b_psm.R aou             # 1:4 NN PSM with replacement
Rscript 02_models.R aou           # base + 6 SDoH domain + joint + race attenuation
Rscript 03_sensitivity.R aou      # S1–S5

# ── INPC (Quartz HPC, /N/project/depot/hw56/irAKI_data/) ───
python 01_etl.py inpc             # ETL → results/inpc/
Rscript 01b_psm.R inpc            # 1:4 NN PSM with replacement
Rscript 02_models.R inpc          # base model only (no SDoH in INPC)
Rscript 03_sensitivity.R inpc     # S1–S5

# ── Tables & Figures ────────────────────────────────────────
python 04_consort_table1.py        # Table 1, CONSORT numbers
python 05_figures.py               # Figures 1–3 (CONSORT, SDoH forest, race/sensitivity)
python 06_supplement.py            # eTables 1–7, eFigures 1–2
```

## Pipeline

| Step | Script | Input | Description |
|------|--------|-------|-------------|
| ETL | `01_etl.py aou\|inpc` | BigQuery / CSV | Cohort, Cr phenotyping, NCI-CCI, SDoH, covariates |
| PSM | `01b_psm.R aou\|inpc` | `07_pre_matching_base.csv` | 1:4 NN matching, balance diagnostics |
| Models | `02_models.R aou\|inpc` | `09_regression_base.csv` | Base + SDoH domain + joint + race attenuation |
| Sensitivity | `03_sensitivity.R aou\|inpc` | `09_regression_base.csv` | S1 (Δ0.3), S2 (KDIGO 2), S3 (KDIGO 3), S4 (180d), S5 (mono-ICI) |
| Tables | `04_consort_table1.py` | Both cohorts | Table 1 characteristics, CONSORT flowchart numbers |
| Figures | `05_figures.py` | Both cohorts | Figure 1 (CONSORT), Figure 2 (SDoH forest), Figure 3 (race + sensitivity) |
| Supplement | `06_supplement.py` | Both cohorts | eTables 1–7, eFigures 1–2, STROBE checklist |
| Shared | `nci_cci_scoring.py` | — | NCI 14-condition Charlson (excludes cancer; both cohorts) |

## Key Methods

- **AKI phenotype:** Serum creatinine ≥1.5× baseline within 365 days of first ICI, with 90-day pre-ICI washout, baseline Cr fallback [-365,−7] → [-365,−1], and CKD-EPI 2021 eGFR
- **Cancer type:** 4-level — Lung (ref), Melanoma, Renal Cell, Other
- **ICI class:** 3-level — anti-PD-1 (ref), anti-PD-L1, CTLA-4 containing
- **Matching:** 1:4 nearest-neighbor PSM with replacement, 0.2 SD caliper
- **Analysis:** Exact conditional logistic regression (clogit); baseline eGFR excluded from base model (mediator, not confounder; reported descriptively and as sensitivity analysis)
- **Comorbidity:** NCI 14-condition Charlson integer score (excludes Malignancy/Metastatic)
- **SDoH (AoU only):** Insurance, income, education, employment, housing, housing stability

## Data Access

- **All of Us:** Controlled Tier via [Researcher Workbench](https://workbench.researchallofus.org)
- **INPC:** De-identified extract available from corresponding author under DUA with Indiana University and Regenstrief Institute
- Person-level CSVs are `.gitignore`d; only aggregate model outputs are committed.

## Requirements

```
Python 3.10+    pandas, numpy, matplotlib, seaborn
R 4.5+          survival, MatchIt, cobalt, dplyr, readr
```

## Legacy Scripts

`01_aou_etl.py` and `01_inpc_etl.py` are superseded by the consolidated `01_etl.py`.

## Contact

[Haining Wang](mailto:hw56@iu.edu) · [Jing Su](mailto:su1@iu.edu)
Su Lab — Biostatistics & Health Data Science, Indiana University School of Medicine
