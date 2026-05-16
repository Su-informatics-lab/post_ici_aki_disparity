# Post-ICI AKI × Social Determinants of Health

Propensity-matched case-control analysis of survey-derived SDoH and
all-cause acute kidney injury (AKI) in ICI-treated cancer patients
from the NIH *All of Us* Research Program.

**Target:** JAMIA (Research and Applications)
**Template:** Gatz et al., *JAMIA* 2024;31(12):2932–2939; Wang et al. (COVID-19)

## Design

- **Population:** Cancer patients treated with immune checkpoint inhibitors (ICIs) in AoU
- **Cases:** All-cause AKI (serum creatinine ≥2.0× baseline) within 365 days of first ICI
- **Controls:** ICI patients without AKI in the same window
- **Matching:** 1:4 nearest-neighbor PSM with replacement, 0.2 SD caliper
- **Analysis:** Conditional logistic regression (clogit)
- **CCI:** NCI 14-condition Charlson (cancer-specific; excludes Malignancy/Metastatic)
- **SDoH:** 6 domains (insurance, income, education, employment, housing, disability)

## Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  DATA (01*)  — runs on AoU Researcher Workbench                 │
├─────────────────────────────────────────────────────────────────┤
│  01_ici_aki_etl.py    AoU ETL (BigQuery → CSV)                  │
│    Steps 1–7: ICI cohort, AKI phenotyping, demographics,        │
│    NCI-CCI, SDoH, cancer type, nephrotoxins, matching vars      │
│  01b_psm.R            PSM via MatchIt + cobalt balance          │
├─────────────────────────────────────────────────────────────────┤
│  MODELS (02–03)  — runs on Workbench, reads CSV                 │
├─────────────────────────────────────────────────────────────────┤
│  02_models.R          Base + 6 SDoH domain + joint + race       │
│                       attenuation                               │
│  03_sensitivity.R     S1–S5 sensitivity specifications          │
├─────────────────────────────────────────────────────────────────┤
│  OUTPUT (04–06)  — TBD                                          │
├─────────────────────────────────────────────────────────────────┤
│  04_tables.py         Table 1, Table 2                          │
│  05_figures.py        Forest plots, CONSORT, attenuation        │
│  06_supplement.py     eTables                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Reproduction

```bash
# ── AoU (on Researcher Workbench) ──────────────────────────
python 01_ici_aki_etl.py              # Steps 1–7: cohort → pre-matching base
Rscript 01b_psm.R ici_aki             # PSM → matched cohort + balance
Rscript 02_models.R ici_aki           # Base + 6 SDoH domain + joint + race attenuation
Rscript 03_sensitivity.R ici_aki      # S1–S5 sensitivity analyses
```

## Key Differences from COVID Pipeline (aou_covid)

| Component | COVID (aou_covid) | This study |
|-----------|------------------|------------|
| Population | COVID+ AoU participants | ICI-treated cancer patients |
| Outcome | COVID-19 hospitalization (14d) | All-cause AKI (Cr ≥2.0× baseline, 365d) |
| CCI | Glasheen 19-item | **NCI 14-condition** |
| Additional covariates | Vaccination, pandemic wave | Cancer type, ICI class, nephrotoxins |
| MarketScan arm | Yes | No (v1) |

## Sensitivity Analyses

| ID | Analysis | Rationale |
|----|----------|-----------|
| S1 | Cr ≥1.5× (KDIGO Stage 1+) | More sensitive threshold |
| S2 | Cr ≥3.0× (KDIGO Stage 3) | Severe AKI only |
| S3 | 180-day window | Earlier-onset AKI |
| S4 | Exclude baseline CKD | Remove pre-existing kidney disease |
| S5 | Anti-PD-1/PD-L1 mono only | Most common regimen |

## Requirements

```
Python  3.10+
  pandas, numpy

R       4.5.x
  survival        3.8+
  MatchIt         4.7+
  cobalt          4.6+
  dplyr           1.1+
  readr           2.1+
  sandwich        3.1+
  lmtest          0.9+
```

## License

[MIT](LICENSE)

## Contact

- [Jing Su](mailto:su1@iu.edu) — PI
- [Haining Wang](mailto:hw56@iu.edu) — pipeline lead

Su Lab in Biomedical Informatics, Biostatistics & Health Data Science
Indiana University School of Medicine
