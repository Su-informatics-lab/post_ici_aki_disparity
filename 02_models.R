#!/usr/bin/env Rscript
# ══════════════════════════════════════════════════════════════════════
# Post-ICI AKI × SDoH — Model Fitting (v2 — NCI-CCI scoring fix)
#
# Adapted from aou_covid/02_models.R (Wang et al.)
# Runs on: AoU Researcher Workbench or Quartz HPC
#
# CCI FIX (v2, 2026-05-16):
#   The Python ETL now produces BOTH scores:
#     - charlson_score: Charlson integer weights (corrected MI/hierarchy)
#     - nci_index:      NCI Cox-model continuous weights (Stedman, 5 decimal)
#     - nci_cci_score:  = charlson_score (backward compat alias)
#   DEFAULT: uses nci_cci_score (Charlson integer). Set USE_NCI_INDEX=TRUE
#   below to switch to the NCI continuous index for cancer-specific
#   comorbidity adjustment.
#
# Usage: Rscript 02_models.R ici_aki
#        Rscript 02_models.R inpc
# ══════════════════════════════════════════════════════════════════════

# ── Configuration ────────────────────────────────────────────────────
# Set TRUE to use NCI continuous index instead of Charlson integer score
USE_NCI_INDEX <- FALSE
# ─────────────────────────────────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)
COHORT <- if (length(args) > 0) args[1] else "ici_aki"

# ── R library path (Quartz HPC compatibility) ────────────────────────
user_lib <- file.path(Sys.getenv("HOME"), "R", "library")
dir.create(user_lib, showWarnings = FALSE, recursive = TRUE)
.libPaths(c(user_lib, .libPaths()))

# ── Package management ───────────────────────────────────────────────
required_pkgs <- c("survival", "dplyr", "readr", "sandwich", "lmtest")
for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, lib = user_lib, repos = "https://cloud.r-project.org")
  }
}
library(survival)
library(dplyr)
library(readr)

cat("\n", rep("=", 70), "\n", sep = "")
cat("POST-ICI AKI × SDoH — MODEL FITTING (v2 — NCI-CCI fix)\n")
cat("Cohort:", COHORT, "\n")
cat("CCI mode:", ifelse(USE_NCI_INDEX, "NCI continuous index", "Charlson integer score"), "\n")
cat(rep("=", 70), "\n", sep = "")


# ══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════

# Auto-detect pipeline path
if (dir.exists(file.path("results", COHORT))) {
  RESULTS <- file.path("results", COHORT)
} else if (dir.exists(COHORT)) {
  RESULTS <- COHORT
} else {
  stop("Cannot find results directory for cohort: ", COHORT)
}
cat("Results dir:", RESULTS, "\n")

# Find matched file (try all naming conventions from PSM output)
matched_files <- c(
  file.path(RESULTS, "09_regression_base.csv"),
  file.path(RESULTS, "08_regression_base_matched.csv"),
  file.path(RESULTS, "09_regression_base_matched.csv"),
  file.path(RESULTS, "08_regression_base.csv")
  # NOTE: do NOT fall back to 07_pre_matching_base.csv — it has no stratum column
)
matched_file <- NULL
for (f in matched_files) {
  if (file.exists(f)) { matched_file <- f; break }
}
if (is.null(matched_file)) stop("No matched regression base found.")
cat("Loading:", matched_file, "\n")

regression_bm <- read_csv(matched_file, show_col_types = FALSE)
cat("  Rows:", nrow(regression_bm), " Cols:", ncol(regression_bm), "\n")

# Load SDoH if available
sdoh_file <- file.path(RESULTS, "04_sdoh.csv")
has_sdoh <- file.exists(sdoh_file)
if (has_sdoh) {
  sdoh <- read_csv(sdoh_file, show_col_types = FALSE)
  if (!"insurance_type" %in% names(regression_bm)) {
    regression_bm <- merge(regression_bm, sdoh, by = "person_id", all.x = TRUE)
    cat("  Merged SDoH:", ncol(sdoh) - 1, "variables\n")
  }
}

cat("  Cases:", sum(regression_bm$severity == 1, na.rm = TRUE),
    " Controls:", sum(regression_bm$severity == 0, na.rm = TRUE), "\n")


# ══════════════════════════════════════════════════════════════════════
# FACTOR SETUP
# ══════════════════════════════════════════════════════════════════════

# Outcome
if ("severity" %in% names(regression_bm)) {
  regression_bm$Treatment <- regression_bm$severity
}

# Demographics
regression_bm$f.sex <- factor(regression_bm$sex_at_birth, levels = c("Male", "Female", "Other"))
regression_bm$f.age <- factor(regression_bm$age_group,
  levels = c("55-64", "18-44", "45-54", "65-74", "75+"))

has_race <- "race" %in% names(regression_bm) &&
  length(unique(regression_bm$race[!is.na(regression_bm$race)])) > 1
if (has_race) {
  regression_bm$f.race <- factor(regression_bm$race,
    levels = c("White", "Black", "Asian", "Hispanic", "AIAN",
               "Native_Hawaiian_PI", "Other"))
}

has_ethnicity <- "ethnicity" %in% names(regression_bm) &&
  length(unique(regression_bm$ethnicity[!is.na(regression_bm$ethnicity)])) > 1
if (has_ethnicity) {
  regression_bm$f.ethnicity <- factor(regression_bm$ethnicity,
    levels = c("Not_Hispanic", "Hispanic", "Unknown"))
}

# Cancer type (collapsed to 3)
has_cancer <- any(c("cancer_type_collapsed", "cancer_type") %in% names(regression_bm))
if (has_cancer) {
  if ("cancer_type_collapsed" %in% names(regression_bm)) {
    regression_bm$f.cancer <- factor(regression_bm$cancer_type_collapsed,
      levels = c("Lung", "Melanoma", "Other"))
  } else {
    regression_bm$f.cancer <- factor(regression_bm$cancer_type)
  }
}

# ICI regimen (collapsed to 2)
has_ici <- any(c("ici_collapsed", "ici_regimen") %in% names(regression_bm))
if (has_ici) {
  if ("ici_collapsed" %in% names(regression_bm)) {
    regression_bm$f.ici <- factor(regression_bm$ici_collapsed,
      levels = c("anti_pd1", "other_combo"))
  } else {
    regression_bm$f.ici <- factor(regression_bm$ici_regimen)
  }
}


# ══════════════════════════════════════════════════════════════════════
# NCI-CCI COVARIATE SETUP (v2 fix)
# ══════════════════════════════════════════════════════════════════════
# v2 ETL produces both charlson_score and nci_index.
# Default: use nci_cci_score (= charlson_score, integer).
# Set USE_NCI_INDEX=TRUE to use NCI continuous index instead.

if (USE_NCI_INDEX && "nci_index" %in% names(regression_bm)) {
  como_terms <- "nci_index"
  cat("  CCI covariate: nci_index (NCI continuous, Stedman weights)\n")
} else if ("nci_cci_score" %in% names(regression_bm)) {
  como_terms <- "nci_cci_score"
  cat("  CCI covariate: nci_cci_score (Charlson integer, corrected v2)\n")
} else {
  # Fallback: individual flags
  como <- c("Acute_MI", "History_MI", "Congestive_Heart_Failure",
            "Peripheral_Vascular_Disease", "Cerebrovascular_Disease",
            "Chronic_Pulmonary_Disease", "Dementia", "Paralysis",
            "Diabetes", "Diabetes_Complicated", "Renal_Disease",
            "Liver_Disease_Mild", "Liver_Disease_Moderate_Severe",
            "Peptic_Ulcer_Disease", "Rheumatic_Disease", "AIDS")
  como_terms <- como[como %in% names(regression_bm)]
  cat("  CCI covariate: individual flags (", length(como_terms), "conditions)\n")
}

# Nephrotoxin flags
all_nephro <- c("ppi", "nsaid", "acei_arb", "diuretic")
nephro <- all_nephro[all_nephro %in% names(regression_bm)]
cat("  Nephrotoxin flags:", length(nephro), "\n")


# ══════════════════════════════════════════════════════════════════════
# BUILD BASE FORMULA
# ══════════════════════════════════════════════════════════════════════

base_terms <- c("f.sex", "f.age")
if (has_race) base_terms <- c(base_terms, "f.race")
if (has_ethnicity) base_terms <- c(base_terms, "f.ethnicity")
if (has_cancer) base_terms <- c(base_terms, "f.cancer")
if (has_ici) base_terms <- c(base_terms, "f.ici")
base_terms <- c(base_terms, como_terms, nephro)

base_rhs <- paste(c(base_terms, "strata(stratum)"), collapse = " + ")
base_formula <- as.formula(paste(
  "Surv(rep(1, nrow(regression_bm)), Treatment) ~", base_rhs
))

cat("\n  Base formula RHS:", length(base_terms), "terms\n")
cat("  EPV:", round(sum(regression_bm$Treatment == 1) / length(base_terms), 1), "\n")


# ══════════════════════════════════════════════════════════════════════
# HELPER: fit clogit, extract coefficients, save
# ══════════════════════════════════════════════════════════════════════

fit_and_save <- function(formula_obj, label, data = regression_bm) {
  cat("\n", rep("=", 60), "\n", sep = "")
  cat(toupper(label), "\n")
  cat(rep("=", 60), "\n", sep = "")

  fit <- tryCatch(
    coxph(formula_obj, data = data, method = "exact"),
    error = function(e) {
      cat("  ERROR:", e$message, "\n")
      return(NULL)
    }
  )
  if (is.null(fit)) return(NULL)

  print(fit)

  s <- summary(fit)
  coef_df <- data.frame(
    variable   = rownames(s$coefficients),
    coef       = s$coefficients[, "coef"],
    exp_coef   = s$coefficients[, "exp(coef)"],
    se         = s$coefficients[, "se(coef)"],
    z          = s$coefficients[, "z"],
    p          = s$coefficients[, "Pr(>|z|)"],
    lower95    = s$conf.int[, "lower .95"],
    upper95    = s$conf.int[, "upper .95"],
    model      = label,
    stringsAsFactors = FALSE,
    row.names  = NULL
  )

  fname <- paste0(gsub(" ", "_", tolower(label)), "_coefficients.csv")
  write_csv(coef_df, file.path(RESULTS, fname))
  cat("  Saved:", fname, "\n")

  rdata_fname <- paste0(gsub(" ", "_", tolower(label)), "_clogit.RData")
  save(fit, file = file.path(RESULTS, rdata_fname))

  return(coef_df)
}


# ══════════════════════════════════════════════════════════════════════
# MODEL A: BASE MODEL
# ══════════════════════════════════════════════════════════════════════
base_coefs <- fit_and_save(base_formula, "base")

if (!is.null(base_coefs)) {
  sig <- base_coefs %>% filter(p < 0.05) %>% arrange(desc(abs(coef)))
  cat("\n  Significant base model (p<0.05):\n")
  for (i in seq_len(nrow(sig))) {
    cat(sprintf("    %-45s AOR %.2f (%.2f-%.2f)  p=%.2e\n",
                sig$variable[i], sig$exp_coef[i],
                sig$lower95[i], sig$upper95[i], sig$p[i]))
  }
}


# ══════════════════════════════════════════════════════════════════════
# MODEL B: DOMAIN-BY-DOMAIN SDoH MODELS
# ══════════════════════════════════════════════════════════════════════
all_coefs <- base_coefs

if (has_sdoh) {

  # ── Insurance ──────────────────────────────────────────────────
  regression_bm$f.insurance <- factor(regression_bm$insurance_type,
    levels = c("Private", "Medicare", "Medicaid", "VA_Military",
               "Uninsured", "Other", "Unknown"))
  ins_formula <- update(base_formula, . ~ . + f.insurance)
  ins_coefs <- fit_and_save(ins_formula, "insurance")
  all_coefs <- bind_rows(all_coefs, ins_coefs)

  # ── Income ─────────────────────────────────────────────────────
  regression_bm$f.income <- factor(regression_bm$income,
    levels = c("gt100k", "75k_100k", "50k_75k", "25k_50k",
               "10k_25k", "lt10k", "Unknown"))
  inc_formula <- update(base_formula, . ~ . + f.income)
  inc_coefs <- fit_and_save(inc_formula, "income")
  all_coefs <- bind_rows(all_coefs, inc_coefs)

  # ── Education ──────────────────────────────────────────────────
  regression_bm$f.education <- factor(regression_bm$education,
    levels = c("Graduate", "College", "Some_College",
               "HS_GED", "lt_HS", "Unknown"))
  edu_formula <- update(base_formula, . ~ . + f.education)
  edu_coefs <- fit_and_save(edu_formula, "education")
  all_coefs <- bind_rows(all_coefs, edu_coefs)

  # ── Employment ─────────────────────────────────────────────────
  regression_bm$f.employment <- factor(regression_bm$employment,
    levels = c("Employed", "Self_Employed", "Retired",
               "Unable_to_Work", "Unemployed", "Student",
               "Homemaker", "Other", "Unknown"))
  emp_formula <- update(base_formula, . ~ . + f.employment)
  emp_coefs <- fit_and_save(emp_formula, "employment")
  all_coefs <- bind_rows(all_coefs, emp_coefs)

  # ── Housing ────────────────────────────────────────────────────
  regression_bm$f.housing <- factor(regression_bm$housing,
    levels = c("Own", "Rent", "Other_Arrangement", "Unknown"))
  hou_formula <- update(base_formula, . ~ . + f.housing)
  hou_coefs <- fit_and_save(hou_formula, "housing")
  all_coefs <- bind_rows(all_coefs, hou_coefs)

  # ── Housing stability ──────────────────────────────────────────
  regression_bm$f.stability <- factor(regression_bm$housing_stability,
    levels = c("Stable", "Unstable", "Unknown"))
  stab_formula <- update(base_formula, . ~ . + f.stability)
  stab_coefs <- fit_and_save(stab_formula, "housing_stability")
  all_coefs <- bind_rows(all_coefs, stab_coefs)

  # ── JOINT MODEL (all SDoH) ────────────────────────────────────
  joint_terms <- c()
  if ("f.insurance" %in% names(regression_bm)) joint_terms <- c(joint_terms, "f.insurance")
  if ("f.income" %in% names(regression_bm)) joint_terms <- c(joint_terms, "f.income")
  if ("f.education" %in% names(regression_bm)) joint_terms <- c(joint_terms, "f.education")
  if ("f.employment" %in% names(regression_bm)) joint_terms <- c(joint_terms, "f.employment")
  if ("f.housing" %in% names(regression_bm)) joint_terms <- c(joint_terms, "f.housing")
  if ("f.stability" %in% names(regression_bm)) joint_terms <- c(joint_terms, "f.stability")

  if (length(joint_terms) > 0) {
    joint_rhs <- paste(c(base_terms, joint_terms, "strata(stratum)"), collapse = " + ")
    joint_formula <- as.formula(paste(
      "Surv(rep(1, nrow(regression_bm)), Treatment) ~", joint_rhs
    ))
    joint_coefs <- fit_and_save(joint_formula, "joint_sdoh")
    all_coefs <- bind_rows(all_coefs, joint_coefs)
  }

  # ── RACE ATTENUATION ───────────────────────────────────────────
  if (has_race && !is.null(base_coefs) && !is.null(joint_coefs)) {
    base_black <- base_coefs %>% filter(variable == "f.raceBlack")
    joint_black <- joint_coefs %>% filter(variable == "f.raceBlack")

    if (nrow(base_black) > 0 && nrow(joint_black) > 0) {
      base_aor <- base_black$exp_coef[1]
      joint_aor <- joint_black$exp_coef[1]
      attenuation <- ((log(base_aor) - log(joint_aor)) / log(base_aor)) * 100

      cat("\n", rep("=", 60), "\n", sep = "")
      cat("RACE ATTENUATION ANALYSIS\n")
      cat(rep("=", 60), "\n", sep = "")
      cat(sprintf("  Base model:  Black AOR = %.2f (%.2f-%.2f) p=%.3f\n",
                  base_aor, base_black$lower95[1], base_black$upper95[1], base_black$p[1]))
      cat(sprintf("  Joint model: Black AOR = %.2f (%.2f-%.2f) p=%.3f\n",
                  joint_aor, joint_black$lower95[1], joint_black$upper95[1], joint_black$p[1]))
      cat(sprintf("  Attenuation: %.1f%%\n", attenuation))

      atten_df <- data.frame(
        model = c("base", "joint_sdoh"),
        black_aor = c(base_aor, joint_aor),
        black_lower = c(base_black$lower95[1], joint_black$lower95[1]),
        black_upper = c(base_black$upper95[1], joint_black$upper95[1]),
        black_p = c(base_black$p[1], joint_black$p[1]),
        attenuation_pct = c(0, attenuation)
      )
      write_csv(atten_df, file.path(RESULTS, "race_attenuation.csv"))
      cat("  Saved: race_attenuation.csv\n")
    }
  }

} else {
  cat("\n  No SDoH data — skipping SDoH models (INPC transportability arm)\n")
}


# ══════════════════════════════════════════════════════════════════════
# SAVE COMBINED COEFFICIENTS
# ══════════════════════════════════════════════════════════════════════

if (!is.null(all_coefs) && nrow(all_coefs) > 0) {
  write_csv(all_coefs, file.path(RESULTS, "all_model_coefficients.csv"))
  cat("\nSaved: all_model_coefficients.csv (", nrow(all_coefs), "rows)\n")
}

cat("\n", rep("=", 70), "\n", sep = "")
cat("MODEL FITTING COMPLETE (v2 — NCI-CCI scoring fix)\n")
cat("CCI mode:", ifelse(USE_NCI_INDEX, "NCI continuous index", "Charlson integer score"), "\n")
cat(rep("=", 70), "\n", sep = "")
