#!/usr/bin/env Rscript
# ══════════════════════════════════════════════════════════════════════
# Post-ICI AKI × SDoH — Sensitivity Analyses
# Adapted from aou_covid/03_sensitivity.R
#
# S1: AKI defined as Cr ≥1.5× baseline (KDIGO Stage 1+)
# S2: AKI defined as Cr ≥3.0× baseline (KDIGO Stage 3)
# S3: 180-day observation window (instead of 365)
# S4: Exclude patients with baseline CKD (Renal_Disease = 1)
# S5: Restrict to anti-PD-1/PD-L1 monotherapy only
#
# Usage: Rscript 03_sensitivity.R ici_aki
# ══════════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
  library(survival)
  library(dplyr)
  library(readr)
})

args <- commandArgs(trailingOnly = TRUE)
COHORT <- ifelse(length(args) >= 1, args[1], "ici_aki")
RESULTS <- file.path("results", COHORT)

cat(rep("=", 70), "\n", sep = "")
cat("POST-ICI AKI × SDoH — SENSITIVITY  [", toupper(COHORT), "]\n")
cat(rep("=", 70), "\n", sep = "")

# ── Load regression base ─────────────────────────────────────────
regression_bm <- read_csv(
  file.path(RESULTS, "09_regression_base.csv"),
  show_col_types = FALSE
)
cat("  Loaded:", nrow(regression_bm), "rows\n")

# ── Factor encoding (copied from 02_models.R) ────────────────────
regression_bm$f.sex  <- factor(regression_bm$sex_at_birth,
                               levels = c("Male", "Female", "Other"))
regression_bm$f.age  <- factor(regression_bm$age_group,
                               levels = c("<45", "45-54", "55-64", "65+"))
has_race <- "race" %in% names(regression_bm) &&
            any(regression_bm$race != "Unknown", na.rm = TRUE)
has_ethnicity <- "ethnicity" %in% names(regression_bm)
has_cancer <- "cancer_type" %in% names(regression_bm)
has_ici <- "ici_regimen" %in% names(regression_bm)

if (has_race) regression_bm$f.race <- factor(regression_bm$race,
                   levels = c("White", "Black", "Asian", "Other"))
if (has_ethnicity) regression_bm$f.ethnicity <- factor(regression_bm$ethnicity,
                   levels = c("Not Hispanic", "Hispanic", "Other"))
if (has_cancer) regression_bm$f.cancer <- factor(regression_bm$cancer_type,
                   levels = c("Lung", "Melanoma", "Renal_Cell", "Urothelial",
                              "Head_Neck", "Breast", "Hepatocellular",
                              "Colorectal", "Other_Solid", "Hematologic", "Unknown"))
if (has_ici) regression_bm$f.ici <- factor(regression_bm$ici_regimen,
                   levels = c("anti_pd1", "anti_pdl1", "anti_ctla4",
                              "anti_lag3", "combination"))

# NCI-CCI conditions
como <- c("Acute_MI", "History_MI", "Congestive_Heart_Failure",
          "Peripheral_Vascular_Disease", "Cerebrovascular_Disease",
          "Chronic_Pulmonary_Disease", "Dementia", "Paralysis",
          "Diabetes", "Diabetes_Complicated", "Renal_Disease",
          "Liver_Disease_Mild", "Liver_Disease_Moderate_Severe",
          "Peptic_Ulcer_Disease", "Rheumatic_Disease", "AIDS")
como <- como[como %in% names(regression_bm)]

nephro <- c("ppi_flag", "nsaid_flag", "acei_arb_flag", "diuretic_flag")
nephro <- nephro[nephro %in% names(regression_bm)]

# Base formula components
base_terms <- c("f.sex", "f.age")
if (has_race) base_terms <- c(base_terms, "f.race")
if (has_ethnicity) base_terms <- c(base_terms, "f.ethnicity")
if (has_cancer) base_terms <- c(base_terms, "f.cancer")
if (has_ici) base_terms <- c(base_terms, "f.ici")
base_terms <- c(base_terms, como, nephro)


# ── Helper ───────────────────────────────────────────────────────
run_sensitivity <- function(data, label, outcome_col = "Treatment") {
  cat("\n", rep("=", 60), "\n", sep = "")
  cat("SENSITIVITY:", label, "\n")
  cat("  N=", nrow(data), " Cases=", sum(data[[outcome_col]] == 1),
      " Controls=", sum(data[[outcome_col]] == 0), "\n")
  cat(rep("=", 60), "\n", sep = "")

  if (sum(data[[outcome_col]] == 1) < 10) {
    cat("  SKIPPED: <10 cases\n")
    return(NULL)
  }

  rhs <- paste(c(base_terms, "strata(stratum)"), collapse = " + ")
  formula_obj <- as.formula(
    paste("Surv(rep(1, nrow(data)),", outcome_col, ") ~", rhs)
  )

  fit <- tryCatch(
    coxph(formula_obj, data = data, method = "exact"),
    error = function(e) { cat("  ERROR:", e$message, "\n"); NULL }
  )
  if (is.null(fit)) return(NULL)

  s <- summary(fit)
  coef_df <- data.frame(
    variable = rownames(s$coefficients),
    coef     = s$coefficients[, "coef"],
    exp_coef = s$coefficients[, "exp(coef)"],
    se       = s$coefficients[, "se(coef)"],
    z        = s$coefficients[, "z"],
    p        = s$coefficients[, "Pr(>|z|)"],
    lower95  = s$conf.int[, "lower .95"],
    upper95  = s$conf.int[, "upper .95"],
    model    = label,
    stringsAsFactors = FALSE, row.names = NULL
  )

  fname <- paste0("sensitivity_", gsub(" ", "_", label), "_coefficients.csv")
  write_csv(coef_df, file.path(RESULTS, fname))
  cat("  Saved:", fname, "\n")

  # Print key results
  if (has_race) {
    black <- coef_df %>% filter(variable == "f.raceBlack")
    if (nrow(black) == 1) {
      cat(sprintf("  Black AOR: %.2f (%.2f-%.2f) p=%.2e\n",
                  black$exp_coef, black$lower95, black$upper95, black$p))
    }
  }

  return(coef_df)
}


# ══════════════════════════════════════════════════════════════════════
# S1: KDIGO Stage 1+ (Cr ≥1.5× baseline)
# ══════════════════════════════════════════════════════════════════════
if ("aki_kdigo1" %in% names(regression_bm)) {
  s1_data <- regression_bm
  s1_data$Treatment <- s1_data$aki_kdigo1
  s1_coefs <- run_sensitivity(s1_data, "S1_KDIGO_Stage1")
}

# ══════════════════════════════════════════════════════════════════════
# S2: KDIGO Stage 3 (Cr ≥3.0× baseline)
# ══════════════════════════════════════════════════════════════════════
if ("aki_kdigo3" %in% names(regression_bm)) {
  s2_data <- regression_bm
  s2_data$Treatment <- s2_data$aki_kdigo3
  s2_coefs <- run_sensitivity(s2_data, "S2_KDIGO_Stage3")
}

# ══════════════════════════════════════════════════════════════════════
# S3: 180-day observation window
# ══════════════════════════════════════════════════════════════════════
if ("aki_180d" %in% names(regression_bm)) {
  s3_data <- regression_bm
  s3_data$Treatment <- s3_data$aki_180d
  s3_coefs <- run_sensitivity(s3_data, "S3_180day_window")
}

# ══════════════════════════════════════════════════════════════════════
# S4: Exclude patients with baseline CKD
# ══════════════════════════════════════════════════════════════════════
if ("Renal_Disease" %in% names(regression_bm)) {
  s4_data <- regression_bm %>% filter(Renal_Disease == 0)
  # Remove Renal_Disease from como for this model
  como_s4 <- setdiff(como, "Renal_Disease")
  base_terms_s4 <- c("f.sex", "f.age")
  if (has_race) base_terms_s4 <- c(base_terms_s4, "f.race")
  if (has_ethnicity) base_terms_s4 <- c(base_terms_s4, "f.ethnicity")
  if (has_cancer) base_terms_s4 <- c(base_terms_s4, "f.cancer")
  if (has_ici) base_terms_s4 <- c(base_terms_s4, "f.ici")
  base_terms_s4 <- c(base_terms_s4, como_s4, nephro)

  # Need custom run since base_terms differ
  cat("\n", rep("=", 60), "\n", sep = "")
  cat("SENSITIVITY: S4_no_CKD\n")
  cat("  N=", nrow(s4_data), "\n")

  rhs_s4 <- paste(c(base_terms_s4, "strata(stratum)"), collapse = " + ")
  f_s4 <- as.formula(
    paste("Surv(rep(1, nrow(s4_data)), Treatment) ~", rhs_s4)
  )
  fit_s4 <- tryCatch(
    coxph(f_s4, data = s4_data, method = "exact"),
    error = function(e) { cat("  ERROR:", e$message, "\n"); NULL }
  )
  if (!is.null(fit_s4)) {
    s_s4 <- summary(fit_s4)
    s4_coefs <- data.frame(
      variable = rownames(s_s4$coefficients),
      coef = s_s4$coefficients[, "coef"],
      exp_coef = s_s4$coefficients[, "exp(coef)"],
      se = s_s4$coefficients[, "se(coef)"],
      z = s_s4$coefficients[, "z"],
      p = s_s4$coefficients[, "Pr(>|z|)"],
      lower95 = s_s4$conf.int[, "lower .95"],
      upper95 = s_s4$conf.int[, "upper .95"],
      model = "S4_no_CKD",
      stringsAsFactors = FALSE, row.names = NULL
    )
    write_csv(s4_coefs, file.path(RESULTS, "sensitivity_S4_no_CKD_coefficients.csv"))
    cat("  Saved: sensitivity_S4_no_CKD_coefficients.csv\n")
  }
}

# ══════════════════════════════════════════════════════════════════════
# S5: Anti-PD-1/PD-L1 monotherapy only
# ══════════════════════════════════════════════════════════════════════
if (has_ici) {
  s5_data <- regression_bm %>%
    filter(ici_regimen %in% c("anti_pd1", "anti_pdl1"))
  # Remove ICI regimen from model (since restricted)
  base_terms_s5 <- setdiff(base_terms, "f.ici")
  s5_coefs <- run_sensitivity(s5_data, "S5_PD1_PDL1_mono")
}


# ══════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════
cat("\n", rep("=", 60), "\n", sep = "")
cat("SENSITIVITY ANALYSES COMPLETE\n")
cat(rep("=", 60), "\n", sep = "")

# Collect Black AOR across sensitivities for comparison
if (has_race) {
  sens_files <- list.files(RESULTS, pattern = "^sensitivity_.*coefficients.csv",
                           full.names = TRUE)
  if (length(sens_files) > 0) {
    sens_all <- bind_rows(lapply(sens_files, read_csv, show_col_types = FALSE))
    black_summary <- sens_all %>%
      filter(variable == "f.raceBlack") %>%
      select(model, exp_coef, lower95, upper95, p) %>%
      arrange(model)

    write_csv(black_summary, file.path(RESULTS, "sensitivity_summary_comparison.csv"))
    cat("\n  Black AOR across sensitivity analyses:\n")
    print(black_summary)
    cat("  Saved: sensitivity_summary_comparison.csv\n")
  }
}

cat("\nDone.\n")
