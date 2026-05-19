#!/usr/bin/env Rscript
# ══════════════════════════════════════════════════════════════════════
# Post-ICI AKI × SDoH — Sensitivity Analyses (v5)
#
# S1: AKI >=0.3 mg/dL absolute increase (Delta Cr)
# S2: AKI >=2.0x baseline (KDIGO Stage 2)
# S3: AKI >=3.0x baseline (KDIGO Stage 3)
# S4: 180-day follow-up window (vs 365d)
# S5: Mono-ICI only (exclude CTLA-4 containing regimens)
#
# Usage: Rscript 03_sensitivity.R aou
#        Rscript 03_sensitivity.R inpc
# ══════════════════════════════════════════════════════════════════════

# ── Configuration ────────────────────────────────────────────────────
USE_NCI_INDEX <- FALSE
# ─────────────────────────────────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1 || !args[1] %in% c("aou", "inpc")) {
  stop("Usage: Rscript 03_sensitivity.R [aou|inpc]")
}
MODE <- args[1]
COHORT <- ifelse(MODE == "aou", "ici_aki", "inpc")

# ── R library path (Quartz HPC compatibility) ────────────────────────
user_lib <- file.path(Sys.getenv("HOME"), "R", "library")
dir.create(user_lib, showWarnings = FALSE, recursive = TRUE)
.libPaths(c(user_lib, .libPaths()))

required_pkgs <- c("survival", "dplyr", "readr")
for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, lib = user_lib, repos = "https://cloud.r-project.org")
  }
}
library(survival)
library(dplyr)
library(readr)

cat("\n", rep("=", 70), "\n", sep = "")
cat("POST-ICI AKI × SDoH — SENSITIVITY ANALYSES (v5)\n")
cat("Cohort:", MODE, "->", COHORT, "\n")
cat("CCI mode:", ifelse(USE_NCI_INDEX, "NCI continuous index", "Charlson integer score"), "\n")
cat(rep("=", 70), "\n", sep = "")


# ══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════

if (dir.exists(file.path("results", COHORT))) {
  RESULTS <- file.path("results", COHORT)
} else if (dir.exists(COHORT)) {
  RESULTS <- COHORT
} else {
  stop("Cannot find results directory for cohort: ", COHORT)
}

matched_files <- c(
  file.path(RESULTS, "09_regression_base.csv"),
  file.path(RESULTS, "08_regression_base_matched.csv"),
  file.path(RESULTS, "09_regression_base_matched.csv"),
  file.path(RESULTS, "08_regression_base.csv")
)
matched_file <- NULL
for (f in matched_files) {
  if (file.exists(f)) { matched_file <- f; break }
}
if (is.null(matched_file)) stop("No matched regression base found.")
cat("Loading:", matched_file, "\n")

regression_bm <- read_csv(matched_file, show_col_types = FALSE)
cat("  Rows:", nrow(regression_bm), " Cases:", sum(regression_bm$severity == 1), "\n")


# ══════════════════════════════════════════════════════════════════════
# FACTOR SETUP (same as 02_models.R)
# ══════════════════════════════════════════════════════════════════════

if ("severity" %in% names(regression_bm)) {
  regression_bm$Treatment <- regression_bm$severity
}

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

has_cancer <- any(c("cancer_type_collapsed", "cancer_type") %in% names(regression_bm))
if (has_cancer) {
  cancer_col <- ifelse("cancer_type_collapsed" %in% names(regression_bm),
                        "cancer_type_collapsed", "cancer_type")
  regression_bm[[cancer_col]][regression_bm[[cancer_col]] == "Renal_Cell"] <- "Other"
  regression_bm$f.cancer <- factor(regression_bm[[cancer_col]],
    levels = c("Lung", "Melanoma", "Other"))
}

has_ici <- any(c("ici_collapsed", "ici_regimen") %in% names(regression_bm))
if (has_ici) {
  if ("ici_collapsed" %in% names(regression_bm)) {
    regression_bm$f.ici <- factor(regression_bm$ici_collapsed,
      levels = c("anti_pd1", "anti_pdl1", "ctla4_containing"))
  } else {
    regression_bm$f.ici <- factor(regression_bm$ici_regimen)
  }
}


# ══════════════════════════════════════════════════════════════════════
# NCI-CCI COVARIATE (v2 fix — same logic as 02_models.R)
# ══════════════════════════════════════════════════════════════════════

if (USE_NCI_INDEX && "nci_index" %in% names(regression_bm)) {
  como_terms <- "nci_index"
  cat("  CCI covariate: nci_index (NCI continuous)\n")
} else if ("nci_cci_score" %in% names(regression_bm)) {
  como_terms <- "nci_cci_score"
  cat("  CCI covariate: nci_cci_score (Charlson integer, corrected v2)\n")
} else {
  como <- c("Acute_MI", "History_MI", "Congestive_Heart_Failure",
            "Peripheral_Vascular_Disease", "Cerebrovascular_Disease",
            "Chronic_Pulmonary_Disease", "Dementia", "Paralysis",
            "Diabetes", "Diabetes_Complicated", "Renal_Disease",
            "Liver_Disease_Mild", "Liver_Disease_Moderate_Severe",
            "Peptic_Ulcer_Disease", "Rheumatic_Disease", "AIDS")
  como_terms <- como[como %in% names(regression_bm)]
  cat("  CCI covariate: individual flags (", length(como_terms), ")\n")
}

nephro <- c("ppi", "nsaid", "acei_arb", "diuretic")
nephro <- nephro[nephro %in% names(regression_bm)]


# ══════════════════════════════════════════════════════════════════════
# BUILD BASE FORMULA
# ══════════════════════════════════════════════════════════════════════

base_terms <- c("f.sex", "f.age")
if (has_race) base_terms <- c(base_terms, "f.race")
if (has_ethnicity) base_terms <- c(base_terms, "f.ethnicity")
if (has_cancer) base_terms <- c(base_terms, "f.cancer")
if (has_ici) base_terms <- c(base_terms, "f.ici")
base_terms <- c(base_terms, como_terms, nephro)


# ══════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════

run_sensitivity <- function(data, outcome_col, label) {
  cat("\n", rep("=", 60), "\n", sep = "")
  cat("SENSITIVITY:", label, "\n")
  cat(rep("=", 60), "\n", sep = "")

  data$Treatment <- data[[outcome_col]]
  n_cases <- sum(data$Treatment == 1, na.rm = TRUE)
  n_controls <- sum(data$Treatment == 0, na.rm = TRUE)
  cat("  Cases:", n_cases, " Controls:", n_controls, "\n")

  if (n_cases < 10) {
    cat("  SKIP: too few cases (<10)\n")
    return(NULL)
  }

  rhs <- paste(c(base_terms, "strata(stratum)"), collapse = " + ")
  formula_obj <- as.formula(paste(
    "Surv(rep(1, nrow(data)), Treatment) ~", rhs
  ))

  fit <- tryCatch(
    coxph(formula_obj, data = data, method = "exact"),
    error = function(e) {
      cat("  ERROR:", e$message, "\n")
      return(NULL)
    }
  )
  if (is.null(fit)) return(NULL)

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

  # Print significant
  sig <- coef_df %>% filter(p < 0.05) %>% arrange(desc(abs(coef)))
  if (nrow(sig) > 0) {
    cat("  Significant (p<0.05):\n")
    for (i in seq_len(min(nrow(sig), 10))) {
      cat(sprintf("    %-40s AOR %.2f (%.2f-%.2f) p=%.3f\n",
                  sig$variable[i], sig$exp_coef[i],
                  sig$lower95[i], sig$upper95[i], sig$p[i]))
    }
  }

  return(coef_df)
}


# ══════════════════════════════════════════════════════════════════════
# RUN ALL SENSITIVITY ANALYSES
# ══════════════════════════════════════════════════════════════════════

all_sens <- data.frame()

# S1: AKI ≥0.3 mg/dL absolute increase
if ("aki_delta03" %in% names(regression_bm)) {
  s1 <- run_sensitivity(regression_bm, "aki_delta03", "S1_delta_0.3")
  if (!is.null(s1)) all_sens <- bind_rows(all_sens, s1)
}

# S2: AKI ≥2.0× baseline (KDIGO Stage 2)
if ("aki_kdigo2" %in% names(regression_bm)) {
  s2 <- run_sensitivity(regression_bm, "aki_kdigo2", "S2_KDIGO2")
  if (!is.null(s2)) all_sens <- bind_rows(all_sens, s2)
}

# S3: AKI ≥3.0× baseline (KDIGO Stage 3)
if ("aki_kdigo3" %in% names(regression_bm)) {
  s3 <- run_sensitivity(regression_bm, "aki_kdigo3", "S3_KDIGO3")
  if (!is.null(s3)) all_sens <- bind_rows(all_sens, s3)
}

# S4: 180-day follow-up window
if ("aki_180d" %in% names(regression_bm)) {
  s4 <- run_sensitivity(regression_bm, "aki_180d", "S4_180day")
  if (!is.null(s4)) all_sens <- bind_rows(all_sens, s4)
}

# S5: Mono-ICI only (exclude combo regimens)
if ("ici_regimen" %in% names(regression_bm) || "ici_collapsed" %in% names(regression_bm)) {
  mono_col <- ifelse("ici_collapsed" %in% names(regression_bm), "ici_collapsed", "ici_regimen")
  # v5: ici_collapsed uses "ctla4_containing" for combo + ctla4 mono
  mono_data <- regression_bm[!grepl("ctla4", regression_bm[[mono_col]], ignore.case = TRUE), ]

  if (nrow(mono_data) > 0 && sum(mono_data$severity == 1) >= 10) {
    # Remove ICI factor (only one level after filtering)
    mono_base_terms <- base_terms[!grepl("f.ici", base_terms)]
    rhs <- paste(c(mono_base_terms, "strata(stratum)"), collapse = " + ")
    mono_formula <- as.formula(paste(
      "Surv(rep(1, nrow(mono_data)), Treatment) ~", rhs
    ))
    mono_data$Treatment <- mono_data$severity

    fit <- tryCatch(
      coxph(mono_formula, data = mono_data, method = "exact"),
      error = function(e) { cat("  S5 ERROR:", e$message, "\n"); NULL }
    )

    if (!is.null(fit)) {
      s <- summary(fit)
      s5 <- data.frame(
        variable = rownames(s$coefficients),
        coef = s$coefficients[, "coef"],
        exp_coef = s$coefficients[, "exp(coef)"],
        se = s$coefficients[, "se(coef)"],
        z = s$coefficients[, "z"],
        p = s$coefficients[, "Pr(>|z|)"],
        lower95 = s$conf.int[, "lower .95"],
        upper95 = s$conf.int[, "upper .95"],
        model = "S5_mono_ICI",
        stringsAsFactors = FALSE, row.names = NULL
      )
      write_csv(s5, file.path(RESULTS, "s5_mono_ici_coefficients.csv"))
      all_sens <- bind_rows(all_sens, s5)
      cat("  S5: Mono-ICI — ", nrow(mono_data), " patients\n")
    }
  }
}


# ══════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════

if (nrow(all_sens) > 0) {
  write_csv(all_sens, file.path(RESULTS, "all_sensitivity_coefficients.csv"))

  # Black race comparison across sensitivities
  if (has_race) {
    black_summary <- all_sens %>%
      filter(variable == "f.raceBlack") %>%
      select(model, exp_coef, lower95, upper95, p) %>%
      arrange(model)

    if (nrow(black_summary) > 0) {
      write_csv(black_summary, file.path(RESULTS, "sensitivity_summary_comparison.csv"))
      cat("\n  Black AOR across sensitivity analyses:\n")
      print(black_summary)
      cat("  Saved: sensitivity_summary_comparison.csv\n")
    }
  }

  # NCI-CCI score comparison across sensitivities
  cci_col <- ifelse(USE_NCI_INDEX, "nci_index", "nci_cci_score")
  cci_summary <- all_sens %>%
    filter(variable == cci_col) %>%
    select(model, exp_coef, lower95, upper95, p) %>%
    arrange(model)

  if (nrow(cci_summary) > 0) {
    cat("\n  NCI-CCI across sensitivity analyses:\n")
    print(cci_summary)
  }
}

cat("\n", rep("=", 70), "\n", sep = "")
cat("SENSITIVITY ANALYSES COMPLETE (v5)\n")
cat(rep("=", 70), "\n", sep = "")
cat("Done.\n")
