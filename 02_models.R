#!/usr/bin/env Rscript
# ══════════════════════════════════════════════════════════════════════
# Post-ICI AKI × SDoH — Regression Models
# Adapted from aou_covid/02_models.R (Wang et al.)
#
# Model A: Base (demographics + NCI-CCI + cancer type + ICI class + nephrotoxins)
# Model B: Base + each SDoH domain (6 domain-by-domain models)
# Model C: Base + all SDoH domains (joint model)
# Race attenuation analysis
#
# Changes from COVID pipeline:
#   - NCI-CCI 14 conditions (no Malignancy, Metastatic, HIV standalone)
#   - Cancer type + ICI regimen + nephrotoxin flags replace vaccination/wave
#   - Same SDoH domains, same clogit framework
#
# Usage: Rscript 02_models.R ici_aki
# Input:  results/ici_aki/09_regression_base.csv
# Output: results/ici_aki/*_coefficients.csv, all_model_coefficients.csv
# ══════════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
  library(survival)
  library(dplyr)
  library(readr)
  library(sandwich)
  library(lmtest)
})

args <- commandArgs(trailingOnly = TRUE)
COHORT <- ifelse(length(args) >= 1, args[1], "ici_aki")
RESULTS <- file.path("results", COHORT)

cat(rep("=", 70), "\n", sep = "")
cat("POST-ICI AKI × SDoH — MODELS  [", toupper(COHORT), "]\n")
cat(rep("=", 70), "\n", sep = "")

# ── Load regression base ─────────────────────────────────────────
regression_bm <- read_csv(
  file.path(RESULTS, "09_regression_base.csv"),
  show_col_types = FALSE
)
cat("  Input:", RESULTS, "\n")
cat("  Regression data:", nrow(regression_bm), "rows,", ncol(regression_bm), "cols\n")
cat("  Cases:", sum(regression_bm$Treatment == 1),
    " Controls:", sum(regression_bm$Treatment == 0), "\n")

# ── Detect available columns ─────────────────────────────────────
has_race      <- "race" %in% names(regression_bm) &&
                 any(regression_bm$race != "Unknown", na.rm = TRUE)
has_ethnicity <- "ethnicity" %in% names(regression_bm) &&
                 any(regression_bm$ethnicity != "Unknown", na.rm = TRUE)
has_sdoh      <- "insurance_type" %in% names(regression_bm)
has_cancer    <- "cancer_type" %in% names(regression_bm)
has_ici       <- "ici_regimen" %in% names(regression_bm)

cat("  Features: race=", has_race, " ethnicity=", has_ethnicity,
    " sdoh=", has_sdoh, " cancer=", has_cancer, " ici=", has_ici, "\n")

# ── Factor encoding ──────────────────────────────────────────────
regression_bm$f.sex  <- factor(regression_bm$sex_at_birth,
                               levels = c("Male", "Female", "Other"))
regression_bm$f.age  <- factor(regression_bm$age_group,
                               levels = c("<45", "45-54", "55-64", "65+"))

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

# ── NCI-CCI comorbidity columns (14 conditions) ─────────────────
# Note: NO Malignancy, NO Metastatic_Solid_Tumor, NO standalone HIV
como <- c("Acute_MI", "History_MI",
          "Congestive_Heart_Failure",
          "Peripheral_Vascular_Disease", "Cerebrovascular_Disease",
          "Chronic_Pulmonary_Disease", "Dementia", "Paralysis",
          "Diabetes", "Diabetes_Complicated",
          "Renal_Disease",
          "Liver_Disease_Mild", "Liver_Disease_Moderate_Severe",
          "Peptic_Ulcer_Disease", "Rheumatic_Disease", "AIDS")

# Verify NCI-CCI columns exist; drop missing ones silently
como <- como[como %in% names(regression_bm)]
cat("  NCI-CCI conditions:", length(como), "\n")

# Nephrotoxin flags
nephro <- c("ppi_flag", "nsaid_flag", "acei_arb_flag", "diuretic_flag")
nephro <- nephro[nephro %in% names(regression_bm)]
cat("  Nephrotoxin flags:", length(nephro), "\n")

# ── Build base formula ───────────────────────────────────────────
base_terms <- c("f.sex", "f.age")
if (has_race) base_terms <- c(base_terms, "f.race")
if (has_ethnicity) base_terms <- c(base_terms, "f.ethnicity")
if (has_cancer) base_terms <- c(base_terms, "f.cancer")
if (has_ici) base_terms <- c(base_terms, "f.ici")
base_terms <- c(base_terms, como, nephro)

base_rhs <- paste(c(base_terms, "strata(stratum)"), collapse = " + ")
base_formula <- as.formula(paste("Surv(rep(1, nrow(regression_bm)), Treatment) ~",
                                  base_rhs))

cat("\n  Base formula RHS:", length(base_terms), "terms\n")


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

  # Extract coefficients
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

  # Save
  fname <- paste0(gsub(" ", "_", tolower(label)), "_coefficients.csv")
  write_csv(coef_df, file.path(RESULTS, fname))
  cat("  Saved:", fname, "\n")

  # Save RData
  rdata_fname <- paste0(gsub(" ", "_", tolower(label)), "_clogit.RData")
  save(fit, file = file.path(RESULTS, rdata_fname))

  return(coef_df)
}


# ══════════════════════════════════════════════════════════════════════
# MODEL A: BASE MODEL
# ══════════════════════════════════════════════════════════════════════
base_coefs <- fit_and_save(base_formula, "base")

# Print significant results
if (!is.null(base_coefs)) {
  sig <- base_coefs %>%
    filter(p < 0.05) %>%
    arrange(desc(abs(coef)))

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
sdoh_models <- list()

if (has_sdoh) {

  # ── Insurance ──────────────────────────────────────────────────
  regression_bm$f.insurance <- factor(regression_bm$insurance_type,
    levels = c("Employer", "Medicare", "Medicaid", "Private",
               "Other_Public", "Uninsured", "Missing"))
  ins_formula <- update(base_formula, . ~ . + f.insurance)
  ins_coefs <- fit_and_save(ins_formula, "insurance")
  all_coefs <- bind_rows(all_coefs, ins_coefs)

  # ── Income ─────────────────────────────────────────────────────
  regression_bm$f.income <- factor(regression_bm$income,
    levels = c("100K_150K", "200K_Plus", "150K_200K", "75K_100K",
               "50K_75K", "35K_50K", "25K_35K", "10K_25K",
               "Less_10K", "Missing"))
  inc_formula <- update(base_formula, . ~ . + f.income)
  inc_coefs <- fit_and_save(inc_formula, "income")
  all_coefs <- bind_rows(all_coefs, inc_coefs)

  # ── Education ──────────────────────────────────────────────────
  regression_bm$f.education <- factor(regression_bm$education,
    levels = c("College_Grad", "Advanced_Degree", "Some_College",
               "HS_GED", "Less_HS", "Never_Attended", "Missing"))
  edu_formula <- update(base_formula, . ~ . + f.education)
  edu_coefs <- fit_and_save(edu_formula, "education")
  all_coefs <- bind_rows(all_coefs, edu_coefs)

  # ── Employment ─────────────────────────────────────────────────
  regression_bm$f.employment <- factor(regression_bm$employment,
    levels = c("Employed", "Retired", "Disabled", "Unemployed_Looking",
               "Not_Working_Not_Looking", "Student", "Missing"))
  emp_formula <- update(base_formula, . ~ . + f.employment)
  emp_coefs <- fit_and_save(emp_formula, "employment")
  all_coefs <- bind_rows(all_coefs, emp_coefs)

  # ── Housing ────────────────────────────────────────────────────
  regression_bm$f.housing <- factor(regression_bm$housing,
    levels = c("Own", "Rent", "Other", "Missing"))
  hou_formula <- update(base_formula, . ~ . + f.housing)
  hou_coefs <- fit_and_save(hou_formula, "housing")
  all_coefs <- bind_rows(all_coefs, hou_coefs)

  # ── Housing stability ──────────────────────────────────────────
  regression_bm$f.stability <- factor(regression_bm$housing_stability,
    levels = c("Not_Worried", "Worried", "Missing"))
  stab_formula <- update(base_formula, . ~ . + f.stability)
  stab_coefs <- fit_and_save(stab_formula, "housing_stability")
  all_coefs <- bind_rows(all_coefs, stab_coefs)

  # ── Disability (mobility) ──────────────────────────────────────
  if ("disability_mobility" %in% names(regression_bm)) {
    dis_formula <- update(base_formula, . ~ . + disability_mobility)
    dis_coefs <- fit_and_save(dis_formula, "disability_mobility")
    all_coefs <- bind_rows(all_coefs, dis_coefs)
  }


  # ════════════════════════════════════════════════════════════════
  # MODEL C: JOINT SDoH MODEL
  # ════════════════════════════════════════════════════════════════
  joint_terms <- c("f.insurance", "f.income", "f.education",
                   "f.employment", "f.housing", "f.stability")
  if ("disability_mobility" %in% names(regression_bm)) {
    joint_terms <- c(joint_terms, "disability_mobility")
  }
  joint_rhs <- paste(c(base_terms, joint_terms, "strata(stratum)"),
                     collapse = " + ")
  joint_formula <- as.formula(
    paste("Surv(rep(1, nrow(regression_bm)), Treatment) ~", joint_rhs)
  )
  joint_coefs <- fit_and_save(joint_formula, "joint_sdoh")
  all_coefs <- bind_rows(all_coefs, joint_coefs)


  # ════════════════════════════════════════════════════════════════
  # RACE ATTENUATION ANALYSIS
  # ════════════════════════════════════════════════════════════════
  if (has_race) {
    cat("\n", rep("=", 60), "\n", sep = "")
    cat("RACE ATTENUATION ANALYSIS\n")
    cat(rep("=", 60), "\n", sep = "")

    # AOR for Black race in base model
    base_black <- base_coefs %>% filter(variable == "f.raceBlack")
    joint_black <- joint_coefs %>% filter(variable == "f.raceBlack")

    if (nrow(base_black) == 1 && nrow(joint_black) == 1) {
      aor_base  <- base_black$exp_coef
      aor_joint <- joint_black$exp_coef

      attenuation_pct <- (aor_base - aor_joint) / (aor_base - 1) * 100

      cat(sprintf("  Black AOR (base):  %.2f (%.2f-%.2f)\n",
                  aor_base, base_black$lower95, base_black$upper95))
      cat(sprintf("  Black AOR (joint): %.2f (%.2f-%.2f)\n",
                  aor_joint, joint_black$lower95, joint_black$upper95))
      cat(sprintf("  Attenuation: %.1f%%\n", attenuation_pct))

      # Domain-by-domain attenuation
      att_rows <- list()
      for (model_name in c("base", "insurance", "income", "education",
                            "employment", "housing", "housing_stability",
                            "disability_mobility", "joint_sdoh")) {
        row <- all_coefs %>%
          filter(model == model_name, variable == "f.raceBlack")
        if (nrow(row) == 1) {
          att <- (aor_base - row$exp_coef) / (aor_base - 1) * 100
          att_rows[[length(att_rows) + 1]] <- data.frame(
            model        = model_name,
            black_aor    = row$exp_coef,
            lower95      = row$lower95,
            upper95      = row$upper95,
            p            = row$p,
            attenuation  = att,
            stringsAsFactors = FALSE
          )
        }
      }
      att_df <- bind_rows(att_rows)
      write_csv(att_df, file.path(RESULTS, "race_attenuation_table.csv"))
      cat("  Saved: race_attenuation_table.csv\n")
      print(att_df)
    }
  }
}


# ══════════════════════════════════════════════════════════════════════
# SAVE ALL COEFFICIENTS
# ══════════════════════════════════════════════════════════════════════
write_csv(all_coefs, file.path(RESULTS, "all_model_coefficients.csv"))
cat("\n", rep("=", 60), "\n", sep = "")
cat(sprintf("  Combined: %d rows from %d models\n",
            nrow(all_coefs), length(unique(all_coefs$model))))


# ══════════════════════════════════════════════════════════════════════
# HEADLINE RESULTS
# ══════════════════════════════════════════════════════════════════════
cat("\n", rep("=", 60), "\n", sep = "")
cat("HEADLINE RESULTS [", toupper(COHORT), "]\n")
cat(rep("=", 60), "\n", sep = "")

sig_all <- all_coefs %>%
  filter(p < 0.05) %>%
  arrange(desc(exp_coef))

cat("\n  Significant across all models (p<0.05):\n")
for (i in seq_len(min(nrow(sig_all), 20))) {
  cat(sprintf("    %-45s AOR %.2f (%.2f-%.2f)  p=%.2e  [%s]\n",
              sig_all$variable[i], sig_all$exp_coef[i],
              sig_all$lower95[i], sig_all$upper95[i],
              sig_all$p[i], sig_all$model[i]))
}


# ══════════════════════════════════════════════════════════════════════
# UPLOAD TO BUCKET
# ══════════════════════════════════════════════════════════════════════
bucket_dir <- Sys.getenv("WORKSPACE_BUCKET")
if (nchar(bucket_dir) > 0) {
  dest <- paste0(bucket_dir, "/data/ici_aki_sdoh/")
  system(paste0("gsutil -m cp ", RESULTS, "/*.csv ", dest), intern = TRUE)
  system(paste0("gsutil -m cp ", RESULTS, "/*.RData ", dest), intern = TRUE)
  cat("  Uploaded to", dest, "\n")
}

cat("\n--- Session Info ---\n")
cat("R:", R.version.string, "\n")
cat("survival:", as.character(packageVersion("survival")), "\n")
cat("dplyr:", as.character(packageVersion("dplyr")), "\n")
cat("readr:", as.character(packageVersion("readr")), "\n")

cat("\nDone.\n")
