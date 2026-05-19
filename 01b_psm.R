#!/usr/bin/env Rscript
# ══════════════════════════════════════════════════════════════════════
# Post-ICI AKI × SDoH — Propensity Score Matching (v5)
#
# Usage: Rscript 01b_psm.R aou      # reads results/ici_aki/
#        Rscript 01b_psm.R inpc     # reads results/inpc/
#
# PS model: severity ~ enrollment_days + n_diagnoses + ehr_length_days
# Matching: 1:4 nearest-neighbor, with replacement, 0.2 SD caliper
# ══════════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
  user_lib <- Sys.getenv("R_LIBS_USER", paste0(Sys.getenv("HOME"), "/R/library"))
  dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
  .libPaths(c(user_lib, .libPaths()))

  required <- c("MatchIt", "cobalt", "dplyr", "readr", "survival")
  missing <- required[!required %in% installed.packages()[, "Package"]]
  if (length(missing) > 0) {
    cat("  Installing:", paste(missing, collapse = ", "), "\n")
    install.packages(missing, lib = user_lib, repos = "https://cloud.r-project.org", quiet = TRUE)
  }
  library(MatchIt)
  library(cobalt)
  library(dplyr)
  library(readr)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1 || !args[1] %in% c("aou", "inpc")) {
  stop("Usage: Rscript 01b_psm.R [aou|inpc]")
}
MODE <- args[1]
RESULTS <- ifelse(MODE == "aou", "results/ici_aki", "results/inpc")

cat(rep("=", 60), "\n", sep = "")
cat("POST-ICI AKI × SDoH — PSM  [", toupper(MODE), "]\n")
cat(rep("=", 60), "\n", sep = "")
cat("  Input/Output:", RESULTS, "\n")

# ── Load pre-matching base ───────────────────────────────────────
base_file <- file.path(RESULTS, "07_pre_matching_base.csv")
if (!file.exists(base_file)) {
  stop("  ERROR: ", base_file, " not found. Run python 01_etl.py ", MODE, " first.")
}

df <- read_csv(base_file, show_col_types = FALSE)
cat("  Loaded:", nrow(df), "rows,", ncol(df), "cols\n")
cat("  Cases:", sum(df$severity == 1), " Controls:", sum(df$severity == 0), "\n")

# ── PS Model ─────────────────────────────────────────────────────
match_df <- df %>%
  select(person_id, severity, enrollment_days, n_diagnoses, ehr_length_days) %>%
  filter(complete.cases(.))

cat("  Complete-case for matching:", nrow(match_df), "\n")

# ── Run MatchIt ──────────────────────────────────────────────────
cat("\n  Running MatchIt (1:4 NN, replacement, 0.2 SD caliper)...\n")

m <- matchit(
  severity ~ enrollment_days + n_diagnoses + ehr_length_days,
  data        = match_df,
  method      = "nearest",
  distance    = "glm",
  ratio       = 4,
  replace     = TRUE,
  caliper     = 0.2,
  std.caliper = TRUE
)

cat("  MatchIt complete.\n")
print(summary(m))

# ── Extract matched pairs ────────────────────────────────────────
matched_pairs <- get_matches(m, data = match_df)

matched <- data.frame(
  person_id = matched_pairs$person_id,
  Treatment = matched_pairs$severity,
  stratum   = as.integer(matched_pairs$subclass),
  stringsAsFactors = FALSE
)

n_cases  <- sum(matched$Treatment == 1)
n_ctrls  <- sum(matched$Treatment == 0)
n_strata <- length(unique(matched$stratum))
cat(sprintf("  Cases: %s | Control rows: %s | Strata: %s | Ratio: 1:%.1f\n",
            format(n_cases, big.mark = ","),
            format(n_ctrls, big.mark = ","),
            format(n_strata, big.mark = ","),
            n_ctrls / n_cases))

n_cases_total <- sum(match_df$severity == 1)
n_dropped <- n_cases_total - n_cases
cat(sprintf("  Dropped (no match within caliper): %d\n", n_dropped))

write_csv(matched, file.path(RESULTS, "08_matched_cohort.csv"))
cat("  Saved: 08_matched_cohort.csv\n")

# ── Control reuse statistics ─────────────────────────────────────
ctrl_rows <- matched[matched$Treatment == 0, ]
ctrl_reuse <- table(ctrl_rows$person_id)
n_unique <- length(ctrl_reuse)
med_reuse <- median(ctrl_reuse)
q1_reuse  <- quantile(ctrl_reuse, 0.25)
q3_reuse  <- quantile(ctrl_reuse, 0.75)
max_reuse <- max(ctrl_reuse)

cat(sprintf("  Control reuse: %s unique, median %.0f (IQR %.0f-%.0f), max %d\n",
            format(n_unique, big.mark = ","),
            med_reuse, q1_reuse, q3_reuse, max_reuse))

reuse_df <- data.frame(
  metric = c("n_unique_controls", "n_control_rows", "median_reuse",
             "iqr_lower", "iqr_upper", "max_reuse", "caliper_sd",
             "n_cases_dropped"),
  value  = c(n_unique, nrow(ctrl_rows), med_reuse,
             q1_reuse, q3_reuse, max_reuse, 0.2, n_dropped)
)
write_csv(reuse_df, file.path(RESULTS, "08b_control_reuse.csv"))

# ══════════════════════════════════════════════════════════════════════
# BALANCE DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════
cat("\n", rep("=", 60), "\n", sep = "")
cat("BALANCE DIAGNOSTICS\n")
cat(rep("=", 60), "\n", sep = "")

bal_pre <- bal.tab(m, un = TRUE, stats = c("mean.diffs", "variance.ratios"))
cat("\n  Pre/post matching balance:\n")
print(bal_pre)

smd_tab <- bal_pre$Balance
smd_df <- data.frame(
  variable         = rownames(smd_tab),
  smd_unadjusted   = smd_tab$Diff.Un,
  smd_adjusted     = smd_tab$Diff.Adj,
  var_ratio_unadj  = if ("V.Ratio.Un" %in% names(smd_tab)) smd_tab$V.Ratio.Un else NA,
  var_ratio_adj    = if ("V.Ratio.Adj" %in% names(smd_tab)) smd_tab$V.Ratio.Adj else NA,
  stringsAsFactors = FALSE, row.names = NULL
)
write_csv(smd_df, file.path(RESULTS, "08c_smd_balance.csv"))
cat("  Saved: 08c_smd_balance.csv\n")

# ══════════════════════════════════════════════════════════════════════
# BUILD REGRESSION BASE
# ══════════════════════════════════════════════════════════════════════
cat("\n", rep("=", 60), "\n", sep = "")
cat("BUILD REGRESSION BASE\n")
cat(rep("=", 60), "\n", sep = "")

regression_base <- matched %>%
  inner_join(df, by = "person_id")

stopifnot(all(regression_base$Treatment == regression_base$severity))

write_csv(regression_base, file.path(RESULTS, "09_regression_base.csv"))
cat(sprintf("  Regression base: %s rows, %d cols\n",
            format(nrow(regression_base), big.mark = ","),
            ncol(regression_base)))
cat("  Saved: 09_regression_base.csv\n")
cat("  Next: Rscript 02_models.R", MODE, "\n")

cat("\n--- Session Info ---\n")
cat("R:", R.version.string, "\n")
cat("MatchIt:", as.character(packageVersion("MatchIt")), "\n")
cat("cobalt:", as.character(packageVersion("cobalt")), "\n")
cat("\nDone.\n")
