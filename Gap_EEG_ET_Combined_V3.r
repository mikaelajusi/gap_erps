# =============================================================================
# GAP-OVERLAP TASK ANALYSIS — Combined Script V3
# Q1. Facilitation & Disengagement in RT (one-sample t-tests on RT only)
# Q2. Neural Correlates: Gap_amp & Overlap_amp (one-sample t-tests)
# Q3. Age Trajectories: RT relative (lm) + ERP amplitude (GAM)
# Q4. Diagnostic Differences: NDD & affected_group moderation
# Q5. ERP Amplitude → RT Coupling (four explicit pairings)
#
# Analytic outcomes:
#   facilitation_rt  (Baseline_rt - Gap_rt)     — RT relative, ICC ≈ 0%, lm
#   disengagement_rt (Overlap_rt - Baseline_rt) — RT relative, ICC ≈ 0%, lm
#   Gap_amp          — ERP raw amplitude,        ICC ≈ 0%, lm / GAM (no fam_id RE)
#   Overlap_amp      — ERP raw amplitude,        ICC > 3%, lmer / GAM + s(fam_id)
#   Gap_rt           — RT raw,                   ICC ≈ 0%, lm         [Q5 only]
#   Overlap_rt       — RT raw,                   ICC > 3%, lmer        [Q5 only]
#
# BH-FDR family summary:
#   Q1:        10 tests (2 RT outcomes × 5 conditions)  → Q1_one_sample_BH_FDR.csv
#   Q2:        10 tests (2 ERP outcomes × 5 conditions) → Q2_one_sample_BH_FDR.csv
#   Q3 main:    4 tests (age p-value per outcome)        → Q3_age_BH_FDR.csv
#   Q4_NDD_main: 4 tests (NDD main LRT, 1 per outcome)  → Q4_NDD_BH_FDR_summary.csv
#   Q4_NDD_int:  4 tests (NDD×age LRT, 1 per outcome)   → Q4_NDD_BH_FDR_summary.csv
#   Q4_grp_main: 4 tests (group main LRT)               → Q4_group_BH_FDR_summary.csv
#   Q4_grp_int:  4 tests (group×age LRT)                → Q4_group_BH_FDR_summary.csv
#   Q5:          4 tests (ERP amp predictor p per model) → Q5_ERP_RT_BH_FDR.csv
#
# ICC-based RE rule (from null-model ICC):
#   ICC ≈ 0%  → lm() / GAM without fam_id RE:
#               facilitation_rt, disengagement_rt, Gap_amp, Gap_rt
#   ICC > 3%  → lmer() + (1|fam_id) / GAM + s(fam_id, bs="re"):
#               Overlap_amp, Overlap_rt
# =============================================================================

library(mgcv)
library(lme4)
library(lmerTest)   # Satterthwaite p-values in summary() for lmer
library(ggplot2)
library(dplyr)
library(tidyr)
library(emmeans)
library(patchwork)
library(car)
library(moments)
library(effectsize)
library(performance)
library(gt)
library(scales)
library(lmtest)
library(ggeffects)
library(stringr)
library(broom)
library(ggsignif)

set.seed(333)


# =============================================================================
# 1. DIRECTORIES & DATA LOAD
# =============================================================================

setwd("C:/Users/gabot/OneDrive - McGill University/Desktop/Github_repos/q1k_neurosubs/code/linear_models/go_task_full")

for (d in c("output", "output/figures", "output/tables",
            "output/diagnostics", "output/supplementary")) {
  dir.create(d, showWarnings = FALSE, recursive = TRUE)
}

data_path <- "C:/Users/gabot/OneDrive - McGill University/Desktop/github_repos//q1k_multimodal_subtypes/outputs/gap/erp_rt_combined.csv"
raw_data  <- read.csv(data_path, stringsAsFactors = FALSE)

cat("Raw data dimensions:", nrow(raw_data), "rows ×", ncol(raw_data), "cols\n")


# =============================================================================
# 2. DATA PREPARATION
# =============================================================================

data <- raw_data

# Extract family ID from subject code prefix
data$fam_id <- str_extract(data$subject, "^[0-9]+")
cat("Unique families:", length(unique(data$fam_id)), "\n")
cat("Total subjects:", nrow(data), "\n")

# Cast relative RT outcomes
data$facilitation_rt  <- as.numeric(data$facilitation_rt)
data$disengagement_rt <- as.numeric(data$disengagement_rt)

# Cast raw ERP amplitude outcomes
data$Gap_amp     <- as.numeric(data$Gap_amp)
data$Overlap_amp <- as.numeric(data$Overlap_amp)

# Cast raw RT outcomes (needed for Q5)
data$Gap_rt     <- as.numeric(data$Gap_rt)
data$Overlap_rt <- as.numeric(data$Overlap_rt)
data$Baseline_rt <- as.numeric(data$Baseline_rt)

# Verify pre-computed RT difference scores match raw components
data$facilitation_rt_check  <- data$Baseline_rt - data$Gap_rt
data$disengagement_rt_check <- data$Overlap_rt  - data$Baseline_rt
rt_discrepancy <- data %>%
  filter(!is.na(facilitation_rt) & !is.na(facilitation_rt_check)) %>%
  summarise(
    max_facil_diff  = max(abs(facilitation_rt  - facilitation_rt_check),  na.rm = TRUE),
    max_diseng_diff = max(abs(disengagement_rt - disengagement_rt_check), na.rm = TRUE)
  )
cat("\nRT difference-score verification (should be ≈ 0):\n"); print(rt_discrepancy)
data$facilitation_rt_check  <- NULL
data$disengagement_rt_check <- NULL

# Factor coding
data$ndd <- factor(data$ndd, levels = c(0, 1), labels = c("Non-NDD", "NDD"))
data$affected_group <- factor(data$affected_group,
                              levels = c("non-affected", "affected", "asd"),
                              labels = c("Non-Affected", "Affected (non-ASD)", "ASD"))
data$sex  <- factor(data$sex,  levels = c("Female", "Male"))
data$site <- droplevels(
  factor(ifelse(data$site == "" | is.na(data$site), NA_character_, as.character(data$site)))
)
data$fam_id <- factor(data$fam_id)
data$age    <- as.numeric(data$eeg_age)

age_bin_order <- c("Children\n(\u226411)", "Adolescents\n(12\u201317)", "Adults\n(18\u201344)", "Older Adults\n(45+)")
data$age_bin  <- factor(data$age_bin, levels = age_bin_order, ordered = TRUE)

cat("\nFactor level counts:\n")
cat("NDD:            "); print(table(data$ndd,            useNA = "always"))
cat("Affected group: "); print(table(data$affected_group, useNA = "always"))
cat("Sex:            "); print(table(data$sex,            useNA = "always"))
cat("Site:           "); print(table(data$site,           useNA = "always"))
cat("Age bin:        "); print(table(data$age_bin,        useNA = "always"))

# Drop rows missing required model covariates
data <- data[!is.na(data$age), ]
data <- data[!is.na(data$ndd), ]
cat("\nFinal analytic sample after dropping missing age/NDD: N =", nrow(data), "\n")


# =============================================================================
# 3. MISSING DATA REPORT
# =============================================================================

analytic_outcomes <- c("facilitation_rt", "disengagement_rt",
                       "Gap_amp", "Overlap_amp",
                       "Gap_rt",  "Overlap_rt")

missing_report <- data.frame(
  Variable    = analytic_outcomes,
  N_total     = nrow(data),
  N_observed  = sapply(analytic_outcomes, function(v) sum(!is.na(data[[v]]))),
  N_missing   = sapply(analytic_outcomes, function(v) sum( is.na(data[[v]]))),
  Pct_missing = sapply(analytic_outcomes, function(v) round(100 * mean(is.na(data[[v]])), 2))
)
cat("\n=== Missing Data Report ===\n"); print(missing_report)
write.csv(missing_report, "output/tables/missing_data_report.csv", row.names = FALSE)

# Missing data characterisation for primary RT outcome
cat("\n--- Missing characterisation: facilitation_rt ---\n")
data$miss_flag <- is.na(data$facilitation_rt)
t_age_miss   <- t.test(age ~ miss_flag, data = data)
chi_ndd_miss <- chisq.test(table(data$ndd, data$miss_flag))
chi_sex_miss <- chisq.test(table(data$sex, data$miss_flag))
cat("Age (missers vs observed): t =", round(t_age_miss$statistic, 2),
    ", p =", round(t_age_miss$p.value, 4), "\n")
cat("NDD × missing: chi2 =", round(chi_ndd_miss$statistic, 2),
    ", p =", round(chi_ndd_miss$p.value, 4), "\n")
cat("Sex × missing: chi2 =", round(chi_sex_miss$statistic, 2),
    ", p =", round(chi_sex_miss$p.value, 4), "\n")
data$miss_flag <- NULL


# =============================================================================
# 4. OUTLIER DETECTION (3 SD rule)
# =============================================================================

detect_outliers_3sd <- function(x) {
  m   <- mean(x, na.rm = TRUE)
  s   <- sd(x,   na.rm = TRUE)
  out <- abs(x - m) > 3 * s
  out[is.na(x)] <- FALSE   # NA values are NOT flagged as outliers
  return(out)
}

out_facil_rt  <- detect_outliers_3sd(data$facilitation_rt)
out_diseng_rt <- detect_outliers_3sd(data$disengagement_rt)
out_gap_amp   <- detect_outliers_3sd(data$Gap_amp)
out_ovlp_amp  <- detect_outliers_3sd(data$Overlap_amp)
out_gap_rt    <- detect_outliers_3sd(data$Gap_rt)
out_ovlp_rt   <- detect_outliers_3sd(data$Overlap_rt)

outlier_report <- data.frame(
  Variable     = analytic_outcomes,
  N_outliers   = c(sum(out_facil_rt),  sum(out_diseng_rt),
                   sum(out_gap_amp),   sum(out_ovlp_amp),
                   sum(out_gap_rt),    sum(out_ovlp_rt)),
  Pct_outliers = round(100 * c(mean(out_facil_rt),  mean(out_diseng_rt),
                                mean(out_gap_amp),   mean(out_ovlp_amp),
                                mean(out_gap_rt),    mean(out_ovlp_rt)), 2)
)
cat("\n=== Outlier Report (3 SD) ===\n"); print(outlier_report)
write.csv(outlier_report, "output/tables/outlier_report.csv", row.names = FALSE)

# To Winsorise or remove outliers, uncomment the relevant lines:
# data$facilitation_rt [out_facil_rt]  <- NA
# data$disengagement_rt[out_diseng_rt] <- NA
# data$Gap_amp         [out_gap_amp]   <- NA
# data$Overlap_amp     [out_ovlp_amp]  <- NA
# data$Gap_rt          [out_gap_rt]    <- NA
# data$Overlap_rt      [out_ovlp_rt]   <- NA


# =============================================================================
# 5. SAMPLE DESCRIPTIVES
# =============================================================================

cat("\n=== Overall Sample ===\n")
cat("N =", nrow(data),
    "| Age: M =", round(mean(data$age, na.rm = TRUE), 1),
    "SD =", round(sd(data$age,   na.rm = TRUE), 1),
    "Range:", round(min(data$age, na.rm = TRUE), 1),
    "-", round(max(data$age, na.rm = TRUE), 1), "\n")

make_desc_summary <- function(grp_var) {
  data %>%
    group_by(.data[[grp_var]]) %>%
    summarise(N             = n(),
              Age_M         = round(mean(age, na.rm = TRUE), 2),
              Age_SD        = round(sd(age,   na.rm = TRUE), 2),
              Pct_Female    = round(100 * mean(sex == "Female", na.rm = TRUE), 1),
              facil_rt_M    = round(mean(facilitation_rt,  na.rm = TRUE), 2),
              facil_rt_SD   = round(sd(facilitation_rt,    na.rm = TRUE), 2),
              diseng_rt_M   = round(mean(disengagement_rt, na.rm = TRUE), 2),
              diseng_rt_SD  = round(sd(disengagement_rt,   na.rm = TRUE), 2),
              Gap_amp_M     = round(mean(Gap_amp,           na.rm = TRUE), 2),
              Gap_amp_SD    = round(sd(Gap_amp,             na.rm = TRUE), 2),
              Ovlp_amp_M    = round(mean(Overlap_amp,       na.rm = TRUE), 2),
              Ovlp_amp_SD   = round(sd(Overlap_amp,         na.rm = TRUE), 2),
              Gap_rt_M      = round(mean(Gap_rt,            na.rm = TRUE), 2),
              Gap_rt_SD     = round(sd(Gap_rt,              na.rm = TRUE), 2),
              Ovlp_rt_M     = round(mean(Overlap_rt,        na.rm = TRUE), 2),
              Ovlp_rt_SD    = round(sd(Overlap_rt,          na.rm = TRUE), 2),
              .groups = "drop")
}

desc_by_ndd    <- make_desc_summary("ndd")
desc_by_group  <- make_desc_summary("affected_group")
desc_by_agebin <- make_desc_summary("age_bin")

cat("\n--- Descriptives by NDD ---\n");            print(desc_by_ndd)
cat("\n--- Descriptives by affected_group ---\n"); print(desc_by_group)
cat("\n--- Descriptives by age_bin ---\n");        print(desc_by_agebin)

write.csv(desc_by_ndd,    "output/tables/descriptives_by_NDD.csv",            row.names = FALSE)
write.csv(desc_by_group,  "output/tables/descriptives_by_affected_group.csv", row.names = FALSE)
write.csv(desc_by_agebin, "output/tables/descriptives_by_age_bin.csv",        row.names = FALSE)

# Age differences across groups
age_aov <- aov(age ~ affected_group, data = data)
cat("\nAge × affected_group ANOVA:\n"); print(summary(age_aov))
cat("\nTukey HSD:\n"); print(TukeyHSD(age_aov))
t_age_ndd <- t.test(age ~ ndd, data = data)
cat("\nAge NDD vs Non-NDD: t =", round(t_age_ndd$statistic, 2),
    ", p =", round(t_age_ndd$p.value, 4), "\n")


# =============================================================================
# 6. FAMILY-LEVEL ICC (null-model clustering)
# =============================================================================

fit_null_icc <- function(df, outcome, cluster = "fam_id") {
  dat <- df[, c(outcome, cluster)]
  dat <- dat[!is.na(dat[[outcome]]) & !is.na(dat[[cluster]]), , drop = FALSE]
  dat[[cluster]] <- droplevels(as.factor(dat[[cluster]]))

  if (nrow(dat) < 10 || nlevels(dat[[cluster]]) < 2) {
    return(data.frame(outcome = outcome, N = nrow(dat),
                      n_clusters = nlevels(dat[[cluster]]),
                      var_cluster = NA_real_, var_resid = NA_real_,
                      ICC = NA_real_, singular = NA,
                      status = "insufficient_data"))
  }

  fml <- as.formula(paste0(outcome, " ~ 1 + (1|", cluster, ")"))
  mod <- tryCatch(lme4::lmer(fml, data = dat, REML = TRUE), error = function(e) e)

  if (inherits(mod, "error")) {
    return(data.frame(outcome = outcome, N = nrow(dat),
                      n_clusters = nlevels(dat[[cluster]]),
                      var_cluster = NA_real_, var_resid = NA_real_,
                      ICC = NA_real_, singular = NA,
                      status = paste0("fit_error: ", mod$message)))
  }

  vc          <- as.data.frame(lme4::VarCorr(mod))
  var_cluster <- vc$vcov[vc$grp == cluster]
  var_resid   <- vc$vcov[vc$grp == "Residual"]
  if (length(var_cluster) == 0) var_cluster <- 0
  if (length(var_resid)   == 0) var_resid   <- NA_real_
  icc <- var_cluster / (var_cluster + var_resid)

  data.frame(outcome = outcome, N = nrow(dat),
             n_clusters = nlevels(dat[[cluster]]),
             var_cluster = var_cluster, var_resid = var_resid,
             ICC = icc, singular = lme4::isSingular(mod, tol = 1e-5),
             status = "ok")
}

icc_results <- dplyr::bind_rows(lapply(analytic_outcomes, function(v) fit_null_icc(data, v)))
icc_results$Pct_var <- round(100 * icc_results$ICC, 2)
cat("\n=== Family ICC Results ===\n"); print(icc_results)
write.csv(icc_results, "output/tables/icc_results.csv", row.names = FALSE)

# ICC rule applied throughout this script:
#   facilitation_rt, disengagement_rt, Gap_amp, Gap_rt  → lm() (no fam_id RE)
#   Overlap_amp, Overlap_rt                             → lmer() + (1|fam_id)


# =============================================================================
# 7. ASSUMPTION CHECKS
# =============================================================================

sw_tests <- lapply(analytic_outcomes, function(v) {
  x <- data[[v]][!is.na(data[[v]])]
  shapiro.test(x)
})
names(sw_tests) <- analytic_outcomes

assumption_df <- data.frame(
  Outcome  = analytic_outcomes,
  N        = sapply(analytic_outcomes, function(v) sum(!is.na(data[[v]]))),
  Skewness = round(sapply(analytic_outcomes, function(v) moments::skewness(data[[v]], na.rm = TRUE)), 3),
  Kurtosis = round(sapply(analytic_outcomes, function(v) moments::kurtosis(data[[v]], na.rm = TRUE)), 3),
  SW_W     = round(sapply(analytic_outcomes, function(v) sw_tests[[v]]$statistic), 4),
  SW_p     = round(sapply(analytic_outcomes, function(v) sw_tests[[v]]$p.value),   4)
)
cat("\n=== Assumption Checks ===\n"); print(assumption_df)
write.csv(assumption_df, "output/tables/assumption_checks.csv", row.names = FALSE)


# =============================================================================
# 8. DISTRIBUTION PLOTS
# =============================================================================

# Shared colour palettes and theme
group_colors   <- c("Non-NDD" = "#2ecc71", "NDD" = "#e74c3c")
group_colors_3 <- c("Non-Affected"       = "#2ecc71",
                    "Affected (non-ASD)" = "#3498db",
                    "ASD"                = "#e74c3c")
theme_pub <- theme_minimal(base_size = 12) +
  theme(strip.background = element_rect(fill = "grey92", color = NA),
        strip.text       = element_text(face = "bold"),
        legend.position  = "bottom",
        panel.grid.minor = element_blank())

save_dist_plot <- function(v, df) {
  d_v <- df[!is.na(df[[v]]), ]
  p1 <- ggplot(d_v, aes(x = .data[[v]])) +
    geom_histogram(aes(y = after_stat(density)), bins = 35,
                   fill = "steelblue", alpha = 0.7, color = "white") +
    geom_density(color = "navy", linewidth = 1) +
    labs(title = paste(v, "— Overall"), x = v, y = "Density") + theme_pub
  p2 <- ggplot(d_v, aes(sample = .data[[v]])) +
    stat_qq() + stat_qq_line(color = "red") +
    labs(title = "Q-Q Plot") + theme_pub
  p3 <- ggplot(d_v, aes(x = .data[[v]], fill = ndd)) +
    geom_density(alpha = 0.5) + scale_fill_manual(values = group_colors) +
    labs(title = "Density by NDD") + theme_pub
  p4 <- ggplot(d_v, aes(x = ndd, y = .data[[v]], fill = ndd)) +
    geom_violin(alpha = 0.4, draw_quantiles = c(0.25, 0.5, 0.75)) +
    geom_jitter(width = 0.15, alpha = 0.3, size = 0.8) +
    stat_summary(fun = mean, geom = "point", shape = 18, size = 3, color = "black") +
    scale_fill_manual(values = group_colors) +
    labs(title = "Violin by NDD", x = "", y = v) + theme_pub +
    theme(legend.position = "none")
  combined <- (p1 + p2) / (p3 + p4) +
    plot_annotation(title = paste("Distribution Diagnostics:", v),
                    theme = theme(plot.title = element_text(face = "bold", size = 14)))
  ggsave(paste0("output/diagnostics/dist_", v, ".png"),
         combined, width = 12, height = 9, dpi = 300, bg = "white")
  invisible(combined)
}

save_dist_plot("facilitation_rt",  data)
save_dist_plot("disengagement_rt", data)
save_dist_plot("Gap_amp",          data)
save_dist_plot("Overlap_amp",      data)
save_dist_plot("Gap_rt",           data)
save_dist_plot("Overlap_rt",       data)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# ── One-sample t-test wrapper ─────────────────────────────────────────────────
one_sample_t <- function(x, label) {
  x  <- x[!is.na(x)]
  tt <- t.test(x, mu = 0)
  data.frame(
    Label    = label,
    N        = length(x),
    Mean     = round(mean(x), 3),
    SD       = round(sd(x),   3),
    SE       = round(sd(x) / sqrt(length(x)), 3),
    t        = round(tt$statistic,  3),
    df       = round(tt$parameter,  1),
    p        = round(tt$p.value,    4),
    CI_lower = round(tt$conf.int[1], 3),
    CI_upper = round(tt$conf.int[2], 3),
    Cohen_d  = round(mean(x) / sd(x), 3)
  )
}

# ── GAM marginal prediction (excludes s(fam_id) smooth) ──────────────────────
# For GAMs without s(fam_id) the exclude= is silently ignored (newdata.guaranteed).
predict_marginal <- function(model, newdata, site_levels, fam_levels) {
  newdata$site   <- factor(site_levels[1],  levels = site_levels)
  newdata$fam_id <- factor(fam_levels[1],   levels = fam_levels)
  predict(model, newdata = newdata, type = "response", se.fit = TRUE,
          exclude = "s(fam_id)", newdata.guaranteed = TRUE)
}

# ── Linear model age trajectory plot ─────────────────────────────────────────
make_traj_plot_lm <- function(model, data_v, outcome_name) {
  age_seq   <- seq(min(data_v$age, na.rm = TRUE),
                   max(data_v$age, na.rm = TRUE), length.out = 200)
  ref_site  <- levels(data_v$site)[1]
  ref_sex   <- levels(data_v$sex)[1]
  pred_grid <- data.frame(age    = age_seq,
                           site   = factor(ref_site, levels = levels(data_v$site)),
                           sex    = factor(ref_sex,  levels = levels(data_v$sex)),
                           fam_id = NA)
  if (inherits(model, "lmerMod")) {
    mm  <- model.matrix(~ age + site + sex, data = pred_grid)
    fe  <- lme4::fixef(model)
    vcv <- as.matrix(vcov(model))
    fit <- as.numeric(mm %*% fe)
    se  <- sqrt(rowSums((mm %*% vcv) * mm))
  } else {
    pr  <- predict(model, newdata = pred_grid, se.fit = TRUE)
    fit <- pr$fit
    se  <- pr$se.fit
  }
  pred_grid$fit <- fit
  pred_grid$lwr <- fit - 1.96 * se
  pred_grid$upr <- fit + 1.96 * se

  ggplot() +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
    geom_jitter(data = data_v, aes(x = age, y = .data[[outcome_name]]),
                alpha = 0.25, size = 1.2, width = 0.2, color = "grey40") +
    geom_ribbon(data = pred_grid, aes(x = age, ymin = lwr, ymax = upr),
                fill = "steelblue", alpha = 0.25) +
    geom_line(data = pred_grid, aes(x = age, y = fit),
              color = "steelblue", linewidth = 1.2) +
    labs(title    = paste("Age Trajectory —", outcome_name),
         subtitle = paste0("Linear model ± 95% CI  [ref: ", ref_sex, ", ", ref_site, "]"),
         x = "Age (years)", y = outcome_name) +
    theme_pub
}

# ── GAM age trajectory plot ───────────────────────────────────────────────────
make_traj_plot_gam <- function(model, data_v, outcome_name) {
  pred_grid <- data.frame(
    age = seq(min(data_v$age, na.rm = TRUE),
              max(data_v$age, na.rm = TRUE), length.out = 200)
  )
  preds <- predict_marginal(model, pred_grid,
                             levels(data_v$site), levels(data_v$fam_id))
  pred_grid$fit <- preds$fit
  pred_grid$lwr <- preds$fit - 1.96 * preds$se.fit
  pred_grid$upr <- preds$fit + 1.96 * preds$se.fit

  ggplot() +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
    geom_jitter(data = data_v, aes(x = age, y = .data[[outcome_name]]),
                alpha = 0.25, size = 1.2, width = 0.2, color = "grey40") +
    geom_ribbon(data = pred_grid, aes(x = age, ymin = lwr, ymax = upr),
                fill = "steelblue", alpha = 0.25) +
    geom_line(data = pred_grid, aes(x = age, y = fit),
              color = "steelblue", linewidth = 1.2) +
    labs(title    = paste("Age Trajectory —", outcome_name),
         subtitle = "GAM smooth ± 95% CI  [marginal; family RE excluded]",
         x = "Age (years)", y = outcome_name) +
    theme_pub
}

# ── LOESS raw scatter trajectory ──────────────────────────────────────────────
make_age_trajectories_raw <- function(data_v, outcome_name) {
  ggplot(data_v, aes(x = age, y = .data[[outcome_name]])) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
    geom_point(alpha = 0.35, size = 1.5, color = "grey40") +
    geom_smooth(method = "loess", se = TRUE,
                fill = "tomato", color = "tomato3", linewidth = 1.1) +
    labs(title    = paste("Raw Scatter —", outcome_name),
         subtitle = "LOESS smoother (no covariates)",
         x = "Age (years)", y = outcome_name) +
    theme_pub
}

# ── Significant pairs helper (for age-bin plots) ─────────────────────────────
pairs_to_signif <- function(pairs_obj, bin_levels, p_thresh = 0.05) {
  df        <- as.data.frame(pairs_obj)
  df$group1 <- trimws(sub("(?s) - .*",  "", df$contrast, perl = TRUE))
  df$group2 <- trimws(sub("(?s).* - ", "", df$contrast, perl = TRUE))
  df        <- df[df$p.value < p_thresh, ]
  if (nrow(df) == 0) return(NULL)

  clean_lbl <- function(x) {
    x <- gsub("\\s+", " ", trimws(x))
    x <- gsub("^\\((.*)\\)$", "\\1", x)
    x
  }
  clean_levels <- clean_lbl(bin_levels)
  df$pos1 <- match(clean_lbl(df$group1), clean_levels)
  df$pos2 <- match(clean_lbl(df$group2), clean_levels)
  df      <- df[!is.na(df$pos1) & !is.na(df$pos2), ]
  if (nrow(df) == 0) return(NULL)

  df$sig_label <- ifelse(df$p.value < .001, "***",
                         ifelse(df$p.value < .01,  "**",
                                ifelse(df$p.value < .05,  "*", "")))
  df
}

# ── Age-bin mean ± CI plot with significance brackets ────────────────────────
make_agebin_plot <- function(data_v, outcome_name, pairs_obj = NULL) {
  bin_levels <- levels(data_v$age_bin)
  summ_v <- data_v %>%
    group_by(age_bin) %>%
    summarise(M  = mean(.data[[outcome_name]], na.rm = TRUE),
              SE = sd(.data[[outcome_name]],   na.rm = TRUE) / sqrt(n()),
              N  = n(), .groups = "drop")

  y_min   <- min(summ_v$M - 1.96 * summ_v$SE, na.rm = TRUE)
  y_max   <- max(summ_v$M + 1.96 * summ_v$SE, na.rm = TRUE)
  y_range <- y_max - y_min
  step    <- y_range * 0.14

  p <- ggplot(summ_v, aes(x = age_bin, y = M, group = 1)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey60") +
    geom_ribbon(aes(ymin = M - 1.96 * SE, ymax = M + 1.96 * SE),
                fill = "steelblue", alpha = 0.15) +
    geom_line(color = "steelblue", linewidth = 1.2) +
    geom_point(color = "steelblue", size = 3.5) +
    geom_errorbar(aes(ymin = M - 1.96 * SE, ymax = M + 1.96 * SE),
                  width = 0.12, color = "steelblue4", linewidth = 0.8) +
    geom_text(aes(label = paste0("n=", N), y = M + 1.96 * SE + y_range * 0.05),
              size = 3, color = "grey40") +
    labs(title    = tools::toTitleCase(gsub("_", " ", outcome_name)),
         subtitle = "Mean \u00b1 95% CI  |  Tukey: * p<.05  ** p<.01  *** p<.001",
         x = "Age Group", y = tools::toTitleCase(gsub("_", " ", outcome_name))) +
    theme_bw(base_size = 12) +
    theme(panel.grid   = element_blank(),
          axis.text.x  = element_text(angle = 30, hjust = 1, size = 10),
          plot.title   = element_text(face = "bold", size = 13))

  if (!is.null(pairs_obj)) {
    sig_df <- pairs_to_signif(pairs_obj, bin_levels, p_thresh = 0.05)
    if (!is.null(sig_df) && nrow(sig_df) > 0) {
      sig_df <- sig_df[order(abs(sig_df$pos2 - sig_df$pos1)), ]
      sig_df$bracket_y <- y_max + step * seq_len(nrow(sig_df))
      p <- p +
        ggsignif::geom_signif(
          xmin        = sig_df$pos1,
          xmax        = sig_df$pos2,
          annotations = sig_df$sig_label,
          y_position  = sig_df$bracket_y,
          tip_length  = 0.02, textsize = 4.5, vjust = 0.3, color = "grey20"
        ) +
        scale_y_continuous(
          expand = expansion(mult = c(0.05, 0.10 + 0.15 * nrow(sig_df))))
    }
  }
  p
}

# ── NDD × age trajectory (lm or lmerMod) ─────────────────────────────────────
# Designed for M3 interaction models; auto-extracts fixed-effects formula for lmerMod.
make_ndd_traj_plot_lm <- function(model, data_v, outcome_name) {
  age_seq  <- seq(min(data_v$age, na.rm = TRUE),
                  max(data_v$age, na.rm = TRUE), length.out = 200)
  ref_site <- levels(data_v$site)[1]
  ref_sex  <- levels(data_v$sex)[1]

  build_pred_ndd <- function(ndd_level) {
    pg <- data.frame(
      age    = age_seq,
      ndd    = factor(ndd_level,  levels = levels(data_v$ndd)),
      sex    = factor(ref_sex,    levels = levels(data_v$sex)),
      site   = factor(ref_site,   levels = levels(data_v$site)),
      fam_id = levels(data_v$fam_id)[1]
    )
    if (inherits(model, "lmerMod")) {
      # Extract fixed-effects formula from the model to support both M2 and M3
      fe_form <- lme4::nobars(formula(model))[-2]
      mm      <- model.matrix(fe_form, data = pg)
      fe      <- lme4::fixef(model)
      vcv     <- as.matrix(vcov(model))
      fit     <- as.numeric(mm %*% fe)
      se      <- sqrt(rowSums((mm %*% vcv) * mm))
    } else {
      pr  <- predict(model, newdata = pg, se.fit = TRUE)
      fit <- pr$fit
      se  <- pr$se.fit
    }
    pg$fit <- fit; pg$lwr <- fit - 1.96 * se; pg$upr <- fit + 1.96 * se
    pg
  }

  pred_df <- rbind(build_pred_ndd("Non-NDD"), build_pred_ndd("NDD"))

  ggplot() +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
    geom_jitter(data = data_v,
                aes(x = age, y = .data[[outcome_name]], color = ndd),
                alpha = 0.25, size = 1.2, width = 0.15) +
    geom_ribbon(data = pred_df,
                aes(x = age, ymin = lwr, ymax = upr, fill = ndd), alpha = 0.20) +
    geom_line(data = pred_df,
              aes(x = age, y = fit, color = ndd), linewidth = 1.3) +
    scale_color_manual(values = group_colors) +
    scale_fill_manual(values  = group_colors) +
    labs(title    = paste("NDD Moderation —", outcome_name),
         subtitle = paste0("Model ± 95% CI  [ref: ", ref_sex, ", ", ref_site, "]"),
         x = "Age (years)", y = outcome_name, color = "Group", fill = "Group") +
    theme_pub
}

# ── 3-level affected_group × age trajectory (lm or lmerMod) ─────────────────
make_group_traj_plot <- function(model, data_v, outcome_name) {
  age_seq      <- seq(min(data_v$age, na.rm = TRUE),
                      max(data_v$age, na.rm = TRUE), length.out = 200)
  ref_site     <- levels(data_v$site)[1]
  ref_sex      <- levels(data_v$sex)[1]
  group_levels <- levels(data_v$affected_group)

  build_pred_grp <- function(grp_level) {
    pg <- data.frame(
      age            = age_seq,
      affected_group = factor(grp_level, levels = group_levels),
      sex            = factor(ref_sex,   levels = levels(data_v$sex)),
      site           = factor(ref_site,  levels = levels(data_v$site)),
      fam_id         = levels(data_v$fam_id)[1]
    )
    if (inherits(model, "lmerMod")) {
      fe_form <- lme4::nobars(formula(model))[-2]
      mm      <- model.matrix(fe_form, data = pg)
      fe      <- lme4::fixef(model)
      vcv     <- as.matrix(vcov(model))
      fit     <- as.numeric(mm %*% fe)
      se      <- sqrt(rowSums((mm %*% vcv) * mm))
    } else {
      pr  <- predict(model, newdata = pg, se.fit = TRUE)
      fit <- pr$fit
      se  <- pr$se.fit
    }
    pg$fit <- fit; pg$lwr <- fit - 1.96 * se; pg$upr <- fit + 1.96 * se
    pg
  }

  pred_df <- do.call(rbind, lapply(group_levels, build_pred_grp))

  ggplot() +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
    geom_jitter(data = data_v,
                aes(x = age, y = .data[[outcome_name]], color = affected_group),
                alpha = 0.25, size = 1.2, width = 0.15) +
    geom_ribbon(data = pred_df,
                aes(x = age, ymin = lwr, ymax = upr, fill = affected_group), alpha = 0.20) +
    geom_line(data = pred_df,
              aes(x = age, y = fit, color = affected_group), linewidth = 1.3) +
    scale_color_manual(values = group_colors_3) +
    scale_fill_manual(values  = group_colors_3) +
    labs(title    = paste("Affected Group Moderation —", outcome_name),
         subtitle = paste0("Model ± 95% CI  [ref: ", ref_sex, ", ", ref_site, "]"),
         x = "Age (years)", y = outcome_name, color = "Group", fill = "Group") +
    theme_pub
}


# =============================================================================
# Q1. FACILITATION & DISENGAGEMENT IN REACTION TIME
# One-sample t-tests (μ = 0) for facilitation_rt and disengagement_rt ONLY.
# No ERP measures in Q1.
# BH-FDR family: 10 tests (2 outcomes × 5 conditions: full + 4 age bins)
# =============================================================================

# Full sample
q1_os_full_facil_rt  <- one_sample_t(data$facilitation_rt,  "Full — facilitation_rt")
q1_os_full_diseng_rt <- one_sample_t(data$disengagement_rt, "Full — disengagement_rt")

# Children (<=11)
q1_os_ch_facil_rt  <- one_sample_t(
  data$facilitation_rt [data$age_bin == "Children\n(\u226411)"],
  "Children — facilitation_rt")
q1_os_ch_diseng_rt <- one_sample_t(
  data$disengagement_rt[data$age_bin == "Children\n(\u226411)"],
  "Children — disengagement_rt")

# Adolescents (12-17)
q1_os_ad_facil_rt  <- one_sample_t(
  data$facilitation_rt [data$age_bin == "Adolescents\n(12\u201317)"],
  "Adolescents — facilitation_rt")
q1_os_ad_diseng_rt <- one_sample_t(
  data$disengagement_rt[data$age_bin == "Adolescents\n(12\u201317)"],
  "Adolescents — disengagement_rt")

# Adults (18-44)
q1_os_au_facil_rt  <- one_sample_t(
  data$facilitation_rt [data$age_bin == "Adults\n(18\u201344)"],
  "Adults — facilitation_rt")
q1_os_au_diseng_rt <- one_sample_t(
  data$disengagement_rt[data$age_bin == "Adults\n(18\u201344)"],
  "Adults — disengagement_rt")

# Older Adults (45+)
q1_os_oa_facil_rt  <- one_sample_t(
  data$facilitation_rt [data$age_bin == "Older Adults\n(45+)"],
  "Older Adults — facilitation_rt")
q1_os_oa_diseng_rt <- one_sample_t(
  data$disengagement_rt[data$age_bin == "Older Adults\n(45+)"],
  "Older Adults — disengagement_rt")

# Combine all 10 Q1 tests
q1_all <- rbind(
  q1_os_full_facil_rt,  q1_os_full_diseng_rt,
  q1_os_ch_facil_rt,    q1_os_ch_diseng_rt,
  q1_os_ad_facil_rt,    q1_os_ad_diseng_rt,
  q1_os_au_facil_rt,    q1_os_au_diseng_rt,
  q1_os_oa_facil_rt,    q1_os_oa_diseng_rt
)

# BH-FDR across all 10 tests (single family)
q1_all$p_FDR   <- round(p.adjust(q1_all$p, method = "BH"), 4)
q1_all$sig_FDR <- ifelse(q1_all$p_FDR < .001, "***",
                          ifelse(q1_all$p_FDR < .01,  "**",
                                 ifelse(q1_all$p_FDR < .05, "*", "ns")))

cat("\n\n=== Q1: One-Sample t-tests — Facilitation & Disengagement in RT (BH-FDR, 10 tests) ===\n")
print(q1_all)
write.csv(q1_all, "output/tables/Q1_one_sample_BH_FDR.csv", row.names = FALSE)


# =============================================================================
# Q2. NEURAL CORRELATES OF FACILITATION & DISENGAGEMENT
# One-sample t-tests (μ = 0) for Gap_amp and Overlap_amp.
# Tests whether there is a reliable ERP response in each condition.
# BH-FDR family: 10 tests (2 outcomes × 5 conditions: full + 4 age bins)
# =============================================================================

# Full sample
q2_os_full_gap_amp  <- one_sample_t(data$Gap_amp,     "Full — Gap_amp")
q2_os_full_ovlp_amp <- one_sample_t(data$Overlap_amp, "Full — Overlap_amp")

# Children (<=11)
q2_os_ch_gap_amp  <- one_sample_t(
  data$Gap_amp    [data$age_bin == "Children\n(\u226411)"],
  "Children — Gap_amp")
q2_os_ch_ovlp_amp <- one_sample_t(
  data$Overlap_amp[data$age_bin == "Children\n(\u226411)"],
  "Children — Overlap_amp")

# Adolescents (12-17)
q2_os_ad_gap_amp  <- one_sample_t(
  data$Gap_amp    [data$age_bin == "Adolescents\n(12\u201317)"],
  "Adolescents — Gap_amp")
q2_os_ad_ovlp_amp <- one_sample_t(
  data$Overlap_amp[data$age_bin == "Adolescents\n(12\u201317)"],
  "Adolescents — Overlap_amp")

# Adults (18-44)
q2_os_au_gap_amp  <- one_sample_t(
  data$Gap_amp    [data$age_bin == "Adults\n(18\u201344)"],
  "Adults — Gap_amp")
q2_os_au_ovlp_amp <- one_sample_t(
  data$Overlap_amp[data$age_bin == "Adults\n(18\u201344)"],
  "Adults — Overlap_amp")

# Older Adults (45+)
q2_os_oa_gap_amp  <- one_sample_t(
  data$Gap_amp    [data$age_bin == "Older Adults\n(45+)"],
  "Older Adults — Gap_amp")
q2_os_oa_ovlp_amp <- one_sample_t(
  data$Overlap_amp[data$age_bin == "Older Adults\n(45+)"],
  "Older Adults — Overlap_amp")

# Combine all 10 Q2 tests
q2_all <- rbind(
  q2_os_full_gap_amp,  q2_os_full_ovlp_amp,
  q2_os_ch_gap_amp,    q2_os_ch_ovlp_amp,
  q2_os_ad_gap_amp,    q2_os_ad_ovlp_amp,
  q2_os_au_gap_amp,    q2_os_au_ovlp_amp,
  q2_os_oa_gap_amp,    q2_os_oa_ovlp_amp
)

# BH-FDR across all 10 tests (single family)
q2_all$p_FDR   <- round(p.adjust(q2_all$p, method = "BH"), 4)
q2_all$sig_FDR <- ifelse(q2_all$p_FDR < .001, "***",
                          ifelse(q2_all$p_FDR < .01,  "**",
                                 ifelse(q2_all$p_FDR < .05, "*", "ns")))

cat("\n\n=== Q2: One-Sample t-tests — Gap_amp & Overlap_amp (BH-FDR, 10 tests) ===\n")
print(q2_all)
write.csv(q2_all, "output/tables/Q2_one_sample_BH_FDR.csv", row.names = FALSE)


# =============================================================================
# Q3. AGE TRAJECTORIES
# Q3a: Linear models for facilitation_rt and disengagement_rt (ICC ~ 0%)
# Q3b: GAMs (k=3/5/8 AIC selection) for Gap_amp and Overlap_amp
#        Gap_amp:     no family RE (ICC ~ 0%)
#        Overlap_amp: s(fam_id, bs="re") added (ICC > 3%)
# Primary BH-FDR family: 4 tests (one age-effect p-value per outcome)
# Q3c (supplementary): Welch ANOVA + Tukey age-bin comparisons (not in main FDR)
# =============================================================================

# Analytic data subsets (complete cases for outcome + ndd covariate)
data_facil_rt  <- data[!is.na(data$facilitation_rt)  & !is.na(data$ndd), ]
data_diseng_rt <- data[!is.na(data$disengagement_rt) & !is.na(data$ndd), ]
data_gap_amp   <- data[!is.na(data$Gap_amp)           & !is.na(data$ndd), ]
data_ovlp_amp  <- data[!is.na(data$Overlap_amp)       & !is.na(data$ndd), ]

cat("\n=== Q3 analytic sample sizes ===\n")
cat("facilitation_rt:  N =", nrow(data_facil_rt),  "\n")
cat("disengagement_rt: N =", nrow(data_diseng_rt), "\n")
cat("Gap_amp:          N =", nrow(data_gap_amp),   "\n")
cat("Overlap_amp:      N =", nrow(data_ovlp_amp),  "\n")


# ── Q3a: facilitation_rt ~ age + site + sex ───────────────────────────────────

cat("\n\n--- Q3a: facilitation_rt ~ age + site + sex ---\n")
lm3a_facil_rt <- lm(facilitation_rt ~ age + site + sex, data = data_facil_rt)
print(summary(lm3a_facil_rt))

p3a_facil_rt_traj <- make_traj_plot_lm(lm3a_facil_rt, data_facil_rt, "facilitation_rt")
p3a_facil_rt_raw  <- make_age_trajectories_raw(data_facil_rt, "facilitation_rt")

ggsave("output/figures/Q3a_traj_facilitation_rt.png",     p3a_facil_rt_traj, width = 9, height = 6, dpi = 300, bg = "white")
ggsave("output/figures/Q3a_traj_facilitation_rt_raw.png", p3a_facil_rt_raw,  width = 9, height = 6, dpi = 300, bg = "white")


# ── Q3a: disengagement_rt ~ age + site + sex ──────────────────────────────────

cat("\n\n--- Q3a: disengagement_rt ~ age + site + sex ---\n")
lm3a_diseng_rt <- lm(disengagement_rt ~ age + site + sex, data = data_diseng_rt)
print(summary(lm3a_diseng_rt))

p3a_diseng_rt_traj <- make_traj_plot_lm(lm3a_diseng_rt, data_diseng_rt, "disengagement_rt")
p3a_diseng_rt_raw  <- make_age_trajectories_raw(data_diseng_rt, "disengagement_rt")

ggsave("output/figures/Q3a_traj_disengagement_rt.png",     p3a_diseng_rt_traj, width = 9, height = 6, dpi = 300, bg = "white")
ggsave("output/figures/Q3a_traj_disengagement_rt_raw.png", p3a_diseng_rt_raw,  width = 9, height = 6, dpi = 300, bg = "white")


# ── Q3b: GAM k-selection — Gap_amp (no family RE, ICC ~ 0%) ──────────────────

cat("\n\n--- Q3b GAM k-selection: Gap_amp ---\n")
gam3b_gap_amp_k3 <- gam(Gap_amp ~ s(age, k = 3) + site + sex, data = data_gap_amp, method = "REML")
gam3b_gap_amp_k5 <- gam(Gap_amp ~ s(age, k = 5) + site + sex, data = data_gap_amp, method = "REML")
gam3b_gap_amp_k8 <- gam(Gap_amp ~ s(age, k = 8) + site + sex, data = data_gap_amp, method = "REML")

aic_gap_amp <- data.frame(
  k       = c(3, 5, 8),
  AIC     = c(AIC(gam3b_gap_amp_k3), AIC(gam3b_gap_amp_k5), AIC(gam3b_gap_amp_k8)),
  edf     = c(sum(gam3b_gap_amp_k3$edf),
              sum(gam3b_gap_amp_k5$edf),
              sum(gam3b_gap_amp_k8$edf)),
  dev_expl = c(summary(gam3b_gap_amp_k3)$dev.expl,
               summary(gam3b_gap_amp_k5)$dev.expl,
               summary(gam3b_gap_amp_k8)$dev.expl)
)
aic_gap_amp$delta_AIC <- aic_gap_amp$AIC - min(aic_gap_amp$AIC)
cat("Gap_amp k-selection:\n"); print(aic_gap_amp)
write.csv(aic_gap_amp, "output/tables/Q3b_GAM_k_selection_Gap_amp.csv", row.names = FALSE)

best_k_gap_amp     <- c(3, 5, 8)[which.min(aic_gap_amp$AIC)]
gam3b_gap_amp_best <- switch(as.character(best_k_gap_amp),
                             "3" = gam3b_gap_amp_k3,
                             "5" = gam3b_gap_amp_k5,
                             "8" = gam3b_gap_amp_k8)
cat("\nBest k for Gap_amp:", best_k_gap_amp, "\n")
cat("Gap_amp best-model summary:\n"); print(summary(gam3b_gap_amp_best))

p3b_gap_amp_traj <- make_traj_plot_gam(gam3b_gap_amp_best, data_gap_amp, "Gap_amp")
p3b_gap_amp_raw  <- make_age_trajectories_raw(data_gap_amp, "Gap_amp")

ggsave("output/figures/Q3b_traj_Gap_amp.png",     p3b_gap_amp_traj, width = 9, height = 6, dpi = 300, bg = "white")
ggsave("output/figures/Q3b_traj_Gap_amp_raw.png", p3b_gap_amp_raw,  width = 9, height = 6, dpi = 300, bg = "white")


# ── Q3b: GAM k-selection — Overlap_amp (with s(fam_id, bs="re"), ICC > 3%) ───

cat("\n\n--- Q3b GAM k-selection: Overlap_amp ---\n")
gam3b_ovlp_amp_k3 <- gam(Overlap_amp ~ s(age, k = 3) + site + sex + s(fam_id, bs = "re"),
                          data = data_ovlp_amp, method = "REML")
gam3b_ovlp_amp_k5 <- gam(Overlap_amp ~ s(age, k = 5) + site + sex + s(fam_id, bs = "re"),
                          data = data_ovlp_amp, method = "REML")
gam3b_ovlp_amp_k8 <- gam(Overlap_amp ~ s(age, k = 8) + site + sex + s(fam_id, bs = "re"),
                          data = data_ovlp_amp, method = "REML")

aic_ovlp_amp <- data.frame(
  k       = c(3, 5, 8),
  AIC     = c(AIC(gam3b_ovlp_amp_k3), AIC(gam3b_ovlp_amp_k5), AIC(gam3b_ovlp_amp_k8)),
  edf     = c(sum(gam3b_ovlp_amp_k3$edf),
              sum(gam3b_ovlp_amp_k5$edf),
              sum(gam3b_ovlp_amp_k8$edf)),
  dev_expl = c(summary(gam3b_ovlp_amp_k3)$dev.expl,
               summary(gam3b_ovlp_amp_k5)$dev.expl,
               summary(gam3b_ovlp_amp_k8)$dev.expl)
)
aic_ovlp_amp$delta_AIC <- aic_ovlp_amp$AIC - min(aic_ovlp_amp$AIC)
cat("Overlap_amp k-selection:\n"); print(aic_ovlp_amp)
write.csv(aic_ovlp_amp, "output/tables/Q3b_GAM_k_selection_Overlap_amp.csv", row.names = FALSE)

best_k_ovlp_amp     <- c(3, 5, 8)[which.min(aic_ovlp_amp$AIC)]
gam3b_ovlp_amp_best <- switch(as.character(best_k_ovlp_amp),
                              "3" = gam3b_ovlp_amp_k3,
                              "5" = gam3b_ovlp_amp_k5,
                              "8" = gam3b_ovlp_amp_k8)
cat("\nBest k for Overlap_amp:", best_k_ovlp_amp, "\n")
cat("Overlap_amp best-model summary:\n"); print(summary(gam3b_ovlp_amp_best))

p3b_ovlp_amp_traj <- make_traj_plot_gam(gam3b_ovlp_amp_best, data_ovlp_amp, "Overlap_amp")
p3b_ovlp_amp_raw  <- make_age_trajectories_raw(data_ovlp_amp, "Overlap_amp")

ggsave("output/figures/Q3b_traj_Overlap_amp.png",     p3b_ovlp_amp_traj, width = 9, height = 6, dpi = 300, bg = "white")
ggsave("output/figures/Q3b_traj_Overlap_amp_raw.png", p3b_ovlp_amp_raw,  width = 9, height = 6, dpi = 300, bg = "white")


# ── Q3 Primary BH-FDR: 4 tests (one age-effect p per outcome) ────────────────

q3_p_facil_rt  <- summary(lm3a_facil_rt)$coefficients["age", "Pr(>|t|)"]
q3_p_diseng_rt <- summary(lm3a_diseng_rt)$coefficients["age", "Pr(>|t|)"]
q3_p_gap_amp   <- summary(gam3b_gap_amp_best)$s.table["s(age)", "p-value"]
q3_p_ovlp_amp  <- summary(gam3b_ovlp_amp_best)$s.table["s(age)", "p-value"]

q3_age_fdr <- data.frame(
  Outcome   = c("facilitation_rt", "disengagement_rt", "Gap_amp", "Overlap_amp"),
  Model     = c("lm", "lm", "GAM", "GAM"),
  p_age     = round(c(q3_p_facil_rt, q3_p_diseng_rt, q3_p_gap_amp, q3_p_ovlp_amp), 4),
  p_age_FDR = round(p.adjust(
    c(q3_p_facil_rt, q3_p_diseng_rt, q3_p_gap_amp, q3_p_ovlp_amp),
    method = "BH"), 4)
)
q3_age_fdr$sig_FDR <- ifelse(q3_age_fdr$p_age_FDR < .001, "***",
                              ifelse(q3_age_fdr$p_age_FDR < .01,  "**",
                                     ifelse(q3_age_fdr$p_age_FDR < .05, "*", "ns")))

cat("\n=== Q3 Age Effects — Primary BH-FDR (4 tests) ===\n")
print(q3_age_fdr)
write.csv(q3_age_fdr, "output/tables/Q3_age_BH_FDR.csv", row.names = FALSE)


# ── Q3c (Supplementary): Welch ANOVA + Tukey post-hoc by age bin ─────────────
# Not included in the primary Q3 BH-FDR family.

cat("\n\n=== Q3c (Supplementary): Welch ANOVAs by Age Bin ===\n")

# facilitation_rt
cat("\n--- facilitation_rt ---\n")
welch_q3c_facil_rt     <- oneway.test(facilitation_rt  ~ age_bin, data = data_facil_rt,  var.equal = FALSE)
eta2_q3c_facil_rt      <- effectsize::eta_squared(aov(facilitation_rt  ~ age_bin, data = data_facil_rt),  partial = FALSE)
emm_q3c_facil_rt_bin   <- emmeans(lm(facilitation_rt  ~ age_bin, data = data_facil_rt),  ~ age_bin)
pairs_q3c_facil_rt     <- pairs(emm_q3c_facil_rt_bin, adjust = "tukey")
print(welch_q3c_facil_rt)
cat("eta2 =", round(as.numeric(eta2_q3c_facil_rt$Eta2), 4), "\n")
print(pairs_q3c_facil_rt)
write.csv(as.data.frame(pairs_q3c_facil_rt),
          "output/supplementary/Q3c_tukey_facilitation_rt.csv", row.names = FALSE)

# disengagement_rt
cat("\n--- disengagement_rt ---\n")
welch_q3c_diseng_rt    <- oneway.test(disengagement_rt ~ age_bin, data = data_diseng_rt, var.equal = FALSE)
eta2_q3c_diseng_rt     <- effectsize::eta_squared(aov(disengagement_rt ~ age_bin, data = data_diseng_rt), partial = FALSE)
emm_q3c_diseng_rt_bin  <- emmeans(lm(disengagement_rt ~ age_bin, data = data_diseng_rt), ~ age_bin)
pairs_q3c_diseng_rt    <- pairs(emm_q3c_diseng_rt_bin, adjust = "tukey")
print(welch_q3c_diseng_rt)
cat("eta2 =", round(as.numeric(eta2_q3c_diseng_rt$Eta2), 4), "\n")
print(pairs_q3c_diseng_rt)
write.csv(as.data.frame(pairs_q3c_diseng_rt),
          "output/supplementary/Q3c_tukey_disengagement_rt.csv", row.names = FALSE)

# Gap_amp
cat("\n--- Gap_amp ---\n")
welch_q3c_gap_amp      <- oneway.test(Gap_amp   ~ age_bin, data = data_gap_amp,   var.equal = FALSE)
eta2_q3c_gap_amp       <- effectsize::eta_squared(aov(Gap_amp   ~ age_bin, data = data_gap_amp),   partial = FALSE)
emm_q3c_gap_amp_bin    <- emmeans(lm(Gap_amp   ~ age_bin, data = data_gap_amp),   ~ age_bin)
pairs_q3c_gap_amp      <- pairs(emm_q3c_gap_amp_bin, adjust = "tukey")
print(welch_q3c_gap_amp)
cat("eta2 =", round(as.numeric(eta2_q3c_gap_amp$Eta2), 4), "\n")
print(pairs_q3c_gap_amp)
write.csv(as.data.frame(pairs_q3c_gap_amp),
          "output/supplementary/Q3c_tukey_Gap_amp.csv", row.names = FALSE)

# Overlap_amp
cat("\n--- Overlap_amp ---\n")
welch_q3c_ovlp_amp     <- oneway.test(Overlap_amp ~ age_bin, data = data_ovlp_amp,  var.equal = FALSE)
eta2_q3c_ovlp_amp      <- effectsize::eta_squared(aov(Overlap_amp ~ age_bin, data = data_ovlp_amp),  partial = FALSE)
emm_q3c_ovlp_amp_bin   <- emmeans(lm(Overlap_amp ~ age_bin, data = data_ovlp_amp),  ~ age_bin)
pairs_q3c_ovlp_amp     <- pairs(emm_q3c_ovlp_amp_bin, adjust = "tukey")
print(welch_q3c_ovlp_amp)
cat("eta2 =", round(as.numeric(eta2_q3c_ovlp_amp$Eta2), 4), "\n")
print(pairs_q3c_ovlp_amp)
write.csv(as.data.frame(pairs_q3c_ovlp_amp),
          "output/supplementary/Q3c_tukey_Overlap_amp.csv", row.names = FALSE)

# Age-bin plots (supplementary)
p3c_facil_rt  <- make_agebin_plot(data_facil_rt,  "facilitation_rt",  pairs_q3c_facil_rt)
p3c_diseng_rt <- make_agebin_plot(data_diseng_rt, "disengagement_rt", pairs_q3c_diseng_rt)
p3c_gap_amp   <- make_agebin_plot(data_gap_amp,   "Gap_amp",          pairs_q3c_gap_amp)
p3c_ovlp_amp  <- make_agebin_plot(data_ovlp_amp,  "Overlap_amp",      pairs_q3c_ovlp_amp)

ggsave("output/supplementary/Q3c_agebin_facilitation_rt.png",  p3c_facil_rt,  width = 8, height = 6, dpi = 300, bg = "white")
ggsave("output/supplementary/Q3c_agebin_disengagement_rt.png", p3c_diseng_rt, width = 8, height = 6, dpi = 300, bg = "white")
ggsave("output/supplementary/Q3c_agebin_Gap_amp.png",          p3c_gap_amp,   width = 8, height = 6, dpi = 300, bg = "white")
ggsave("output/supplementary/Q3c_agebin_Overlap_amp.png",      p3c_ovlp_amp,  width = 8, height = 6, dpi = 300, bg = "white")


# =============================================================================
# Q4. DIAGNOSTIC DIFFERENCES
# Outcomes: facilitation_rt, disengagement_rt, Gap_amp, Overlap_amp
#   (facilitation_rt & disengagement_rt always included per user specification)
#   (Gap_amp & Overlap_amp included as primary ERP amplitude outcomes)
#   No latency measures.
#
# NDD moderation (binary):
#   M1: outcome ~ age + sex + site [+ (1|fam_id)]          — null
#   M2: outcome ~ ndd + age + sex + site [+ (1|fam_id)]    — NDD main effect
#   M3: outcome ~ ndd * age + sex + site [+ (1|fam_id)]    — NDD × age interaction
#
# Affected-group moderation (3-level):
#   M0: outcome ~ age + sex + site [+ (1|fam_id)]                    — null
#   M1: outcome ~ affected_group + age + sex + site [+ (1|fam_id)]   — group main
#   M2: outcome ~ affected_group * age + sex + site [+ (1|fam_id)]   — group × age
#
# RE rule:
#   facilitation_rt, disengagement_rt, Gap_amp → lm()
#   Overlap_amp                                → lmer() with (1|fam_id)
#     Use REML = FALSE for LRT (anova()); refit REML = TRUE for summaries/emmeans.
#
# BH-FDR families (4 separate families, each 4 tests):
#   Q4_NDD_main:   NDD main-effect LRT p, one per outcome
#   Q4_NDD_int:    NDD × age interaction LRT p, one per outcome
#   Q4_grp_main:   affected_group main-effect LRT p, one per outcome
#   Q4_grp_int:    affected_group × age interaction LRT p, one per outcome
# =============================================================================

# 3-level group subsets
data_3lvl_facil_rt  <- data[!is.na(data$facilitation_rt)  & !is.na(data$affected_group), ]
data_3lvl_diseng_rt <- data[!is.na(data$disengagement_rt) & !is.na(data$affected_group), ]
data_3lvl_gap_amp   <- data[!is.na(data$Gap_amp)           & !is.na(data$affected_group), ]
data_3lvl_ovlp_amp  <- data[!is.na(data$Overlap_amp)       & !is.na(data$affected_group), ]


# =============================================================================
# Q4a: NDD MODERATION
# =============================================================================

# ── NDD: facilitation_rt (lm, ICC ~ 0%) ──────────────────────────────────────

cat("\n\n========== Q4 NDD: facilitation_rt ==========\n")
cat("N =", nrow(data_facil_rt), "\n")

lm_q4_facil_rt_m1 <- lm(facilitation_rt ~ age + sex + site,           data = data_facil_rt)
lm_q4_facil_rt_m2 <- lm(facilitation_rt ~ ndd + age + sex + site,     data = data_facil_rt)
lm_q4_facil_rt_m3 <- lm(facilitation_rt ~ ndd * age + sex + site,     data = data_facil_rt)

cat("\nM1 vs M2 (NDD main effect):\n");       print(anova(lm_q4_facil_rt_m1, lm_q4_facil_rt_m2))
cat("\nM2 vs M3 (NDD x age interaction):\n"); print(anova(lm_q4_facil_rt_m2, lm_q4_facil_rt_m3))
cat("\nM2 summary:\n"); print(summary(lm_q4_facil_rt_m2))
cat("\nM3 summary:\n"); print(summary(lm_q4_facil_rt_m3))

emm_q4_facil_rt_ndd   <- emmeans(lm_q4_facil_rt_m2, ~ ndd,
                                   at = list(age = median(data_facil_rt$age, na.rm = TRUE)))
pairs_q4_facil_rt_ndd <- pairs(emm_q4_facil_rt_ndd, adjust = "tukey")
eff_q4_facil_rt_ndd   <- eff_size(emm_q4_facil_rt_ndd,
                                   sigma = sigma(lm_q4_facil_rt_m2),
                                   edf   = df.residual(lm_q4_facil_rt_m2))
cat("\nMarginal means (M2, at median age):\n"); print(emm_q4_facil_rt_ndd)
cat("\nPairwise contrasts:\n");                 print(pairs_q4_facil_rt_ndd)
cat("\nEffect sizes (Cohen's d):\n");           print(eff_q4_facil_rt_ndd)
write.csv(as.data.frame(summary(pairs_q4_facil_rt_ndd)),
          "output/tables/Q4_NDD_pairwise_facilitation_rt.csv", row.names = FALSE)


# ── NDD: disengagement_rt (lm, ICC ~ 0%) ─────────────────────────────────────

cat("\n\n========== Q4 NDD: disengagement_rt ==========\n")
cat("N =", nrow(data_diseng_rt), "\n")

lm_q4_diseng_rt_m1 <- lm(disengagement_rt ~ age + sex + site,         data = data_diseng_rt)
lm_q4_diseng_rt_m2 <- lm(disengagement_rt ~ ndd + age + sex + site,   data = data_diseng_rt)
lm_q4_diseng_rt_m3 <- lm(disengagement_rt ~ ndd * age + sex + site,   data = data_diseng_rt)

cat("\nM1 vs M2 (NDD main effect):\n");       print(anova(lm_q4_diseng_rt_m1, lm_q4_diseng_rt_m2))
cat("\nM2 vs M3 (NDD x age interaction):\n"); print(anova(lm_q4_diseng_rt_m2, lm_q4_diseng_rt_m3))
cat("\nM2 summary:\n"); print(summary(lm_q4_diseng_rt_m2))
cat("\nM3 summary:\n"); print(summary(lm_q4_diseng_rt_m3))

emm_q4_diseng_rt_ndd   <- emmeans(lm_q4_diseng_rt_m2, ~ ndd,
                                    at = list(age = median(data_diseng_rt$age, na.rm = TRUE)))
pairs_q4_diseng_rt_ndd <- pairs(emm_q4_diseng_rt_ndd, adjust = "tukey")
eff_q4_diseng_rt_ndd   <- eff_size(emm_q4_diseng_rt_ndd,
                                    sigma = sigma(lm_q4_diseng_rt_m2),
                                    edf   = df.residual(lm_q4_diseng_rt_m2))
cat("\nMarginal means (M2, at median age):\n"); print(emm_q4_diseng_rt_ndd)
cat("\nPairwise contrasts:\n");                 print(pairs_q4_diseng_rt_ndd)
cat("\nEffect sizes (Cohen's d):\n");           print(eff_q4_diseng_rt_ndd)
write.csv(as.data.frame(summary(pairs_q4_diseng_rt_ndd)),
          "output/tables/Q4_NDD_pairwise_disengagement_rt.csv", row.names = FALSE)


# ── NDD: Gap_amp (lm, ICC ~ 0%) ──────────────────────────────────────────────

cat("\n\n========== Q4 NDD: Gap_amp ==========\n")
cat("N =", nrow(data_gap_amp), "\n")

lm_q4_gap_amp_m1 <- lm(Gap_amp ~ age + sex + site,       data = data_gap_amp)
lm_q4_gap_amp_m2 <- lm(Gap_amp ~ ndd + age + sex + site, data = data_gap_amp)
lm_q4_gap_amp_m3 <- lm(Gap_amp ~ ndd * age + sex + site, data = data_gap_amp)

cat("\nM1 vs M2 (NDD main effect):\n");       print(anova(lm_q4_gap_amp_m1, lm_q4_gap_amp_m2))
cat("\nM2 vs M3 (NDD x age interaction):\n"); print(anova(lm_q4_gap_amp_m2, lm_q4_gap_amp_m3))
cat("\nM2 summary:\n"); print(summary(lm_q4_gap_amp_m2))
cat("\nM3 summary:\n"); print(summary(lm_q4_gap_amp_m3))

emm_q4_gap_amp_ndd   <- emmeans(lm_q4_gap_amp_m2, ~ ndd,
                                  at = list(age = median(data_gap_amp$age, na.rm = TRUE)))
pairs_q4_gap_amp_ndd <- pairs(emm_q4_gap_amp_ndd, adjust = "tukey")
eff_q4_gap_amp_ndd   <- eff_size(emm_q4_gap_amp_ndd,
                                  sigma = sigma(lm_q4_gap_amp_m2),
                                  edf   = df.residual(lm_q4_gap_amp_m2))
cat("\nMarginal means (M2, at median age):\n"); print(emm_q4_gap_amp_ndd)
cat("\nPairwise contrasts:\n");                 print(pairs_q4_gap_amp_ndd)
cat("\nEffect sizes (Cohen's d):\n");           print(eff_q4_gap_amp_ndd)
write.csv(as.data.frame(summary(pairs_q4_gap_amp_ndd)),
          "output/tables/Q4_NDD_pairwise_Gap_amp.csv", row.names = FALSE)


# ── NDD: Overlap_amp (lmer, ICC > 3%) ────────────────────────────────────────

cat("\n\n========== Q4 NDD: Overlap_amp ==========\n")
cat("N =", nrow(data_ovlp_amp), "\n")

# REML = FALSE for likelihood-ratio tests
lmer_q4_ovlp_amp_m1 <- lmer(Overlap_amp ~ age + sex + site + (1 | fam_id),
                             data = data_ovlp_amp, REML = FALSE)
lmer_q4_ovlp_amp_m2 <- lmer(Overlap_amp ~ ndd + age + sex + site + (1 | fam_id),
                             data = data_ovlp_amp, REML = FALSE)
lmer_q4_ovlp_amp_m3 <- lmer(Overlap_amp ~ ndd * age + sex + site + (1 | fam_id),
                             data = data_ovlp_amp, REML = FALSE)

cat("\nM1 vs M2 (NDD main effect):\n");       print(anova(lmer_q4_ovlp_amp_m1, lmer_q4_ovlp_amp_m2))
cat("\nM2 vs M3 (NDD x age interaction):\n"); print(anova(lmer_q4_ovlp_amp_m2, lmer_q4_ovlp_amp_m3))

# Refit REML = TRUE for summaries, emmeans, and trajectory plots
lmer_q4_ovlp_amp_m2r <- lmer(Overlap_amp ~ ndd + age + sex + site + (1 | fam_id),
                              data = data_ovlp_amp, REML = TRUE)
lmer_q4_ovlp_amp_m3r <- lmer(Overlap_amp ~ ndd * age + sex + site + (1 | fam_id),
                              data = data_ovlp_amp, REML = TRUE)
cat("\nM2 summary (REML):\n"); print(summary(lmer_q4_ovlp_amp_m2r))
cat("\nM3 summary (REML):\n"); print(summary(lmer_q4_ovlp_amp_m3r))

emm_q4_ovlp_amp_ndd   <- emmeans(lmer_q4_ovlp_amp_m2r, ~ ndd,
                                   at = list(age = median(data_ovlp_amp$age, na.rm = TRUE)))
pairs_q4_ovlp_amp_ndd <- pairs(emm_q4_ovlp_amp_ndd, adjust = "tukey")
eff_q4_ovlp_amp_ndd   <- eff_size(emm_q4_ovlp_amp_ndd,
                                   sigma = sigma(lmer_q4_ovlp_amp_m2r),
                                   edf   = df.residual(lmer_q4_ovlp_amp_m2r))
cat("\nMarginal means (M2, at median age):\n"); print(emm_q4_ovlp_amp_ndd)
cat("\nPairwise contrasts:\n");                 print(pairs_q4_ovlp_amp_ndd)
cat("\nEffect sizes (Cohen's d):\n");           print(eff_q4_ovlp_amp_ndd)
write.csv(as.data.frame(summary(pairs_q4_ovlp_amp_ndd)),
          "output/tables/Q4_NDD_pairwise_Overlap_amp.csv", row.names = FALSE)


# ── Q4 NDD BH-FDR (2 families: main effect, interaction) ─────────────────────

q4_ndd_p_main <- c(
  anova(lm_q4_facil_rt_m1,   lm_q4_facil_rt_m2)$`Pr(>F)`[2],
  anova(lm_q4_diseng_rt_m1,  lm_q4_diseng_rt_m2)$`Pr(>F)`[2],
  anova(lm_q4_gap_amp_m1,    lm_q4_gap_amp_m2)$`Pr(>F)`[2],
  anova(lmer_q4_ovlp_amp_m1, lmer_q4_ovlp_amp_m2)$`Pr(>Chisq)`[2]
)
q4_ndd_p_int <- c(
  anova(lm_q4_facil_rt_m2,   lm_q4_facil_rt_m3)$`Pr(>F)`[2],
  anova(lm_q4_diseng_rt_m2,  lm_q4_diseng_rt_m3)$`Pr(>F)`[2],
  anova(lm_q4_gap_amp_m2,    lm_q4_gap_amp_m3)$`Pr(>F)`[2],
  anova(lmer_q4_ovlp_amp_m2, lmer_q4_ovlp_amp_m3)$`Pr(>Chisq)`[2]
)

q4_ndd_fdr <- data.frame(
  Outcome         = c("facilitation_rt", "disengagement_rt", "Gap_amp", "Overlap_amp"),
  p_NDD_main      = round(q4_ndd_p_main, 4),
  p_NDD_main_FDR  = round(p.adjust(q4_ndd_p_main, method = "BH"), 4),
  sig_main        = ifelse(p.adjust(q4_ndd_p_main, method = "BH") < .05, "*", "ns"),
  p_NDD_x_age     = round(q4_ndd_p_int, 4),
  p_NDD_x_age_FDR = round(p.adjust(q4_ndd_p_int, method = "BH"), 4),
  sig_int         = ifelse(p.adjust(q4_ndd_p_int, method = "BH") < .05, "*", "ns")
)
cat("\n=== Q4 NDD BH-FDR Summary ===\n"); print(q4_ndd_fdr)
write.csv(q4_ndd_fdr, "output/tables/Q4_NDD_BH_FDR_summary.csv", row.names = FALSE)

# ── Q4 NDD Trajectory Plots (M3: NDD × age) ───────────────────────────────────
p4_ndd_facil_rt  <- make_ndd_traj_plot_lm(lm_q4_facil_rt_m3,    data_facil_rt,  "facilitation_rt")
p4_ndd_diseng_rt <- make_ndd_traj_plot_lm(lm_q4_diseng_rt_m3,   data_diseng_rt, "disengagement_rt")
p4_ndd_gap_amp   <- make_ndd_traj_plot_lm(lm_q4_gap_amp_m3,     data_gap_amp,   "Gap_amp")
p4_ndd_ovlp_amp  <- make_ndd_traj_plot_lm(lmer_q4_ovlp_amp_m3r, data_ovlp_amp,  "Overlap_amp")

ggsave("output/figures/Q4_NDD_traj_facilitation_rt.png",  p4_ndd_facil_rt,  width = 10, height = 6, dpi = 300, bg = "white")
ggsave("output/figures/Q4_NDD_traj_disengagement_rt.png", p4_ndd_diseng_rt, width = 10, height = 6, dpi = 300, bg = "white")
ggsave("output/figures/Q4_NDD_traj_Gap_amp.png",          p4_ndd_gap_amp,   width = 10, height = 6, dpi = 300, bg = "white")
ggsave("output/figures/Q4_NDD_traj_Overlap_amp.png",      p4_ndd_ovlp_amp,  width = 10, height = 6, dpi = 300, bg = "white")


# =============================================================================
# Q4b: AFFECTED_GROUP MODERATION (3-level)
# =============================================================================

# ── Affected group: facilitation_rt (lm, ICC ~ 0%) ───────────────────────────

cat("\n\n========== Q4 Affected Group: facilitation_rt ==========\n")
cat("N =", nrow(data_3lvl_facil_rt), "\n")

lm_q4g_facil_rt_m0 <- lm(facilitation_rt ~ age + sex + site, data = data_3lvl_facil_rt)
lm_q4g_facil_rt_m1 <- lm(facilitation_rt ~ affected_group + age + sex + site,
                           data = data_3lvl_facil_rt)
lm_q4g_facil_rt_m2 <- lm(facilitation_rt ~ affected_group * age + sex + site,
                           data = data_3lvl_facil_rt)

cat("\nM0 vs M1 (group main effect):\n");       print(anova(lm_q4g_facil_rt_m0, lm_q4g_facil_rt_m1))
cat("\nM1 vs M2 (group x age interaction):\n"); print(anova(lm_q4g_facil_rt_m1, lm_q4g_facil_rt_m2))
cat("\nM1 summary:\n"); print(summary(lm_q4g_facil_rt_m1))
cat("\nM2 summary:\n"); print(summary(lm_q4g_facil_rt_m2))

emm_q4g_facil_rt   <- emmeans(lm_q4g_facil_rt_m1, ~ affected_group,
                                at = list(age = median(data_3lvl_facil_rt$age, na.rm = TRUE)))
pairs_q4g_facil_rt <- pairs(emm_q4g_facil_rt, adjust = "tukey")
eff_q4g_facil_rt   <- eff_size(emm_q4g_facil_rt,
                                sigma = sigma(lm_q4g_facil_rt_m1),
                                edf   = df.residual(lm_q4g_facil_rt_m1))
cat("\nMarginal means (M1, at median age):\n"); print(emm_q4g_facil_rt)
cat("\nPairwise contrasts:\n");                 print(pairs_q4g_facil_rt)
cat("\nEffect sizes:\n");                       print(eff_q4g_facil_rt)
write.csv(as.data.frame(summary(pairs_q4g_facil_rt)),
          "output/tables/Q4_group_pairwise_facilitation_rt.csv", row.names = FALSE)


# ── Affected group: disengagement_rt (lm, ICC ~ 0%) ──────────────────────────

cat("\n\n========== Q4 Affected Group: disengagement_rt ==========\n")
cat("N =", nrow(data_3lvl_diseng_rt), "\n")

lm_q4g_diseng_rt_m0 <- lm(disengagement_rt ~ age + sex + site, data = data_3lvl_diseng_rt)
lm_q4g_diseng_rt_m1 <- lm(disengagement_rt ~ affected_group + age + sex + site,
                            data = data_3lvl_diseng_rt)
lm_q4g_diseng_rt_m2 <- lm(disengagement_rt ~ affected_group * age + sex + site,
                            data = data_3lvl_diseng_rt)

cat("\nM0 vs M1 (group main effect):\n");       print(anova(lm_q4g_diseng_rt_m0, lm_q4g_diseng_rt_m1))
cat("\nM1 vs M2 (group x age interaction):\n"); print(anova(lm_q4g_diseng_rt_m1, lm_q4g_diseng_rt_m2))
cat("\nM1 summary:\n"); print(summary(lm_q4g_diseng_rt_m1))
cat("\nM2 summary:\n"); print(summary(lm_q4g_diseng_rt_m2))

emm_q4g_diseng_rt   <- emmeans(lm_q4g_diseng_rt_m1, ~ affected_group,
                                 at = list(age = median(data_3lvl_diseng_rt$age, na.rm = TRUE)))
pairs_q4g_diseng_rt <- pairs(emm_q4g_diseng_rt, adjust = "tukey")
eff_q4g_diseng_rt   <- eff_size(emm_q4g_diseng_rt,
                                 sigma = sigma(lm_q4g_diseng_rt_m1),
                                 edf   = df.residual(lm_q4g_diseng_rt_m1))
cat("\nMarginal means (M1, at median age):\n"); print(emm_q4g_diseng_rt)
cat("\nPairwise contrasts:\n");                 print(pairs_q4g_diseng_rt)
cat("\nEffect sizes:\n");                       print(eff_q4g_diseng_rt)
write.csv(as.data.frame(summary(pairs_q4g_diseng_rt)),
          "output/tables/Q4_group_pairwise_disengagement_rt.csv", row.names = FALSE)


# ── Affected group: Gap_amp (lm, ICC ~ 0%) ───────────────────────────────────

cat("\n\n========== Q4 Affected Group: Gap_amp ==========\n")
cat("N =", nrow(data_3lvl_gap_amp), "\n")

lm_q4g_gap_amp_m0 <- lm(Gap_amp ~ age + sex + site, data = data_3lvl_gap_amp)
lm_q4g_gap_amp_m1 <- lm(Gap_amp ~ affected_group + age + sex + site,
                          data = data_3lvl_gap_amp)
lm_q4g_gap_amp_m2 <- lm(Gap_amp ~ affected_group * age + sex + site,
                          data = data_3lvl_gap_amp)

cat("\nM0 vs M1 (group main effect):\n");       print(anova(lm_q4g_gap_amp_m0, lm_q4g_gap_amp_m1))
cat("\nM1 vs M2 (group x age interaction):\n"); print(anova(lm_q4g_gap_amp_m1, lm_q4g_gap_amp_m2))
cat("\nM1 summary:\n"); print(summary(lm_q4g_gap_amp_m1))
cat("\nM2 summary:\n"); print(summary(lm_q4g_gap_amp_m2))

emm_q4g_gap_amp   <- emmeans(lm_q4g_gap_amp_m1, ~ affected_group,
                               at = list(age = median(data_3lvl_gap_amp$age, na.rm = TRUE)))
pairs_q4g_gap_amp <- pairs(emm_q4g_gap_amp, adjust = "tukey")
eff_q4g_gap_amp   <- eff_size(emm_q4g_gap_amp,
                               sigma = sigma(lm_q4g_gap_amp_m1),
                               edf   = df.residual(lm_q4g_gap_amp_m1))
cat("\nMarginal means (M1, at median age):\n"); print(emm_q4g_gap_amp)
cat("\nPairwise contrasts:\n");                 print(pairs_q4g_gap_amp)
cat("\nEffect sizes:\n");                       print(eff_q4g_gap_amp)
write.csv(as.data.frame(summary(pairs_q4g_gap_amp)),
          "output/tables/Q4_group_pairwise_Gap_amp.csv", row.names = FALSE)


# ── Affected group: Overlap_amp (lmer, ICC > 3%) ─────────────────────────────

cat("\n\n========== Q4 Affected Group: Overlap_amp ==========\n")
cat("N =", nrow(data_3lvl_ovlp_amp), "\n")

# REML = FALSE for LRT
lmer_q4g_ovlp_amp_m0 <- lmer(Overlap_amp ~ age + sex + site + (1 | fam_id),
                              data = data_3lvl_ovlp_amp, REML = FALSE)
lmer_q4g_ovlp_amp_m1 <- lmer(Overlap_amp ~ affected_group + age + sex + site + (1 | fam_id),
                              data = data_3lvl_ovlp_amp, REML = FALSE)
lmer_q4g_ovlp_amp_m2 <- lmer(Overlap_amp ~ affected_group * age + sex + site + (1 | fam_id),
                              data = data_3lvl_ovlp_amp, REML = FALSE)

cat("\nM0 vs M1 (group main effect):\n");       print(anova(lmer_q4g_ovlp_amp_m0, lmer_q4g_ovlp_amp_m1))
cat("\nM1 vs M2 (group x age interaction):\n"); print(anova(lmer_q4g_ovlp_amp_m1, lmer_q4g_ovlp_amp_m2))

# Refit REML = TRUE for summaries/emmeans
lmer_q4g_ovlp_amp_m1r <- lmer(Overlap_amp ~ affected_group + age + sex + site + (1 | fam_id),
                               data = data_3lvl_ovlp_amp, REML = TRUE)
lmer_q4g_ovlp_amp_m2r <- lmer(Overlap_amp ~ affected_group * age + sex + site + (1 | fam_id),
                               data = data_3lvl_ovlp_amp, REML = TRUE)
cat("\nM1 summary (REML):\n"); print(summary(lmer_q4g_ovlp_amp_m1r))
cat("\nM2 summary (REML):\n"); print(summary(lmer_q4g_ovlp_amp_m2r))

emm_q4g_ovlp_amp   <- emmeans(lmer_q4g_ovlp_amp_m1r, ~ affected_group,
                                at = list(age = median(data_3lvl_ovlp_amp$age, na.rm = TRUE)))
pairs_q4g_ovlp_amp <- pairs(emm_q4g_ovlp_amp, adjust = "tukey")
eff_q4g_ovlp_amp   <- eff_size(emm_q4g_ovlp_amp,
                                sigma = sigma(lmer_q4g_ovlp_amp_m1r),
                                edf   = df.residual(lmer_q4g_ovlp_amp_m1r))
cat("\nMarginal means (M1, at median age):\n"); print(emm_q4g_ovlp_amp)
cat("\nPairwise contrasts:\n");                 print(pairs_q4g_ovlp_amp)
cat("\nEffect sizes:\n");                       print(eff_q4g_ovlp_amp)
write.csv(as.data.frame(summary(pairs_q4g_ovlp_amp)),
          "output/tables/Q4_group_pairwise_Overlap_amp.csv", row.names = FALSE)


# ── Q4 Affected-Group BH-FDR (2 families: main effect, interaction) ──────────

q4_grp_p_main <- c(
  anova(lm_q4g_facil_rt_m0,    lm_q4g_facil_rt_m1)$`Pr(>F)`[2],
  anova(lm_q4g_diseng_rt_m0,   lm_q4g_diseng_rt_m1)$`Pr(>F)`[2],
  anova(lm_q4g_gap_amp_m0,     lm_q4g_gap_amp_m1)$`Pr(>F)`[2],
  anova(lmer_q4g_ovlp_amp_m0,  lmer_q4g_ovlp_amp_m1)$`Pr(>Chisq)`[2]
)
q4_grp_p_int <- c(
  anova(lm_q4g_facil_rt_m1,    lm_q4g_facil_rt_m2)$`Pr(>F)`[2],
  anova(lm_q4g_diseng_rt_m1,   lm_q4g_diseng_rt_m2)$`Pr(>F)`[2],
  anova(lm_q4g_gap_amp_m1,     lm_q4g_gap_amp_m2)$`Pr(>F)`[2],
  anova(lmer_q4g_ovlp_amp_m1,  lmer_q4g_ovlp_amp_m2)$`Pr(>Chisq)`[2]
)

q4_grp_fdr <- data.frame(
  Outcome          = c("facilitation_rt", "disengagement_rt", "Gap_amp", "Overlap_amp"),
  p_grp_main       = round(q4_grp_p_main, 4),
  p_grp_main_FDR   = round(p.adjust(q4_grp_p_main, method = "BH"), 4),
  sig_main         = ifelse(p.adjust(q4_grp_p_main, method = "BH") < .05, "*", "ns"),
  p_grp_x_age      = round(q4_grp_p_int, 4),
  p_grp_x_age_FDR  = round(p.adjust(q4_grp_p_int,  method = "BH"), 4),
  sig_int          = ifelse(p.adjust(q4_grp_p_int,  method = "BH") < .05, "*", "ns")
)
cat("\n=== Q4 Affected-Group BH-FDR Summary ===\n"); print(q4_grp_fdr)
write.csv(q4_grp_fdr, "output/tables/Q4_group_BH_FDR_summary.csv", row.names = FALSE)

# ── Q4 Affected-Group Trajectory Plots (M2: group × age) ─────────────────────
p4_grp_facil_rt  <- make_group_traj_plot(lm_q4g_facil_rt_m2,     data_3lvl_facil_rt,  "facilitation_rt")
p4_grp_diseng_rt <- make_group_traj_plot(lm_q4g_diseng_rt_m2,    data_3lvl_diseng_rt, "disengagement_rt")
p4_grp_gap_amp   <- make_group_traj_plot(lm_q4g_gap_amp_m2,      data_3lvl_gap_amp,   "Gap_amp")
p4_grp_ovlp_amp  <- make_group_traj_plot(lmer_q4g_ovlp_amp_m2r,  data_3lvl_ovlp_amp,  "Overlap_amp")

ggsave("output/figures/Q4_group_traj_facilitation_rt.png",  p4_grp_facil_rt,  width = 10, height = 6, dpi = 300, bg = "white")
ggsave("output/figures/Q4_group_traj_disengagement_rt.png", p4_grp_diseng_rt, width = 10, height = 6, dpi = 300, bg = "white")
ggsave("output/figures/Q4_group_traj_Gap_amp.png",          p4_grp_gap_amp,   width = 10, height = 6, dpi = 300, bg = "white")
ggsave("output/figures/Q4_group_traj_Overlap_amp.png",      p4_grp_ovlp_amp,  width = 10, height = 6, dpi = 300, bg = "white")


# =============================================================================
# Q4c: PEDIATRIC SENSITIVITY (age <= 18, supplementary)
# M1 only (main effect) for both NDD and affected_group.
# Not included in primary BH-FDR families.
# =============================================================================

cat("\n\n=== Q4c (Supplementary): Pediatric Sensitivity (age <= 18) ===\n")

data_peds_facil_rt  <- data_facil_rt [data_facil_rt$age  <= 18, ]
data_peds_diseng_rt <- data_diseng_rt[data_diseng_rt$age <= 18, ]
data_peds_gap_amp   <- data_gap_amp  [data_gap_amp$age   <= 18, ]
data_peds_ovlp_amp  <- data_ovlp_amp [data_ovlp_amp$age  <= 18, ]

cat("Pediatric N: facil_rt =", nrow(data_peds_facil_rt),
    "| diseng_rt =", nrow(data_peds_diseng_rt),
    "| Gap_amp =",   nrow(data_peds_gap_amp),
    "| Overlap_amp =", nrow(data_peds_ovlp_amp), "\n")

# NDD M1 — pediatric
lm_peds_ndd_facil_rt  <- lm(facilitation_rt  ~ ndd + age + sex + site, data = data_peds_facil_rt)
lm_peds_ndd_diseng_rt <- lm(disengagement_rt ~ ndd + age + sex + site, data = data_peds_diseng_rt)
lm_peds_ndd_gap_amp   <- lm(Gap_amp          ~ ndd + age + sex + site, data = data_peds_gap_amp)
lmer_peds_ndd_ovlp_amp <- lmer(Overlap_amp   ~ ndd + age + sex + site + (1 | fam_id),
                                data = data_peds_ovlp_amp, REML = TRUE)

cat("\n--- Peds NDD: facilitation_rt ---\n");  print(summary(lm_peds_ndd_facil_rt))
cat("\n--- Peds NDD: disengagement_rt ---\n"); print(summary(lm_peds_ndd_diseng_rt))
cat("\n--- Peds NDD: Gap_amp ---\n");          print(summary(lm_peds_ndd_gap_amp))
cat("\n--- Peds NDD: Overlap_amp ---\n");      print(summary(lmer_peds_ndd_ovlp_amp))

# Affected-group M1 — pediatric
data_peds_3lvl_facil_rt  <- data_3lvl_facil_rt [data_3lvl_facil_rt$age  <= 18, ]
data_peds_3lvl_diseng_rt <- data_3lvl_diseng_rt[data_3lvl_diseng_rt$age <= 18, ]
data_peds_3lvl_gap_amp   <- data_3lvl_gap_amp  [data_3lvl_gap_amp$age   <= 18, ]
data_peds_3lvl_ovlp_amp  <- data_3lvl_ovlp_amp [data_3lvl_ovlp_amp$age  <= 18, ]

lm_peds_grp_facil_rt  <- lm(facilitation_rt  ~ affected_group + age + sex + site, data = data_peds_3lvl_facil_rt)
lm_peds_grp_diseng_rt <- lm(disengagement_rt ~ affected_group + age + sex + site, data = data_peds_3lvl_diseng_rt)
lm_peds_grp_gap_amp   <- lm(Gap_amp          ~ affected_group + age + sex + site, data = data_peds_3lvl_gap_amp)
lmer_peds_grp_ovlp_amp <- lmer(Overlap_amp   ~ affected_group + age + sex + site + (1 | fam_id),
                                data = data_peds_3lvl_ovlp_amp, REML = TRUE)

cat("\n--- Peds Group: facilitation_rt ---\n");  print(summary(lm_peds_grp_facil_rt))
cat("\n--- Peds Group: disengagement_rt ---\n"); print(summary(lm_peds_grp_diseng_rt))
cat("\n--- Peds Group: Gap_amp ---\n");          print(summary(lm_peds_grp_gap_amp))
cat("\n--- Peds Group: Overlap_amp ---\n");      print(summary(lmer_peds_grp_ovlp_amp))


# =============================================================================
# Q5. ERP AMPLITUDE → RT COUPLING
# Four explicit pairings (no latency, no NDD interaction):
#   1. Gap_amp    → Gap_rt         (lm,   Gap_rt ICC ~ 0%)
#   2. Overlap_amp → Overlap_rt    (lmer, Overlap_rt ICC > 3%)
#   3. Gap_amp    → facilitation_rt  (lm,   facilitation_rt ICC ~ 0%)
#   4. Overlap_amp → disengagement_rt (lm,  disengagement_rt ICC ~ 0%)
#
# Steps per pairing: bivariate correlation → scatter plot → adjusted model
# BH-FDR: single family of 4 tests (ERP amplitude predictor p from each model)
# =============================================================================

# Data subsets — complete cases for each pairing (outcome + predictor + covariates)
data_q5_gap_rt     <- data[!is.na(data$Gap_rt)            & !is.na(data$Gap_amp),     ]
data_q5_ovlp_rt    <- data[!is.na(data$Overlap_rt)        & !is.na(data$Overlap_amp), ]
data_q5_facil_rt   <- data[!is.na(data$facilitation_rt)   & !is.na(data$Gap_amp),     ]
data_q5_diseng_rt  <- data[!is.na(data$disengagement_rt)  & !is.na(data$Overlap_amp), ]

cat("\n=== Q5 sample sizes ===\n")
cat("Gap_amp    → Gap_rt:          N =", nrow(data_q5_gap_rt),    "\n")
cat("Overlap_amp → Overlap_rt:     N =", nrow(data_q5_ovlp_rt),   "\n")
cat("Gap_amp    → facilitation_rt: N =", nrow(data_q5_facil_rt),  "\n")
cat("Overlap_amp → disengagement_rt: N =", nrow(data_q5_diseng_rt), "\n")


# ── Bivariate Pearson correlations ────────────────────────────────────────────

cat("\n--- Q5 Bivariate Correlations ---\n")

cor_q5_gap_rt    <- cor.test(data_q5_gap_rt$Gap_amp,     data_q5_gap_rt$Gap_rt,
                              method = "pearson")
cor_q5_ovlp_rt   <- cor.test(data_q5_ovlp_rt$Overlap_amp, data_q5_ovlp_rt$Overlap_rt,
                              method = "pearson")
cor_q5_facil_rt  <- cor.test(data_q5_facil_rt$Gap_amp,   data_q5_facil_rt$facilitation_rt,
                              method = "pearson")
cor_q5_diseng_rt <- cor.test(data_q5_diseng_rt$Overlap_amp, data_q5_diseng_rt$disengagement_rt,
                              method = "pearson")

cor_q5_summary <- data.frame(
  Pairing   = c("Gap_amp -> Gap_rt", "Overlap_amp -> Overlap_rt",
                "Gap_amp -> facilitation_rt", "Overlap_amp -> disengagement_rt"),
  r         = round(c(cor_q5_gap_rt$estimate,   cor_q5_ovlp_rt$estimate,
                      cor_q5_facil_rt$estimate, cor_q5_diseng_rt$estimate), 3),
  t         = round(c(cor_q5_gap_rt$statistic,   cor_q5_ovlp_rt$statistic,
                      cor_q5_facil_rt$statistic, cor_q5_diseng_rt$statistic), 3),
  df        = round(c(cor_q5_gap_rt$parameter,   cor_q5_ovlp_rt$parameter,
                      cor_q5_facil_rt$parameter, cor_q5_diseng_rt$parameter), 1),
  p         = round(c(cor_q5_gap_rt$p.value,     cor_q5_ovlp_rt$p.value,
                      cor_q5_facil_rt$p.value,   cor_q5_diseng_rt$p.value), 4),
  CI_lower  = round(c(cor_q5_gap_rt$conf.int[1],   cor_q5_ovlp_rt$conf.int[1],
                      cor_q5_facil_rt$conf.int[1],  cor_q5_diseng_rt$conf.int[1]), 3),
  CI_upper  = round(c(cor_q5_gap_rt$conf.int[2],   cor_q5_ovlp_rt$conf.int[2],
                      cor_q5_facil_rt$conf.int[2],  cor_q5_diseng_rt$conf.int[2]), 3)
)
print(cor_q5_summary)
write.csv(cor_q5_summary, "output/tables/Q5_bivariate_correlations.csv", row.names = FALSE)


# ── Scatter plots ─────────────────────────────────────────────────────────────

p5_scatter_gap_rt <- ggplot(data_q5_gap_rt, aes(x = Gap_amp, y = Gap_rt)) +
  geom_point(alpha = 0.4, size = 1.8, color = "steelblue") +
  geom_smooth(method = "lm", se = TRUE, color = "steelblue4", fill = "steelblue", alpha = 0.2) +
  labs(title    = "Q5: Gap_amp vs Gap_rt",
       subtitle = paste0("r = ", round(cor_q5_gap_rt$estimate, 3),
                         ", p = ", round(cor_q5_gap_rt$p.value, 4)),
       x = "Gap_amp", y = "Gap_rt") + theme_pub

p5_scatter_ovlp_rt <- ggplot(data_q5_ovlp_rt, aes(x = Overlap_amp, y = Overlap_rt)) +
  geom_point(alpha = 0.4, size = 1.8, color = "tomato") +
  geom_smooth(method = "lm", se = TRUE, color = "tomato3", fill = "tomato", alpha = 0.2) +
  labs(title    = "Q5: Overlap_amp vs Overlap_rt",
       subtitle = paste0("r = ", round(cor_q5_ovlp_rt$estimate, 3),
                         ", p = ", round(cor_q5_ovlp_rt$p.value, 4)),
       x = "Overlap_amp", y = "Overlap_rt") + theme_pub

p5_scatter_facil_rt <- ggplot(data_q5_facil_rt, aes(x = Gap_amp, y = facilitation_rt)) +
  geom_point(alpha = 0.4, size = 1.8, color = "darkorchid") +
  geom_smooth(method = "lm", se = TRUE, color = "darkorchid4", fill = "darkorchid", alpha = 0.2) +
  labs(title    = "Q5: Gap_amp vs facilitation_rt",
       subtitle = paste0("r = ", round(cor_q5_facil_rt$estimate, 3),
                         ", p = ", round(cor_q5_facil_rt$p.value, 4)),
       x = "Gap_amp", y = "facilitation_rt") + theme_pub

p5_scatter_diseng_rt <- ggplot(data_q5_diseng_rt, aes(x = Overlap_amp, y = disengagement_rt)) +
  geom_point(alpha = 0.4, size = 1.8, color = "darkorange") +
  geom_smooth(method = "lm", se = TRUE, color = "darkorange4", fill = "darkorange", alpha = 0.2) +
  labs(title    = "Q5: Overlap_amp vs disengagement_rt",
       subtitle = paste0("r = ", round(cor_q5_diseng_rt$estimate, 3),
                         ", p = ", round(cor_q5_diseng_rt$p.value, 4)),
       x = "Overlap_amp", y = "disengagement_rt") + theme_pub

ggsave("output/figures/Q5_scatter_Gap_amp_Gap_rt.png",             p5_scatter_gap_rt,    width = 7, height = 5, dpi = 300, bg = "white")
ggsave("output/figures/Q5_scatter_Overlap_amp_Overlap_rt.png",     p5_scatter_ovlp_rt,   width = 7, height = 5, dpi = 300, bg = "white")
ggsave("output/figures/Q5_scatter_Gap_amp_facilitation_rt.png",    p5_scatter_facil_rt,  width = 7, height = 5, dpi = 300, bg = "white")
ggsave("output/figures/Q5_scatter_Overlap_amp_disengagement_rt.png", p5_scatter_diseng_rt, width = 7, height = 5, dpi = 300, bg = "white")

# Combined scatter panel
p5_panel <- (p5_scatter_gap_rt + p5_scatter_ovlp_rt) /
            (p5_scatter_facil_rt + p5_scatter_diseng_rt) +
  plot_annotation(title = "Q5: ERP Amplitude -> RT",
                  theme = theme(plot.title = element_text(face = "bold", size = 14)))
ggsave("output/figures/Q5_scatter_panel.png", p5_panel, width = 14, height = 10, dpi = 300, bg = "white")


# ── Adjusted regression models ────────────────────────────────────────────────

# 1. Gap_amp -> Gap_rt (lm; Gap_rt ICC ~ 0%)
cat("\n\n--- Q5 Model 1: Gap_rt ~ Gap_amp + age + sex + site ---\n")
lm_q5_gap_rt <- lm(Gap_rt ~ Gap_amp + age + sex + site, data = data_q5_gap_rt)
print(summary(lm_q5_gap_rt))

# 2. Overlap_amp -> Overlap_rt (lmer; Overlap_rt ICC > 3%)
cat("\n\n--- Q5 Model 2: Overlap_rt ~ Overlap_amp + age + sex + site + (1|fam_id) ---\n")
lmer_q5_ovlp_rt <- lmer(Overlap_rt ~ Overlap_amp + age + sex + site + (1 | fam_id),
                         data = data_q5_ovlp_rt, REML = TRUE)
print(summary(lmer_q5_ovlp_rt))

# 3. Gap_amp -> facilitation_rt (lm; facilitation_rt ICC ~ 0%)
cat("\n\n--- Q5 Model 3: facilitation_rt ~ Gap_amp + age + sex + site ---\n")
lm_q5_facil_rt <- lm(facilitation_rt ~ Gap_amp + age + sex + site, data = data_q5_facil_rt)
print(summary(lm_q5_facil_rt))

# 4. Overlap_amp -> disengagement_rt (lm; disengagement_rt ICC ~ 0%)
cat("\n\n--- Q5 Model 4: disengagement_rt ~ Overlap_amp + age + sex + site ---\n")
lm_q5_diseng_rt <- lm(disengagement_rt ~ Overlap_amp + age + sex + site, data = data_q5_diseng_rt)
print(summary(lm_q5_diseng_rt))


# ── Q5 BH-FDR: single family of 4 tests ──────────────────────────────────────
# Extract p-value for the ERP amplitude predictor from each model.
# For lmer: lmerTest Satterthwaite p-value via summary()$coefficients.

q5_p_gap_rt    <- summary(lm_q5_gap_rt)$coefficients["Gap_amp",     "Pr(>|t|)"]
q5_p_ovlp_rt   <- summary(lmer_q5_ovlp_rt)$coefficients["Overlap_amp", "Pr(>|t|)"]
q5_p_facil_rt  <- summary(lm_q5_facil_rt)$coefficients["Gap_amp",     "Pr(>|t|)"]
q5_p_diseng_rt <- summary(lm_q5_diseng_rt)$coefficients["Overlap_amp", "Pr(>|t|)"]

q5_fdr <- data.frame(
  Pairing    = c("Gap_amp -> Gap_rt", "Overlap_amp -> Overlap_rt",
                 "Gap_amp -> facilitation_rt", "Overlap_amp -> disengagement_rt"),
  Model      = c("lm", "lmer", "lm", "lm"),
  p_amp      = round(c(q5_p_gap_rt, q5_p_ovlp_rt, q5_p_facil_rt, q5_p_diseng_rt), 4),
  p_amp_FDR  = round(p.adjust(
    c(q5_p_gap_rt, q5_p_ovlp_rt, q5_p_facil_rt, q5_p_diseng_rt),
    method = "BH"), 4)
)
q5_fdr$sig_FDR <- ifelse(q5_fdr$p_amp_FDR < .001, "***",
                          ifelse(q5_fdr$p_amp_FDR < .01,  "**",
                                 ifelse(q5_fdr$p_amp_FDR < .05, "*", "ns")))

cat("\n=== Q5 ERP Amplitude -> RT: BH-FDR (4 tests, single family) ===\n")
print(q5_fdr)
write.csv(q5_fdr, "output/tables/Q5_ERP_RT_BH_FDR.csv", row.names = FALSE)

cat("\n\n=== ANALYSIS COMPLETE ===\n")
cat("All outputs saved to: output/tables/, output/figures/, output/supplementary/\n")
