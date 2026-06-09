1+1
# =============================================================================
# GAP-OVERLAP TASK ANALYSIS — Full Script
# Q1. Are facilitation and disengagement effects present, and how do they
#     change across age?
# Q2. Are age trajectories moderated by NDD status, ASD diagnosis, or Sex?
# Q3. Do ERP amplitude and latency predict reaction time (neural-behavioral
#     coupling)?
# =============================================================================

library(mgcv)
library(lme4)
library(lmerTest)
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
library(pwr)
library(ggsignif)

set.seed(333)


# ── 1. DIRECTORIES & DATA LOAD ───────────────────────────────────────────────

setwd("C:/Users/gabot/OneDrive - McGill University/Desktop/Github_repos/q1k_neurosubs/code/linear_models/go_task_full")

for (d in c("output", "output/figures", "output/tables",
            "output/diagnostics", "output/supplementary")) {
  dir.create(d, showWarnings = FALSE, recursive = TRUE)
}

data_path <- "C:/Users/gabot/OneDrive - McGill University/Desktop/github_repos//q1k_multimodal_subtypes/outputs/gap/erp_rt_combined.csv"
raw_data  <- read.csv(data_path, stringsAsFactors = FALSE)

cat("Raw data dimensions:", nrow(raw_data), "rows x", ncol(raw_data), "cols\n")
print(head(raw_data))
print(str(raw_data))


# ── 2. DATA PREPARATION ──────────────────────────────────────────────────────

data <- raw_data

data$fam_id <- str_extract(data$subject, "^[0-9]+")
cat("\nNumber of unique families:", length(unique(data$fam_id)), "\n")
cat("Number of subjects:", nrow(data), "\n")

data$facilitation_rt_check  <- data$Baseline_rt - data$Gap_rt
data$disengagement_rt_check <- data$Overlap_rt  - data$Baseline_rt

rt_discrepancy <- data %>%
  filter(!is.na(facilitation_rt) & !is.na(facilitation_rt_check)) %>%
  summarise(
    max_facil_diff  = max(abs(facilitation_rt  - facilitation_rt_check),  na.rm = TRUE),
    max_diseng_diff = max(abs(disengagement_rt - disengagement_rt_check), na.rm = TRUE)
  )
cat("\nRT difference score verification:\n")
print(rt_discrepancy)

data$ndd <- factor(data$ndd, levels = c(0, 1), labels = c("Non-NDD", "NDD"))
data$asd <- factor(data$asd, levels = c(0, 1), labels = c("Non-ASD", "ASD"))
data$affected_group <- factor(data$affected_group,
                              levels = c("non-affected", "affected", "asd"),
                              labels = c("Non-Affected", "Affected (non-ASD)", "ASD"))
data$sex <- factor(data$sex, levels = c("Female", "Male"))
data$site <- droplevels(
  factor(ifelse(data$site == "" | is.na(data$site), NA_character_, as.character(data$site)))
)
data$fam_id  <- factor(data$fam_id)
data$age_bin <- factor(data$age_bin,
                       levels = c("Children\n(≤11)", "Adolescents\n(12–17)",
                                  "Adults\n(18–44)", "Older Adults\n(45+)"))
data$age <- as.numeric(data$eeg_age)

data$facilitation_amp  <- as.numeric(data$facilitation_amp)
data$disengagement_amp <- as.numeric(data$disengagement_amp)
data$facilitation_lat  <- as.numeric(data$facilitation_lat)
data$disengagement_lat <- as.numeric(data$disengagement_lat)
data$facilitation_rt   <- as.numeric(data$facilitation_rt)
data$disengagement_rt  <- as.numeric(data$disengagement_rt)
data$Gap_amp           <- as.numeric(data$Gap_amp)
data$Overlap_amp       <- as.numeric(data$Overlap_amp)
data$Gap_rt            <- as.numeric(data$Gap_rt)
data$Overlap_rt        <- as.numeric(data$Overlap_rt)

cat("\nFactor level counts:\n")
cat("NDD:           ");     print(table(data$ndd,            useNA = "always"))
cat("ASD:           ");     print(table(data$asd,            useNA = "always"))
cat("Affected group:");     print(table(data$affected_group, useNA = "always"))
cat("Sex:           ");     print(table(data$sex,            useNA = "always"))
cat("Site:          ");     print(table(data$site,           useNA = "always"))
cat("Age bin:       ");     print(table(data$age_bin,        useNA = "always"))

# Drop those with Nas in EEG age and NDD
data <- data[!is.na(data$age), ]
data <- data[!is.na(data$ndd), ]

age_bin_order <- c("Children\n(≤11)", "Adolescents\n(12–17)", "Adults\n(18–44)", "Older Adults\n(45+)")
data$age_bin  <- factor(data$age_bin, levels = age_bin_order, ordered = TRUE)
# ── HELPER: marginal prediction from GAM (excludes site & family RE) ─────────

predict_marginal <- function(model, newdata, site_levels, fam_levels) {
  newdata$site   <- factor("hsj", levels = site_levels)
  newdata$fam_id <- factor(fam_levels[1], levels = fam_levels)
  predict(model, newdata = newdata, type = "response", se.fit = TRUE,
          exclude = "s(fam_id)", newdata.guaranteed = TRUE)
}


# ── 3. MISSING DATA REPORT ───────────────────────────────────────────────────

all_outcomes    <- c("facilitation_amp", "disengagement_amp",
                     "facilitation_lat", "disengagement_lat",
                     "facilitation_rt",  "disengagement_rt")

missing_report <- data.frame(
  Variable    = all_outcomes,
  N_total     = nrow(data),
  N_observed  = sapply(all_outcomes, function(v) sum(!is.na(data[[v]]))),
  N_missing   = sapply(all_outcomes, function(v) sum( is.na(data[[v]]))),
  Pct_missing = sapply(all_outcomes, function(v) round(100 * mean(is.na(data[[v]])), 2))
)
print(missing_report)
write.csv(missing_report, "output/tables/missing_data_report.csv", row.names = FALSE)

cat("\n--- Missing data characterization: facilitation_rt ---\n")
data$rt_missing_flag <- is.na(data$facilitation_rt)
t_facil_age  <- t.test(age ~ rt_missing_flag, data = data)
chi_facil_ndd <- chisq.test(table(data$ndd, data$rt_missing_flag))
chi_facil_sex <- chisq.test(table(data$sex, data$rt_missing_flag))
cat("Age: missers M =", round(mean(data$age[data$rt_missing_flag],  na.rm = TRUE), 1),
    "vs observed M =", round(mean(data$age[!data$rt_missing_flag], na.rm = TRUE), 1),
    "| t =", round(t_facil_age$statistic, 2), ", p =", round(t_facil_age$p.value, 4), "\n")
cat("NDD × missing χ²:", round(chi_facil_ndd$statistic, 2), ", p =", round(chi_facil_ndd$p.value, 4), "\n")
cat("Sex × missing χ²:", round(chi_facil_sex$statistic, 2), ", p =", round(chi_facil_sex$p.value, 4), "\n")
data$rt_missing_flag <- NULL

cat("\n--- Missing data characterization: disengagement_rt ---\n")
data$rt_missing_flag <- is.na(data$disengagement_rt)
t_diseng_age  <- t.test(age ~ rt_missing_flag, data = data)
chi_diseng_ndd <- chisq.test(table(data$ndd, data$rt_missing_flag))
chi_diseng_sex <- chisq.test(table(data$sex, data$rt_missing_flag))
cat("Age: missers M =", round(mean(data$age[data$rt_missing_flag],  na.rm = TRUE), 1),
    "vs observed M =", round(mean(data$age[!data$rt_missing_flag], na.rm = TRUE), 1),
    "| t =", round(t_diseng_age$statistic, 2), ", p =", round(t_diseng_age$p.value, 4), "\n")
cat("NDD × missing χ²:", round(chi_diseng_ndd$statistic, 2), ", p =", round(chi_diseng_ndd$p.value, 4), "\n")
cat("Sex × missing χ²:", round(chi_diseng_sex$statistic, 2), ", p =", round(chi_diseng_sex$p.value, 4), "\n")
data$rt_missing_flag <- NULL


# ── 4. OUTLIER DETECTION (3 SD) ──────────────────────────────────────────────

detect_outliers_3sd <- function(x) {
  m <- mean(x, na.rm = TRUE)
  s <- sd(x, na.rm = TRUE)
  
  out <- abs(x - m) > 3 * s
  out[is.na(x)] <- FALSE   # KEY FIX
  
  return(out)
}

out_facil_amp  <- detect_outliers_3sd(data$facilitation_amp)
out_diseng_amp <- detect_outliers_3sd(data$disengagement_amp)
out_facil_lat  <- detect_outliers_3sd(data$facilitation_lat)
out_diseng_lat <- detect_outliers_3sd(data$disengagement_lat)
out_facil_rt   <- detect_outliers_3sd(data$facilitation_rt)
out_diseng_rt  <- detect_outliers_3sd(data$disengagement_rt)

cat("Outliers (3 SD) — facilitation_amp:",  sum(out_facil_amp),  "\n")
cat("Outliers (3 SD) — disengagement_amp:", sum(out_diseng_amp), "\n")
cat("Outliers (3 SD) — facilitation_lat:",  sum(out_facil_lat),  "\n")
cat("Outliers (3 SD) — disengagement_lat:", sum(out_diseng_lat), "\n")
cat("Outliers (3 SD) — facilitation_rt:",   sum(out_facil_rt),   "\n")
cat("Outliers (3 SD) — disengagement_rt:",  sum(out_diseng_rt),  "\n")

#data$facilitation_amp [out_facil_amp]  <- NA
#data$disengagement_amp[out_diseng_amp] <- NA
#data$facilitation_lat [out_facil_lat]  <- NA
#data$disengagement_lat[out_diseng_lat] <- NA
#data$facilitation_rt  [out_facil_rt]   <- NA
#data$disengagement_rt [out_diseng_rt]  <- NA

outlier_report <- data.frame(
  Variable     = all_outcomes,
  N_outliers   = c(sum(out_facil_amp), sum(out_diseng_amp),
                   sum(out_facil_lat), sum(out_diseng_lat),
                   sum(out_facil_rt),  sum(out_diseng_rt)),
  Pct_outliers = round(100 * c(mean(out_facil_amp), mean(out_diseng_amp),
                               mean(out_facil_lat), mean(out_diseng_lat),
                               mean(out_facil_rt),  mean(out_diseng_rt)), 2)
)
print(outlier_report)
write.csv(outlier_report, "output/tables/outlier_report.csv", row.names = FALSE)


# ── 5. SAMPLE DESCRIPTIVES ───────────────────────────────────────────────────

cat("\n=== OVERALL SAMPLE ===\n")
cat("N =", nrow(data), "\n")
cat("Age: M =", round(mean(data$age, na.rm = TRUE), 1),
    "SD =", round(sd(data$age, na.rm = TRUE), 1),
    "Range:", round(min(data$age, na.rm = TRUE), 1),
    "–", round(max(data$age, na.rm = TRUE), 1), "\n")

desc_by_ndd <- data %>%
  group_by(ndd) %>%
  summarise(N          = n(),
            Age_M      = round(mean(age, na.rm = TRUE), 2),
            Age_SD     = round(sd(age,   na.rm = TRUE), 2),
            Pct_Female = round(100 * mean(sex == "Female", na.rm = TRUE), 1),
            facil_amp_M  = round(mean(facilitation_amp,  na.rm = TRUE), 2),
            facil_amp_SD = round(sd(facilitation_amp,    na.rm = TRUE), 2),
            diseng_amp_M  = round(mean(disengagement_amp, na.rm = TRUE), 2),
            diseng_amp_SD = round(sd(disengagement_amp,   na.rm = TRUE), 2),
            facil_lat_M  = round(mean(facilitation_lat,  na.rm = TRUE), 2),
            facil_lat_SD = round(sd(facilitation_lat,    na.rm = TRUE), 2),
            diseng_lat_M  = round(mean(disengagement_lat, na.rm = TRUE), 2),
            diseng_lat_SD = round(sd(disengagement_lat,   na.rm = TRUE), 2),
            facil_rt_M  = round(mean(facilitation_rt,  na.rm = TRUE), 2),
            facil_rt_SD = round(sd(facilitation_rt,    na.rm = TRUE), 2),
            diseng_rt_M  = round(mean(disengagement_rt, na.rm = TRUE), 2),
            diseng_rt_SD = round(sd(disengagement_rt,   na.rm = TRUE), 2),
            .groups = "drop")
print(desc_by_ndd)
write.csv(desc_by_ndd, "output/tables/descriptives_by_NDD.csv", row.names = FALSE)

desc_by_group <- data %>%
  group_by(affected_group) %>%
  summarise(N          = n(),
            Age_M      = round(mean(age, na.rm = TRUE), 2),
            Age_SD     = round(sd(age,   na.rm = TRUE), 2),
            Pct_Female = round(100 * mean(sex == "Female", na.rm = TRUE), 1),
            facil_amp_M  = round(mean(facilitation_amp,  na.rm = TRUE), 2),
            facil_amp_SD = round(sd(facilitation_amp,    na.rm = TRUE), 2),
            diseng_amp_M  = round(mean(disengagement_amp, na.rm = TRUE), 2),
            diseng_amp_SD = round(sd(disengagement_amp,   na.rm = TRUE), 2),
            facil_lat_M  = round(mean(facilitation_lat,  na.rm = TRUE), 2),
            facil_lat_SD = round(sd(facilitation_lat,    na.rm = TRUE), 2),
            diseng_lat_M  = round(mean(disengagement_lat, na.rm = TRUE), 2),
            diseng_lat_SD = round(sd(disengagement_lat,   na.rm = TRUE), 2),
            facil_rt_M  = round(mean(facilitation_rt,  na.rm = TRUE), 2),
            facil_rt_SD = round(sd(facilitation_rt,    na.rm = TRUE), 2),
            diseng_rt_M  = round(mean(disengagement_rt, na.rm = TRUE), 2),
            diseng_rt_SD = round(sd(disengagement_rt,   na.rm = TRUE), 2),
            .groups = "drop")
print(desc_by_group)
write.csv(desc_by_group, "output/tables/descriptives_by_affected_group.csv", row.names = FALSE)

desc_by_agebin <- data %>%
  group_by(age_bin) %>%
  summarise(N      = n(),
            Age_M  = round(mean(age, na.rm = TRUE), 2),
            Age_SD = round(sd(age,   na.rm = TRUE), 2),
            facil_amp_M    = round(mean(facilitation_amp,  na.rm = TRUE), 2),
            facil_amp_SD   = round(sd(facilitation_amp,    na.rm = TRUE), 2),
            facil_amp_Nobs = sum(!is.na(facilitation_amp)),
            diseng_amp_M    = round(mean(disengagement_amp, na.rm = TRUE), 2),
            diseng_amp_SD   = round(sd(disengagement_amp,   na.rm = TRUE), 2),
            diseng_amp_Nobs = sum(!is.na(disengagement_amp)),
            facil_lat_M    = round(mean(facilitation_lat,  na.rm = TRUE), 2),
            facil_lat_SD   = round(sd(facilitation_lat,    na.rm = TRUE), 2),
            facil_lat_Nobs = sum(!is.na(facilitation_lat)),
            diseng_lat_M    = round(mean(disengagement_lat, na.rm = TRUE), 2),
            diseng_lat_SD   = round(sd(disengagement_lat,   na.rm = TRUE), 2),
            diseng_lat_Nobs = sum(!is.na(disengagement_lat)),
            facil_rt_M    = round(mean(facilitation_rt,  na.rm = TRUE), 2),
            facil_rt_SD   = round(sd(facilitation_rt,    na.rm = TRUE), 2),
            facil_rt_Nobs = sum(!is.na(facilitation_rt)),
            diseng_rt_M    = round(mean(disengagement_rt, na.rm = TRUE), 2),
            diseng_rt_SD   = round(sd(disengagement_rt,   na.rm = TRUE), 2),
            diseng_rt_Nobs = sum(!is.na(disengagement_rt)),
            .groups = "drop")
print(desc_by_agebin)
write.csv(desc_by_agebin, "output/tables/descriptives_by_age_bin.csv", row.names = FALSE)

cat("\n=== Age distribution × affected_group ===\n")
age_group_summary <- data %>%
  group_by(affected_group) %>%
  summarise(N       = n(),
            Age_M   = round(mean(age, na.rm = TRUE), 1),
            Age_SD  = round(sd(age,   na.rm = TRUE), 1),
            Age_Min = round(min(age,  na.rm = TRUE), 1),
            Age_Max = round(max(age,  na.rm = TRUE), 1),
            .groups = "drop")
print(age_group_summary)
write.csv(age_group_summary, "output/tables/age_distribution_by_group.csv", row.names = FALSE)

age_aov <- aov(age ~ affected_group, data = data)
cat("\nAge difference across affected_group:\n"); print(summary(age_aov))
cat("\nTukey HSD:\n");                            print(TukeyHSD(age_aov))

t_age_ndd <- t.test(age ~ ndd, data = data)
cat("\nAge NDD vs Non-NDD: t =", round(t_age_ndd$statistic, 2),
    ", p =", round(t_age_ndd$p.value, 4), "\n")
cat("Non-NDD M =", round(mean(data$age[data$ndd == "Non-NDD"], na.rm = TRUE), 1),
    "SD =", round(sd(data$age[data$ndd == "Non-NDD"], na.rm = TRUE), 1), "\n")
cat("NDD    M =", round(mean(data$age[data$ndd == "NDD"], na.rm = TRUE), 1),
    "SD =", round(sd(data$age[data$ndd == "NDD"], na.rm = TRUE), 1), "\n")


# ── 6. FAMILY ICC ────────────────────────────────────────────────────────────

cat("\n=== NULL MODEL ICCs (Family clustering) ===\n")
# ---------- ICC helper (never fails silently) ----------
fit_null_icc <- function(df, outcome, cluster = "fam_id") {
  dat <- df[, c(outcome, cluster)]
  dat <- dat[!is.na(dat[[outcome]]) & !is.na(dat[[cluster]]), , drop = FALSE]
  dat[[cluster]] <- droplevels(as.factor(dat[[cluster]]))
  
  if (nrow(dat) < 10 || nlevels(dat[[cluster]]) < 2) {
    return(data.frame(
      outcome = outcome, N = nrow(dat), n_clusters = nlevels(dat[[cluster]]),
      var_cluster = NA_real_, var_resid = NA_real_, ICC = NA_real_,
      singular = NA, status = "insufficient_data"
    ))
  }
  
  fml <- as.formula(paste0(outcome, " ~ 1 + (1|", cluster, ")"))
  mod <- tryCatch(
    lme4::lmer(fml, data = dat, REML = TRUE),
    error = function(e) e
  )
  
  if (inherits(mod, "error")) {
    return(data.frame(
      outcome = outcome, N = nrow(dat), n_clusters = nlevels(dat[[cluster]]),
      var_cluster = NA_real_, var_resid = NA_real_, ICC = NA_real_,
      singular = NA, status = paste0("fit_error: ", mod$message)
    ))
  }
  
  vc <- as.data.frame(lme4::VarCorr(mod))
  var_cluster <- vc$vcov[vc$grp == cluster]
  var_resid   <- vc$vcov[vc$grp == "Residual"]
  
  if (length(var_cluster) == 0) var_cluster <- 0
  if (length(var_resid) == 0)   var_resid <- NA_real_
  
  icc <- var_cluster / (var_cluster + var_resid)
  
  data.frame(
    outcome = outcome,
    N = nrow(dat),
    n_clusters = nlevels(dat[[cluster]]),
    var_cluster = var_cluster,
    var_resid = var_resid,
    ICC = icc,
    singular = lme4::isSingular(mod, tol = 1e-5),
    status = "ok"
  )
}

outcomes <- c(
  "facilitation_amp", "disengagement_amp",
  "facilitation_lat", "disengagement_lat",
  "facilitation_rt",  "disengagement_rt"
)

icc_results <- dplyr::bind_rows(lapply(outcomes, function(v) fit_null_icc(data, v, "fam_id")))
icc_results$Pct_var <- round(100 * icc_results$ICC, 2)

print(icc_results)
write.csv(icc_results, "output/tables/icc_results.csv", row.names = FALSE)


#Family-level clustering was essentially absent for all outcomes except facilitation latency
#(ICC = 7.96%); accordingly, family random effects were omitted for variables with near-zero 
#ICCs and retained only for facilitation latency.



# ── 7. ASSUMPTION CHECKS ─────────────────────────────────────────────────────

sw_facil_amp  <- shapiro.test(data$facilitation_amp [!is.na(data$facilitation_amp)])
sw_diseng_amp <- shapiro.test(data$disengagement_amp[!is.na(data$disengagement_amp)])
sw_facil_lat  <- shapiro.test(data$facilitation_lat [!is.na(data$facilitation_lat)])
sw_diseng_lat <- shapiro.test(data$disengagement_lat[!is.na(data$disengagement_lat)])
sw_facil_rt   <- shapiro.test(data$facilitation_rt  [!is.na(data$facilitation_rt)])
sw_diseng_rt  <- shapiro.test(data$disengagement_rt [!is.na(data$disengagement_rt)])

assumption_df <- data.frame(
  Outcome   = all_outcomes,
  N         = c(sum(!is.na(data$facilitation_amp)),  sum(!is.na(data$disengagement_amp)),
                sum(!is.na(data$facilitation_lat)),  sum(!is.na(data$disengagement_lat)),
                sum(!is.na(data$facilitation_rt)),   sum(!is.na(data$disengagement_rt))),
  Skewness  = round(c(moments::skewness(data$facilitation_amp,  na.rm = TRUE),
                      moments::skewness(data$disengagement_amp, na.rm = TRUE),
                      moments::skewness(data$facilitation_lat,  na.rm = TRUE),
                      moments::skewness(data$disengagement_lat, na.rm = TRUE),
                      moments::skewness(data$facilitation_rt,   na.rm = TRUE),
                      moments::skewness(data$disengagement_rt,  na.rm = TRUE)), 3),
  Kurtosis  = round(c(moments::kurtosis(data$facilitation_amp,  na.rm = TRUE),
                      moments::kurtosis(data$disengagement_amp, na.rm = TRUE),
                      moments::kurtosis(data$facilitation_lat,  na.rm = TRUE),
                      moments::kurtosis(data$disengagement_lat, na.rm = TRUE),
                      moments::kurtosis(data$facilitation_rt,   na.rm = TRUE),
                      moments::kurtosis(data$disengagement_rt,  na.rm = TRUE)), 3),
  SW_W      = round(c(sw_facil_amp$statistic,  sw_diseng_amp$statistic,
                      sw_facil_lat$statistic,  sw_diseng_lat$statistic,
                      sw_facil_rt$statistic,   sw_diseng_rt$statistic), 4),
  SW_p      = round(c(sw_facil_amp$p.value,    sw_diseng_amp$p.value,
                      sw_facil_lat$p.value,    sw_diseng_lat$p.value,
                      sw_facil_rt$p.value,     sw_diseng_rt$p.value), 4)
)
print(assumption_df)
write.csv(assumption_df, "output/tables/assumption_checks.csv", row.names = FALSE)


# ── 8. DISTRIBUTION PLOTS ────────────────────────────────────────────────────

group_colors   <- c("Non-NDD" = "#2ecc71", "NDD" = "#e74c3c")
group_colors_3 <- c("Non-Affected"       = "#2ecc71",
                    "Affected (non-ASD)" = "#3498db",
                    "ASD"                = "#e74c3c")
theme_pub <- theme_minimal(base_size = 12) +
  theme(strip.background = element_rect(fill = "grey92", color = NA),
        strip.text       = element_text(face = "bold"),
        legend.position  = "bottom",
        panel.grid.minor = element_blank())

save_dist_plot <- function(v, data) {
  d_v <- data[!is.na(data[[v]]), ]
  p1 <- ggplot(d_v, aes(x = .data[[v]])) +
    geom_histogram(aes(y = after_stat(density)), bins = 35,
                   fill = "steelblue", alpha = 0.7, color = "white") +
    geom_density(color = "navy", linewidth = 1) +
    labs(title = paste(v, "— Overall"), x = v, y = "Density") + theme_pub
  p2 <- ggplot(d_v, aes(sample = .data[[v]])) +
    stat_qq() + stat_qq_line(color = "red") +
    labs(title = "Q-Q (Overall)") + theme_pub
  p3 <- ggplot(d_v, aes(x = .data[[v]], fill = ndd)) +
    geom_density(alpha = 0.5) + scale_fill_manual(values = group_colors) +
    labs(title = "Density by NDD") + theme_pub
  p4 <- ggplot(d_v, aes(x = ndd, y = .data[[v]], fill = ndd)) +
    geom_violin(alpha = 0.4, draw_quantiles = c(0.25, 0.5, 0.75)) +
    geom_jitter(width = 0.15, alpha = 0.3, size = 0.8) +
    stat_summary(fun = mean, geom = "point", shape = 18, size = 3, color = "black") +
    scale_fill_manual(values = group_colors) +
    labs(title = "Boxviolin by NDD", x = "", y = v) + theme_pub +
    theme(legend.position = "none")
  combined <- (p1 + p2) / (p3 + p4) +
    plot_annotation(title = paste("Distribution Diagnostics:", v),
                    theme = theme(plot.title = element_text(face = "bold", size = 14)))
  ggsave(paste0("output/diagnostics/distribution_", v, ".png"),
         combined, width = 12, height = 9, dpi = 300, bg = "white")
  combined
}

save_dist_plot("facilitation_amp",  data)
save_dist_plot("disengagement_amp", data)
save_dist_plot("facilitation_lat",  data)
save_dist_plot("disengagement_lat", data)
save_dist_plot("facilitation_rt",   data)
save_dist_plot("disengagement_rt",  data)


# =============================================================================
# Q1. FACILITATION/DISENGAGEMENT EFFECTS IN RT (ONE-SAMPLE TESTS ONLY)
# =============================================================================

one_sample_t <- function(x, label) {
  x  <- x[!is.na(x)]
  tt <- t.test(x, mu = 0)
  data.frame(Label    = label,
             N        = length(x),
             Mean     = round(mean(x), 3),
             SD       = round(sd(x), 3),
             SE       = round(sd(x)/sqrt(length(x)), 3),
             t        = round(tt$statistic, 3),
             df       = round(tt$parameter, 1),
             p        = round(tt$p.value, 4),
             p_one    = round(tt$p.value / 2, 4),
             CI_lower = round(tt$conf.int[1], 3),
             CI_upper = round(tt$conf.int[2], 3),
             Cohen_d  = round(mean(x) / sd(x), 3))
}

# Full sample
os_full_facil_rt   <- one_sample_t(data$facilitation_rt,   "Full — facilitation_rt")
os_full_diseng_rt  <- one_sample_t(data$disengagement_rt,  "Full — disengagement_rt")

# By age bin
os_ch_facil_rt   <- one_sample_t(data$facilitation_rt  [data$age_bin == "Children\n(≤11)"],      "Children — facilitation_rt")
os_ad_facil_rt   <- one_sample_t(data$facilitation_rt  [data$age_bin == "Adolescents\n(12–17)"], "Adolescents — facilitation_rt")
os_au_facil_rt   <- one_sample_t(data$facilitation_rt  [data$age_bin == "Adults\n(18–44)"],      "Adults — facilitation_rt")
os_oa_facil_rt   <- one_sample_t(data$facilitation_rt  [data$age_bin == "Older Adults\n(45+)"],  "Older Adults — facilitation_rt")
os_ch_diseng_rt  <- one_sample_t(data$disengagement_rt [data$age_bin == "Children\n(≤11)"],      "Children — disengagement_rt")
os_ad_diseng_rt  <- one_sample_t(data$disengagement_rt [data$age_bin == "Adolescents\n(12–17)"], "Adolescents — disengagement_rt")
os_au_diseng_rt  <- one_sample_t(data$disengagement_rt [data$age_bin == "Adults\n(18–44)"],      "Adults — disengagement_rt")
os_oa_diseng_rt  <- one_sample_t(data$disengagement_rt [data$age_bin == "Older Adults\n(45+)"],  "Older Adults — disengagement_rt")

one_sample_df <- rbind(
  os_full_facil_rt,   os_full_diseng_rt,
  os_ch_facil_rt,     os_ad_facil_rt,     os_au_facil_rt,     os_oa_facil_rt,
  os_ch_diseng_rt,    os_ad_diseng_rt,    os_au_diseng_rt,    os_oa_diseng_rt
)
one_sample_df$p_bh   <- p.adjust(one_sample_df$p, method = "BH")
one_sample_df$sig_bh <- ifelse(one_sample_df$p_bh < .05, "*", "ns")
rownames(one_sample_df) <- NULL
print(one_sample_df)
write.csv(one_sample_df, "output/tables/Q1_RT_one_sample_tests.csv", row.names = FALSE)


# =============================================================================
# Q3. AGE TRAJECTORIES
# - Relative RT outcomes: linear models
# - Raw Gap/Overlap amplitude outcomes: GAMs with k selection (3, 5, 8)
# =============================================================================

get_icc <- function(outcome_name, default = 0) {
  idx <- which(icc_results$outcome == outcome_name)
  if (length(idx) == 0 || is.na(icc_results$ICC[idx][1])) return(default)
  as.numeric(icc_results$ICC[idx][1])
}

data_q3_facil_rt <- data %>% filter(!is.na(facilitation_rt), !is.na(age), !is.na(sex), !is.na(site))
data_q3_diseng_rt <- data %>% filter(!is.na(disengagement_rt), !is.na(age), !is.na(sex), !is.na(site))
data_q3_gap_amp <- data %>% filter(!is.na(Gap_amp), !is.na(age), !is.na(sex), !is.na(site))
data_q3_overlap_amp <- data %>% filter(!is.na(Overlap_amp), !is.na(age), !is.na(sex), !is.na(site))

cat("\n=== Q3 age trajectories: relative RT (linear models) ===\n")
lm_q3_facil_rt <- lm(facilitation_rt ~ age + sex + site, data = data_q3_facil_rt)
lm_q3_diseng_rt <- lm(disengagement_rt ~ age + sex + site, data = data_q3_diseng_rt)
cat("\n--- facilitation_rt ---\n"); print(summary(lm_q3_facil_rt))
cat("\n--- disengagement_rt ---\n"); print(summary(lm_q3_diseng_rt))

cat("\n=== Q3 age trajectories: raw amplitude GAMs (k = 3, 5, 8) ===\n")

q3_gap_amp_icc <- get_icc("Gap_amp")
if (q3_gap_amp_icc > 0.03) {
  gam_q3_gap_amp_k3 <- gam(Gap_amp ~ s(age, k = 3) + sex + site + s(fam_id, bs = "re"), data = data_q3_gap_amp, method = "REML")
  gam_q3_gap_amp_k5 <- gam(Gap_amp ~ s(age, k = 5) + sex + site + s(fam_id, bs = "re"), data = data_q3_gap_amp, method = "REML")
  gam_q3_gap_amp_k8 <- gam(Gap_amp ~ s(age, k = 8) + sex + site + s(fam_id, bs = "re"), data = data_q3_gap_amp, method = "REML")
} else {
  gam_q3_gap_amp_k3 <- gam(Gap_amp ~ s(age, k = 3) + sex + site, data = data_q3_gap_amp, method = "REML")
  gam_q3_gap_amp_k5 <- gam(Gap_amp ~ s(age, k = 5) + sex + site, data = data_q3_gap_amp, method = "REML")
  gam_q3_gap_amp_k8 <- gam(Gap_amp ~ s(age, k = 8) + sex + site, data = data_q3_gap_amp, method = "REML")
}

q3_overlap_amp_icc <- get_icc("Overlap_amp")
if (q3_overlap_amp_icc > 0.03) {
  gam_q3_overlap_amp_k3 <- gam(Overlap_amp ~ s(age, k = 3) + sex + site + s(fam_id, bs = "re"), data = data_q3_overlap_amp, method = "REML")
  gam_q3_overlap_amp_k5 <- gam(Overlap_amp ~ s(age, k = 5) + sex + site + s(fam_id, bs = "re"), data = data_q3_overlap_amp, method = "REML")
  gam_q3_overlap_amp_k8 <- gam(Overlap_amp ~ s(age, k = 8) + sex + site + s(fam_id, bs = "re"), data = data_q3_overlap_amp, method = "REML")
} else {
  gam_q3_overlap_amp_k3 <- gam(Overlap_amp ~ s(age, k = 3) + sex + site, data = data_q3_overlap_amp, method = "REML")
  gam_q3_overlap_amp_k5 <- gam(Overlap_amp ~ s(age, k = 5) + sex + site, data = data_q3_overlap_amp, method = "REML")
  gam_q3_overlap_amp_k8 <- gam(Overlap_amp ~ s(age, k = 8) + sex + site, data = data_q3_overlap_amp, method = "REML")
}

q3_gap_amp_aic <- data.frame(
  Outcome = "Gap_amp",
  k = c(3, 5, 8),
  AIC = c(AIC(gam_q3_gap_amp_k3), AIC(gam_q3_gap_amp_k5), AIC(gam_q3_gap_amp_k8)),
  DevExpl = c(summary(gam_q3_gap_amp_k3)$dev.expl, summary(gam_q3_gap_amp_k5)$dev.expl, summary(gam_q3_gap_amp_k8)$dev.expl)
)
q3_gap_amp_aic$delta_AIC <- q3_gap_amp_aic$AIC - min(q3_gap_amp_aic$AIC)
print(q3_gap_amp_aic)

q3_overlap_amp_aic <- data.frame(
  Outcome = "Overlap_amp",
  k = c(3, 5, 8),
  AIC = c(AIC(gam_q3_overlap_amp_k3), AIC(gam_q3_overlap_amp_k5), AIC(gam_q3_overlap_amp_k8)),
  DevExpl = c(summary(gam_q3_overlap_amp_k3)$dev.expl, summary(gam_q3_overlap_amp_k5)$dev.expl, summary(gam_q3_overlap_amp_k8)$dev.expl)
)
q3_overlap_amp_aic$delta_AIC <- q3_overlap_amp_aic$AIC - min(q3_overlap_amp_aic$AIC)
print(q3_overlap_amp_aic)

q3_gap_amp_models <- list(gam_q3_gap_amp_k3, gam_q3_gap_amp_k5, gam_q3_gap_amp_k8)
q3_overlap_amp_models <- list(gam_q3_overlap_amp_k3, gam_q3_overlap_amp_k5, gam_q3_overlap_amp_k8)
gam_q3_gap_amp_best <- q3_gap_amp_models[[which.min(q3_gap_amp_aic$AIC)]]
gam_q3_overlap_amp_best <- q3_overlap_amp_models[[which.min(q3_overlap_amp_aic$AIC)]]

cat("\n--- Best Gap_amp model summary ---\n"); print(summary(gam_q3_gap_amp_best))
cat("\n--- Best Overlap_amp model summary ---\n"); print(summary(gam_q3_overlap_amp_best))

get_age_smooth_p <- function(gam_model) {
  s_tab <- summary(gam_model)$s.table
  age_row <- grep("^s\\(age", rownames(s_tab))[1]
  as.numeric(s_tab[age_row, "p-value"])
}

q3_age_pvals <- c(
  summary(lm_q3_facil_rt)$coefficients["age", "Pr(>|t|)"],
  summary(lm_q3_diseng_rt)$coefficients["age", "Pr(>|t|)"],
  get_age_smooth_p(gam_q3_gap_amp_best),
  get_age_smooth_p(gam_q3_overlap_amp_best)
)

q3_age_results <- data.frame(
  Outcome = c("facilitation_rt", "disengagement_rt", "Gap_amp", "Overlap_amp"),
  p_age = round(q3_age_pvals, 6)
)
q3_age_results$p_age_BH <- round(p.adjust(q3_age_results$p_age, method = "BH"), 6)
q3_age_results$sig_BH <- ifelse(q3_age_results$p_age_BH < .05, "*", "ns")
print(q3_age_results)
write.csv(q3_age_results, "output/tables/Q3_age_trajectory_BH_FDR.csv", row.names = FALSE)
write.csv(rbind(q3_gap_amp_aic, q3_overlap_amp_aic), "output/tables/Q3_amp_k_selection.csv", row.names = FALSE)


# =============================================================================
# Q4. DIAGNOSTIC DIFFERENCES
# - Always include facilitation_rt and disengagement_rt
# - Include Gap_amp / Overlap_amp only if significant in Q3 BH-FDR
# - No latency models
# =============================================================================

q4_sig_amp <- q3_age_results$Outcome[q3_age_results$Outcome %in% c("Gap_amp", "Overlap_amp") & q3_age_results$p_age_BH < 0.05]
cat("\nQ4 amplitude outcomes carried forward from Q3 BH-FDR:\n")
print(q4_sig_amp)

q4_results <- data.frame(
  Outcome = character(),
  Test = character(),
  p = numeric(),
  stringsAsFactors = FALSE
)

# facilitation_rt: NDD and affected_group interaction comparisons
q4_data_facil_rt <- data %>% filter(!is.na(facilitation_rt), !is.na(age), !is.na(sex), !is.na(site), !is.na(ndd), !is.na(affected_group))
q4_facil_rt_base <- lm(facilitation_rt ~ age + sex + site, data = q4_data_facil_rt)
q4_facil_rt_ndd_main <- lm(facilitation_rt ~ age + sex + site + ndd, data = q4_data_facil_rt)
q4_facil_rt_ndd_int <- lm(facilitation_rt ~ age + sex + site + ndd * age, data = q4_data_facil_rt)
q4_facil_rt_grp_main <- lm(facilitation_rt ~ age + sex + site + affected_group, data = q4_data_facil_rt)
q4_facil_rt_grp_int <- lm(facilitation_rt ~ age + sex + site + affected_group * age, data = q4_data_facil_rt)

q4_results <- rbind(
  q4_results,
  data.frame(Outcome = "facilitation_rt", Test = "NDD_main", p = anova(q4_facil_rt_base, q4_facil_rt_ndd_main)$`Pr(>F)`[2]),
  data.frame(Outcome = "facilitation_rt", Test = "NDD_x_age", p = anova(q4_facil_rt_ndd_main, q4_facil_rt_ndd_int)$`Pr(>F)`[2]),
  data.frame(Outcome = "facilitation_rt", Test = "affected_group_main", p = anova(q4_facil_rt_base, q4_facil_rt_grp_main)$`Pr(>F)`[2]),
  data.frame(Outcome = "facilitation_rt", Test = "affected_group_x_age", p = anova(q4_facil_rt_grp_main, q4_facil_rt_grp_int)$`Pr(>F)`[2])
)

# disengagement_rt: NDD and affected_group interaction comparisons
q4_data_diseng_rt <- data %>% filter(!is.na(disengagement_rt), !is.na(age), !is.na(sex), !is.na(site), !is.na(ndd), !is.na(affected_group))
q4_diseng_rt_base <- lm(disengagement_rt ~ age + sex + site, data = q4_data_diseng_rt)
q4_diseng_rt_ndd_main <- lm(disengagement_rt ~ age + sex + site + ndd, data = q4_data_diseng_rt)
q4_diseng_rt_ndd_int <- lm(disengagement_rt ~ age + sex + site + ndd * age, data = q4_data_diseng_rt)
q4_diseng_rt_grp_main <- lm(disengagement_rt ~ age + sex + site + affected_group, data = q4_data_diseng_rt)
q4_diseng_rt_grp_int <- lm(disengagement_rt ~ age + sex + site + affected_group * age, data = q4_data_diseng_rt)

q4_results <- rbind(
  q4_results,
  data.frame(Outcome = "disengagement_rt", Test = "NDD_main", p = anova(q4_diseng_rt_base, q4_diseng_rt_ndd_main)$`Pr(>F)`[2]),
  data.frame(Outcome = "disengagement_rt", Test = "NDD_x_age", p = anova(q4_diseng_rt_ndd_main, q4_diseng_rt_ndd_int)$`Pr(>F)`[2]),
  data.frame(Outcome = "disengagement_rt", Test = "affected_group_main", p = anova(q4_diseng_rt_base, q4_diseng_rt_grp_main)$`Pr(>F)`[2]),
  data.frame(Outcome = "disengagement_rt", Test = "affected_group_x_age", p = anova(q4_diseng_rt_grp_main, q4_diseng_rt_grp_int)$`Pr(>F)`[2])
)

# Gap_amp if significant in Q3
if ("Gap_amp" %in% q4_sig_amp) {
  q4_data_gap_amp <- data %>% filter(!is.na(Gap_amp), !is.na(age), !is.na(sex), !is.na(site), !is.na(ndd), !is.na(affected_group))
  q4_gap_amp_base <- lm(Gap_amp ~ age + sex + site, data = q4_data_gap_amp)
  q4_gap_amp_ndd <- lm(Gap_amp ~ age + sex + site + ndd, data = q4_data_gap_amp)
  q4_gap_amp_group <- lm(Gap_amp ~ age + sex + site + affected_group, data = q4_data_gap_amp)
  q4_results <- rbind(
    q4_results,
    data.frame(Outcome = "Gap_amp", Test = "NDD_main", p = anova(q4_gap_amp_base, q4_gap_amp_ndd)$`Pr(>F)`[2]),
    data.frame(Outcome = "Gap_amp", Test = "affected_group_main", p = anova(q4_gap_amp_base, q4_gap_amp_group)$`Pr(>F)`[2])
  )
}

# Overlap_amp if significant in Q3
if ("Overlap_amp" %in% q4_sig_amp) {
  q4_data_overlap_amp <- data %>% filter(!is.na(Overlap_amp), !is.na(age), !is.na(sex), !is.na(site), !is.na(ndd), !is.na(affected_group))
  q4_overlap_amp_base <- lm(Overlap_amp ~ age + sex + site, data = q4_data_overlap_amp)
  q4_overlap_amp_ndd <- lm(Overlap_amp ~ age + sex + site + ndd, data = q4_data_overlap_amp)
  q4_overlap_amp_group <- lm(Overlap_amp ~ age + sex + site + affected_group, data = q4_data_overlap_amp)
  q4_results <- rbind(
    q4_results,
    data.frame(Outcome = "Overlap_amp", Test = "NDD_main", p = anova(q4_overlap_amp_base, q4_overlap_amp_ndd)$`Pr(>F)`[2]),
    data.frame(Outcome = "Overlap_amp", Test = "affected_group_main", p = anova(q4_overlap_amp_base, q4_overlap_amp_group)$`Pr(>F)`[2])
  )
}

q4_results$p_BH <- round(p.adjust(q4_results$p, method = "BH"), 6)
q4_results$sig_BH <- ifelse(q4_results$p_BH < .05, "*", "ns")
print(q4_results)
write.csv(q4_results, "output/tables/Q4_diagnostic_tests_BH_FDR.csv", row.names = FALSE)


# =============================================================================
# Q5. ERP AMPLITUDE → RT
# Matched and theoretically-linked relative RT models only (no latency)
# Single BH-FDR family across the four planned amplitude→RT tests
# =============================================================================

extract_model_p <- function(model_obj, term_name) {
  if (inherits(model_obj, "lmerModLmerTest") || inherits(model_obj, "lmerMod")) {
    return(summary(model_obj)$coefficients[term_name, "Pr(>|t|)"])
  }
  summary(model_obj)$coefficients[term_name, "Pr(>|t|)"]
}

fit_rt_model <- function(df, outcome, predictor) {
  icc_val <- get_icc(outcome)
  fml_lm <- as.formula(paste0(outcome, " ~ ", predictor, " + age + sex + site"))
  fml_lmer <- as.formula(paste0(outcome, " ~ ", predictor, " + age + sex + site + (1 | fam_id)"))
  if (icc_val > 0.03) {
    lmer(fml_lmer, data = df, REML = TRUE)
  } else {
    lm(fml_lm, data = df)
  }
}

q5_data_gap_match <- data %>% filter(!is.na(Gap_amp), !is.na(Gap_rt), !is.na(age), !is.na(sex), !is.na(site))
q5_data_overlap_match <- data %>% filter(!is.na(Overlap_amp), !is.na(Overlap_rt), !is.na(age), !is.na(sex), !is.na(site))
q5_data_gap_rel <- data %>% filter(!is.na(Gap_amp), !is.na(facilitation_rt), !is.na(age), !is.na(sex), !is.na(site))
q5_data_overlap_rel <- data %>% filter(!is.na(Overlap_amp), !is.na(disengagement_rt), !is.na(age), !is.na(sex), !is.na(site))

q5_model_gap_match <- fit_rt_model(q5_data_gap_match, "Gap_rt", "Gap_amp")
q5_model_overlap_match <- fit_rt_model(q5_data_overlap_match, "Overlap_rt", "Overlap_amp")
q5_model_gap_rel <- fit_rt_model(q5_data_gap_rel, "facilitation_rt", "Gap_amp")
q5_model_overlap_rel <- fit_rt_model(q5_data_overlap_rel, "disengagement_rt", "Overlap_amp")

q5_results <- data.frame(
  Pairing = c(
    "Gap_amp -> Gap_rt",
    "Overlap_amp -> Overlap_rt",
    "Gap_amp -> facilitation_rt",
    "Overlap_amp -> disengagement_rt"
  ),
  N = c(nrow(q5_data_gap_match), nrow(q5_data_overlap_match), nrow(q5_data_gap_rel), nrow(q5_data_overlap_rel)),
  p = c(
    extract_model_p(q5_model_gap_match, "Gap_amp"),
    extract_model_p(q5_model_overlap_match, "Overlap_amp"),
    extract_model_p(q5_model_gap_rel, "Gap_amp"),
    extract_model_p(q5_model_overlap_rel, "Overlap_amp")
  )
)
q5_results$p_BH <- round(p.adjust(q5_results$p, method = "BH"), 6)
q5_results$sig_BH <- ifelse(q5_results$p_BH < .05, "*", "ns")

print(q5_results)
write.csv(q5_results, "output/tables/Q5_ERP_amp_to_RT_BH_FDR.csv", row.names = FALSE)
