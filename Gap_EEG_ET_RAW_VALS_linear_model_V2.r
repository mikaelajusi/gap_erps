1+1
# =============================================================================
# GAP-OVERLAP TASK ANALYSIS — Full Script
# Q1. Are  Gap and Overlap effects present, and how do they
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

data$Gap_rt_check  <- data$Baseline_rt - data$Gap_rt
data$Overlap_rt_check <- data$Overlap_rt  - data$Baseline_rt

rt_discrepancy <- data %>%
  filter(!is.na(Gap_rt) & !is.na(Gap_rt_check)) %>%
  summarise(
    max_facil_diff  = max(abs(Gap_rt  - Gap_rt_check),  na.rm = TRUE),
    max_diseng_diff = max(abs(Overlap_rt - Overlap_rt_check), na.rm = TRUE)
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

data$Gap_amp  <- as.numeric(data$Gap_amp)
data$Overlap_amp <- as.numeric(data$Overlap_amp)
data$Gap_lat  <- as.numeric(data$Gap_lat)
data$Overlap_lat <- as.numeric(data$Overlap_lat)
data$Gap_rt   <- as.numeric(data$Gap_rt)
data$Overlap_rt  <- as.numeric(data$Overlap_rt)

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

all_outcomes    <- c("Gap_amp", "Overlap_amp",
                     "Gap_lat", "Overlap_lat",
                     "Gap_rt",  "Overlap_rt")

missing_report <- data.frame(
  Variable    = all_outcomes,
  N_total     = nrow(data),
  N_observed  = sapply(all_outcomes, function(v) sum(!is.na(data[[v]]))),
  N_missing   = sapply(all_outcomes, function(v) sum( is.na(data[[v]]))),
  Pct_missing = sapply(all_outcomes, function(v) round(100 * mean(is.na(data[[v]])), 2))
)
print(missing_report)
write.csv(missing_report, "output/tables/missing_data_report.csv", row.names = FALSE)

cat("\n--- Missing data characterization: Gap_rt ---\n")
data$rt_missing_flag <- is.na(data$Gap_rt)
t_facil_age  <- t.test(age ~ rt_missing_flag, data = data)
chi_facil_ndd <- chisq.test(table(data$ndd, data$rt_missing_flag))
chi_facil_sex <- chisq.test(table(data$sex, data$rt_missing_flag))
cat("Age: missers M =", round(mean(data$age[data$rt_missing_flag],  na.rm = TRUE), 1),
    "vs observed M =", round(mean(data$age[!data$rt_missing_flag], na.rm = TRUE), 1),
    "| t =", round(t_facil_age$statistic, 2), ", p =", round(t_facil_age$p.value, 4), "\n")
cat("NDD × missing χ²:", round(chi_facil_ndd$statistic, 2), ", p =", round(chi_facil_ndd$p.value, 4), "\n")
cat("Sex × missing χ²:", round(chi_facil_sex$statistic, 2), ", p =", round(chi_facil_sex$p.value, 4), "\n")
data$rt_missing_flag <- NULL

cat("\n--- Missing data characterization: Overlap_rt ---\n")
data$rt_missing_flag <- is.na(data$Overlap_rt)
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

out_facil_amp  <- detect_outliers_3sd(data$Gap_amp)
out_diseng_amp <- detect_outliers_3sd(data$Overlap_amp)
out_facil_lat  <- detect_outliers_3sd(data$Gap_lat)
out_diseng_lat <- detect_outliers_3sd(data$Overlap_lat)
out_facil_rt   <- detect_outliers_3sd(data$Gap_rt)
out_diseng_rt  <- detect_outliers_3sd(data$Overlap_rt)

cat("Outliers (3 SD) — Gap_amp:",  sum(out_facil_amp),  "\n")
cat("Outliers (3 SD) — Overlap_amp:", sum(out_diseng_amp), "\n")
cat("Outliers (3 SD) — Gap_lat:",  sum(out_facil_lat),  "\n")
cat("Outliers (3 SD) — Overlap_lat:", sum(out_diseng_lat), "\n")
cat("Outliers (3 SD) — Gap_rt:",   sum(out_facil_rt),   "\n")
cat("Outliers (3 SD) — Overlap_rt:",  sum(out_diseng_rt),  "\n")

#data$Gap_amp [out_facil_amp]  <- NA
#data$Overlap_amp[out_diseng_amp] <- NA
#data$Gap_lat [out_facil_lat]  <- NA
#data$Overlap_lat[out_diseng_lat] <- NA
#data$Gap_rt  [out_facil_rt]   <- NA
#data$Overlap_rt [out_diseng_rt]  <- NA

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
            facil_amp_M  = round(mean(Gap_amp,  na.rm = TRUE), 2),
            facil_amp_SD = round(sd(Gap_amp,    na.rm = TRUE), 2),
            diseng_amp_M  = round(mean(Overlap_amp, na.rm = TRUE), 2),
            diseng_amp_SD = round(sd(Overlap_amp,   na.rm = TRUE), 2),
            facil_lat_M  = round(mean(Gap_lat,  na.rm = TRUE), 2),
            facil_lat_SD = round(sd(Gap_lat,    na.rm = TRUE), 2),
            diseng_lat_M  = round(mean(Overlap_lat, na.rm = TRUE), 2),
            diseng_lat_SD = round(sd(Overlap_lat,   na.rm = TRUE), 2),
            facil_rt_M  = round(mean(Gap_rt,  na.rm = TRUE), 2),
            facil_rt_SD = round(sd(Gap_rt,    na.rm = TRUE), 2),
            diseng_rt_M  = round(mean(Overlap_rt, na.rm = TRUE), 2),
            diseng_rt_SD = round(sd(Overlap_rt,   na.rm = TRUE), 2),
            .groups = "drop")
print(desc_by_ndd)
write.csv(desc_by_ndd, "output/tables/descriptives_by_NDD.csv", row.names = FALSE)

desc_by_group <- data %>%
  group_by(affected_group) %>%
  summarise(N          = n(),
            Age_M      = round(mean(age, na.rm = TRUE), 2),
            Age_SD     = round(sd(age,   na.rm = TRUE), 2),
            Pct_Female = round(100 * mean(sex == "Female", na.rm = TRUE), 1),
            facil_amp_M  = round(mean(Gap_amp,  na.rm = TRUE), 2),
            facil_amp_SD = round(sd(Gap_amp,    na.rm = TRUE), 2),
            diseng_amp_M  = round(mean(Overlap_amp, na.rm = TRUE), 2),
            diseng_amp_SD = round(sd(Overlap_amp,   na.rm = TRUE), 2),
            facil_lat_M  = round(mean(Gap_lat,  na.rm = TRUE), 2),
            facil_lat_SD = round(sd(Gap_lat,    na.rm = TRUE), 2),
            diseng_lat_M  = round(mean(Overlap_lat, na.rm = TRUE), 2),
            diseng_lat_SD = round(sd(Overlap_lat,   na.rm = TRUE), 2),
            facil_rt_M  = round(mean(Gap_rt,  na.rm = TRUE), 2),
            facil_rt_SD = round(sd(Gap_rt,    na.rm = TRUE), 2),
            diseng_rt_M  = round(mean(Overlap_rt, na.rm = TRUE), 2),
            diseng_rt_SD = round(sd(Overlap_rt,   na.rm = TRUE), 2),
            .groups = "drop")
print(desc_by_group)
write.csv(desc_by_group, "output/tables/descriptives_by_affected_group.csv", row.names = FALSE)

desc_by_agebin <- data %>%
  group_by(age_bin) %>%
  summarise(N      = n(),
            Age_M  = round(mean(age, na.rm = TRUE), 2),
            Age_SD = round(sd(age,   na.rm = TRUE), 2),
            facil_amp_M    = round(mean(Gap_amp,  na.rm = TRUE), 2),
            facil_amp_SD   = round(sd(Gap_amp,    na.rm = TRUE), 2),
            facil_amp_Nobs = sum(!is.na(Gap_amp)),
            diseng_amp_M    = round(mean(Overlap_amp, na.rm = TRUE), 2),
            diseng_amp_SD   = round(sd(Overlap_amp,   na.rm = TRUE), 2),
            diseng_amp_Nobs = sum(!is.na(Overlap_amp)),
            facil_lat_M    = round(mean(Gap_lat,  na.rm = TRUE), 2),
            facil_lat_SD   = round(sd(Gap_lat,    na.rm = TRUE), 2),
            facil_lat_Nobs = sum(!is.na(Gap_lat)),
            diseng_lat_M    = round(mean(Overlap_lat, na.rm = TRUE), 2),
            diseng_lat_SD   = round(sd(Overlap_lat,   na.rm = TRUE), 2),
            diseng_lat_Nobs = sum(!is.na(Overlap_lat)),
            facil_rt_M    = round(mean(Gap_rt,  na.rm = TRUE), 2),
            facil_rt_SD   = round(sd(Gap_rt,    na.rm = TRUE), 2),
            facil_rt_Nobs = sum(!is.na(Gap_rt)),
            diseng_rt_M    = round(mean(Overlap_rt, na.rm = TRUE), 2),
            diseng_rt_SD   = round(sd(Overlap_rt,   na.rm = TRUE), 2),
            diseng_rt_Nobs = sum(!is.na(Overlap_rt)),
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
  "Gap_amp", "Overlap_amp",
  "Gap_lat", "Overlap_lat",
  "Gap_rt",  "Overlap_rt"
)

icc_results <- dplyr::bind_rows(lapply(outcomes, function(v) fit_null_icc(data, v, "fam_id")))
icc_results$Pct_var <- round(100 * icc_results$ICC, 2)

print(icc_results)
write.csv(icc_results, "output/tables/icc_results.csv", row.names = FALSE)


#Family-level clustering was essentially absent for all Gap outcomes 
#ICCs and retained only for Overlap 



# ── 7. ASSUMPTION CHECKS ─────────────────────────────────────────────────────

sw_facil_amp  <- shapiro.test(data$Gap_amp [!is.na(data$Gap_amp)])
sw_diseng_amp <- shapiro.test(data$Overlap_amp[!is.na(data$Overlap_amp)])
sw_facil_lat  <- shapiro.test(data$Gap_lat [!is.na(data$Gap_lat)])
sw_diseng_lat <- shapiro.test(data$Overlap_lat[!is.na(data$Overlap_lat)])
sw_facil_rt   <- shapiro.test(data$Gap_rt  [!is.na(data$Gap_rt)])
sw_diseng_rt  <- shapiro.test(data$Overlap_rt [!is.na(data$Overlap_rt)])

assumption_df <- data.frame(
  Outcome   = all_outcomes,
  N         = c(sum(!is.na(data$Gap_amp)),  sum(!is.na(data$Overlap_amp)),
                sum(!is.na(data$Gap_lat)),  sum(!is.na(data$Overlap_lat)),
                sum(!is.na(data$Gap_rt)),   sum(!is.na(data$Overlap_rt))),
  Skewness  = round(c(moments::skewness(data$Gap_amp,  na.rm = TRUE),
                      moments::skewness(data$Overlap_amp, na.rm = TRUE),
                      moments::skewness(data$Gap_lat,  na.rm = TRUE),
                      moments::skewness(data$Overlap_lat, na.rm = TRUE),
                      moments::skewness(data$Gap_rt,   na.rm = TRUE),
                      moments::skewness(data$Overlap_rt,  na.rm = TRUE)), 3),
  Kurtosis  = round(c(moments::kurtosis(data$Gap_amp,  na.rm = TRUE),
                      moments::kurtosis(data$Overlap_amp, na.rm = TRUE),
                      moments::kurtosis(data$Gap_lat,  na.rm = TRUE),
                      moments::kurtosis(data$Overlap_lat, na.rm = TRUE),
                      moments::kurtosis(data$Gap_rt,   na.rm = TRUE),
                      moments::kurtosis(data$Overlap_rt,  na.rm = TRUE)), 3),
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

save_dist_plot("Gap_amp",  data)
save_dist_plot("Overlap_amp", data)
save_dist_plot("Gap_lat",  data)
save_dist_plot("Overlap_lat", data)
save_dist_plot("Gap_rt",   data)
save_dist_plot("Overlap_rt",  data)


# =============================================================================
# Q1b. LINEAR MODELS: AGE TRAJECTORIES
 #Gap_lat retains (1|fam_id) via lmer (ICC = 7.96%); all others lm()
# =============================================================================

# =============================================================================
# Q1b. GAM: AGE TRAJECTORIES
# =============================================================================

data_facil_amp  <- data[!is.na(data$Gap_amp)  & !is.na(data$ndd), ]
data_diseng_amp <- data[!is.na(data$Overlap_amp) & !is.na(data$ndd), ]
data_facil_lat  <- data[!is.na(data$Gap_lat)  & !is.na(data$ndd), ]
data_diseng_lat <- data[!is.na(data$Overlap_lat) & !is.na(data$ndd), ]
data_facil_rt   <- data[!is.na(data$Gap_rt)   & !is.na(data$ndd), ]
data_diseng_rt  <- data[!is.na(data$Overlap_rt)  & !is.na(data$ndd), ]

# Gap_amp
gam1b_facil_amp_k3 <- gam(Gap_amp ~ s(age, k=3) + site + sex), data = data_facil_amp,  method = "REML")
gam1b_facil_amp_k5 <- gam(Gap_amp ~ s(age, k=5) + site + sex  , data = data_facil_amp,  method = "REML")
gam1b_facil_amp_k8 <- gam(Gap_amp ~ s(age, k=8) + site + sex , data = data_facil_amp,  method = "REML")

aic_facil_amp <- data.frame(
  k = c(3, 5, 8),
  AIC     = c(AIC(gam1b_facil_amp_k3), AIC(gam1b_facil_amp_k5), AIC(gam1b_facil_amp_k8)),
  edf     = c(sum(gam1b_facil_amp_k3$edf), sum(gam1b_facil_amp_k5$edf), sum(gam1b_facil_amp_k8$edf)),
  devExp  = c(summary(gam1b_facil_amp_k3)$dev.expl, summary(gam1b_facil_amp_k5)$dev.expl, summary(gam1b_facil_amp_k8)$dev.expl)
)
aic_facil_amp$delta_AIC <- aic_facil_amp$AIC - min(aic_facil_amp$AIC)
cat("\n--- Q1b Gap_amp k selection ---\n"); print(aic_facil_amp)

# Overlap_amp
gam1b_diseng_amp_k3 <- gam(Overlap_amp ~ s(age, k=3) + site + sex + s(fam_id, bs="re"), data = data_diseng_amp, method = "REML")
gam1b_diseng_amp_k5 <- gam(Overlap_amp ~ s(age, k=5) + site + sex + s(fam_id, bs="re"), data = data_diseng_amp, method = "REML")
gam1b_diseng_amp_k8 <- gam(Overlap_amp ~ s(age, k=8) + site + sex + s(fam_id, bs="re"), data = data_diseng_amp, method = "REML")

aic_diseng_amp <- data.frame(
  k = c(3, 5, 8),
  AIC     = c(AIC(gam1b_diseng_amp_k3), AIC(gam1b_diseng_amp_k5), AIC(gam1b_diseng_amp_k8)),
  edf     = c(sum(gam1b_diseng_amp_k3$edf), sum(gam1b_diseng_amp_k5$edf), sum(gam1b_diseng_amp_k8$edf)),
  devExp  = c(summary(gam1b_diseng_amp_k3)$dev.expl, summary(gam1b_diseng_amp_k5)$dev.expl, summary(gam1b_diseng_amp_k8)$dev.expl)
)
aic_diseng_amp$delta_AIC <- aic_diseng_amp$AIC - min(aic_diseng_amp$AIC)
cat("\n--- Q1b Overlap_amp k selection ---\n"); print(aic_diseng_amp)

# Gap_lat
gam1b_facil_lat_k3 <- gam(Gap_lat ~ s(age, k=3) + site + sex  , data = data_facil_lat,  method = "REML")
gam1b_facil_lat_k5 <- gam(Gap_lat ~ s(age, k=5) + site + sex  , data = data_facil_lat,  method = "REML")
gam1b_facil_lat_k8 <- gam(Gap_lat ~ s(age, k=8) + site + sex  , data = data_facil_lat,  method = "REML")

aic_facil_lat <- data.frame(
  k = c(3, 5, 8),
  AIC     = c(AIC(gam1b_facil_lat_k3), AIC(gam1b_facil_lat_k5), AIC(gam1b_facil_lat_k8)),
  edf     = c(sum(gam1b_facil_lat_k3$edf), sum(gam1b_facil_lat_k5$edf), sum(gam1b_facil_lat_k8$edf)),
  devExp  = c(summary(gam1b_facil_lat_k3)$dev.expl, summary(gam1b_facil_lat_k5)$dev.expl, summary(gam1b_facil_lat_k8)$dev.expl)
)
aic_facil_lat$delta_AIC <- aic_facil_lat$AIC - min(aic_facil_lat$AIC)
cat("\n--- Q1b Gap_lat k selection ---\n"); print(aic_facil_lat)

# Overlap_lat
gam1b_diseng_lat_k3 <- gam(Overlap_lat ~ s(age, k=3) + site + sex + s(fam_id, bs="re"), data = data_diseng_lat, method = "REML")
gam1b_diseng_lat_k5 <- gam(Overlap_lat ~ s(age, k=5) + site + sex + s(fam_id, bs="re"), data = data_diseng_lat, method = "REML")
gam1b_diseng_lat_k8 <- gam(Overlap_lat ~ s(age, k=8) + site + sex + s(fam_id, bs="re"), data = data_diseng_lat, method = "REML")

aic_diseng_lat <- data.frame(
  k = c(3, 5, 8),
  AIC     = c(AIC(gam1b_diseng_lat_k3), AIC(gam1b_diseng_lat_k5), AIC(gam1b_diseng_lat_k8)),
  edf     = c(sum(gam1b_diseng_lat_k3$edf), sum(gam1b_diseng_lat_k5$edf), sum(gam1b_diseng_lat_k8$edf)),
  devExp  = c(summary(gam1b_diseng_lat_k3)$dev.expl, summary(gam1b_diseng_lat_k5)$dev.expl, summary(gam1b_diseng_lat_k8)$dev.expl)
)
aic_diseng_lat$delta_AIC <- aic_diseng_lat$AIC - min(aic_diseng_lat$AIC)
cat("\n--- Q1b Overlap_lat k selection ---\n"); print(aic_diseng_lat)

# Gap_rt
gam1b_facil_rt_k3 <- gam(Gap_rt ~ s(age, k=3) + site + sex  , data = data_facil_rt,   method = "REML")
gam1b_facil_rt_k5 <- gam(Gap_rt ~ s(age, k=5) + site + sex  , data = data_facil_rt,   method = "REML")
gam1b_facil_rt_k8 <- gam(Gap_rt ~ s(age, k=8) + site + sex  , data = data_facil_rt,   method = "REML")

aic_facil_rt <- data.frame(
  k = c(3, 5, 8),
  AIC     = c(AIC(gam1b_facil_rt_k3), AIC(gam1b_facil_rt_k5), AIC(gam1b_facil_rt_k8)),
  edf     = c(sum(gam1b_facil_rt_k3$edf), sum(gam1b_facil_rt_k5$edf), sum(gam1b_facil_rt_k8$edf)),
  devExp  = c(summary(gam1b_facil_rt_k3)$dev.expl, summary(gam1b_facil_rt_k5)$dev.expl, summary(gam1b_facil_rt_k8)$dev.expl)
)
aic_facil_rt$delta_AIC <- aic_facil_rt$AIC - min(aic_facil_rt$AIC)
cat("\n--- Q1b Gap_rt k selection ---\n"); print(aic_facil_rt)

# Overlap_rt
gam1b_diseng_rt_k3 <- gam(Overlap_rt ~ s(age, k=3) + site + s(fam_id, bs="re"), data = data_diseng_rt,  method = "REML")
gam1b_diseng_rt_k5 <- gam(Overlap_rt ~ s(age, k=5) + site + s(fam_id, bs="re"), data = data_diseng_rt,  method = "REML")
gam1b_diseng_rt_k8 <- gam(Overlap_rt ~ s(age, k=8) + site + s(fam_id, bs="re"), data = data_diseng_rt,  method = "REML")

aic_diseng_rt <- data.frame(
  k = c(3, 5, 8),
  AIC     = c(AIC(gam1b_diseng_rt_k3), AIC(gam1b_diseng_rt_k5), AIC(gam1b_diseng_rt_k8)),
  edf     = c(sum(gam1b_diseng_rt_k3$edf), sum(gam1b_diseng_rt_k5$edf), sum(gam1b_diseng_rt_k8$edf)),
  devExp  = c(summary(gam1b_diseng_rt_k3)$dev.expl, summary(gam1b_diseng_rt_k5)$dev.expl, summary(gam1b_diseng_rt_k8)$dev.expl)
)
aic_diseng_rt$delta_AIC <- aic_diseng_rt$AIC - min(aic_diseng_rt$AIC)
cat("\n--- Q1b Overlap_rt k selection ---\n"); print(aic_diseng_rt)

# Print summaries of all best-k models side by side
cat("\n=== Q1b Summaries — best-k models ===\n")
cat("\n--- Gap_amp ---\n");  print(summary(gam1b_diseng_amp_k8))
cat("\n--- Overlap_amp ---\n"); print(summary(gam1b_diseng_amp_k5))
cat("\n--- Gap_lat ---\n");  print(summary(gam1b_facil_lat_k5))
cat("\n--- Overlap_lat ---\n"); print(summary(gam1b_diseng_lat_k8))
cat("\n--- Gap_rt ---\n");   print(summary(gam1b_facil_rt_k5))
cat("\n--- Overlap_rt ---\n");  print(summary(gam1b_diseng_rt_k5))

# Trajectory plot


# Q1b trajectory plots (one per outcome, using k=5 as default; swap if AIC favours another)
make_traj_plot <- function(model, data_v, outcome_name) {
  pred_grid <- data.frame(
    age = seq(min(data_v$age, na.rm = TRUE), max(data_v$age, na.rm = TRUE), length.out = 200))
  preds <- predict_marginal(model, pred_grid, levels(data_v$site), levels(data_v$fam_id))
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
    labs(title    = paste("Q1b: Age Trajectory —", outcome_name),
         subtitle = "GAM smooth ± 95% CI (marginal over site, family RE excluded)",
         x = "Age (years)", y = outcome_name) + theme_pub
}

# Raw trajectory plot

make_age_trajectories_raw <- function(data_v, outcome_name, color_by = NULL) {
  p <- ggplot(data_v, aes(x = age, y = .data[[outcome_name]])) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50")
  
  if (!is.null(color_by)) {
    p <- p +
      geom_point(aes(color = .data[[color_by]]), alpha = 0.4, size = 1.5) +
      geom_smooth(aes(color = .data[[color_by]]),
                  method = "loess", se = FALSE, linewidth = 0.8) +
      scale_color_viridis_d(name = color_by)
  } else {
    p <- p +
      geom_point(alpha = 0.35, size = 1.5, color = "grey40") +
      geom_smooth(method = "loess", se = TRUE,
                  fill = "tomato", color = "tomato3", linewidth = 1.1)
  }
  
  p + labs(
    title    = paste("Raw Age Trajectory —", outcome_name),
    subtitle = "LOESS smoother; no model",
    x = "Age (years)", y = outcome_name
  ) + theme_pub
}

# Model trajectory plots
p1b_facil_amp  <- make_traj_plot(lm1b_facil_amp,  data_facil_amp,  "Gap_amp")
p1b_diseng_amp <- make_traj_plot(lm1b_diseng_amp, data_diseng_amp, "Overlap_amp")
p1b_facil_lat  <- make_traj_plot(lm1b_facil_lat,  data_facil_lat,  "Gap_lat")
p1b_diseng_lat <- make_traj_plot(lm1b_diseng_lat, data_diseng_lat, "Overlap_lat")
p1b_facil_rt   <- make_traj_plot(lm1b_facil_rt,   data_facil_rt,   "Gap_rt")
p1b_diseng_rt  <- make_traj_plot(lm1b_diseng_rt,  data_diseng_rt,  "Overlap_rt")

print(p1b_facil_amp);  ggsave("output/figures/Q1b_trajectory_Gap_amp.png",  p1b_facil_amp,  width=9, height=6, dpi=300, bg="white")
print(p1b_diseng_amp); ggsave("output/figures/Q1b_trajectory_Overlap_amp.png", p1b_diseng_amp, width=9, height=6, dpi=300, bg="white")
print(p1b_facil_lat);  ggsave("output/figures/Q1b_trajectory_Gap_lat.png",  p1b_facil_lat,  width=9, height=6, dpi=300, bg="white")
print(p1b_diseng_lat); ggsave("output/figures/Q1b_trajectory_Overlap_lat.png", p1b_diseng_lat, width=9, height=6, dpi=300, bg="white")
print(p1b_facil_rt);   ggsave("output/figures/Q1b_trajectory_Gap_rt.png",   p1b_facil_rt,   width=9, height=6, dpi=300, bg="white")
print(p1b_diseng_rt);  ggsave("output/figures/Q1b_trajectory_Overlap_rt.png",  p1b_diseng_rt,  width=9, height=6, dpi=300, bg="white")

# --------------------------------------------------------------------------
# Raw LOESS trajectory plots
# --------------------------------------------------------------------------
p1b_facil_amp_raw  <- make_age_trajectories_raw(data_facil_amp,  "Gap_amp")
p1b_diseng_amp_raw <- make_age_trajectories_raw(data_diseng_amp, "Overlap_amp")
p1b_facil_lat_raw  <- make_age_trajectories_raw(data_facil_lat,  "Gap_lat")
p1b_diseng_lat_raw <- make_age_trajectories_raw(data_diseng_lat, "Overlap_lat")
p1b_facil_rt_raw   <- make_age_trajectories_raw(data_facil_rt,   "Gap_rt")
p1b_diseng_rt_raw  <- make_age_trajectories_raw(data_diseng_rt,  "Overlap_rt")

print(p1b_facil_amp_raw);  ggsave("output/figures/Q1b_trajectory_Gap_amp_raw.png",  p1b_facil_amp_raw,  width=9, height=6, dpi=300, bg="white")
print(p1b_diseng_amp_raw); ggsave("output/figures/Q1b_trajectory_Overlap_amp_raw.png", p1b_diseng_amp_raw, width=9, height=6, dpi=300, bg="white")
print(p1b_facil_lat_raw);  ggsave("output/figures/Q1b_trajectory_Gap_lat_raw.png",  p1b_facil_lat_raw,  width=9, height=6, dpi=300, bg="white")
print(p1b_diseng_lat_raw); ggsave("output/figures/Q1b_trajectory_Overlap_lat_raw.png", p1b_diseng_lat_raw, width=9, height=6, dpi=300, bg="white")
print(p1b_facil_rt_raw);   ggsave("output/figures/Q1b_trajectory_Gap_rt_raw.png",   p1b_facil_rt_raw,   width=9, height=6, dpi=300, bg="white")
print(p1b_diseng_rt_raw);  ggsave("output/figures/Q1b_trajectory_Overlap_rt_raw.png",  p1b_diseng_rt_raw,  width=9, height=6, dpi=300, bg="white")

# BH CORRECTIONS 
q1b_age_pvals <- c(
  summary(lm1b_facil_amp)$coefficients["age", "Pr(>|t|)"],
  summary(lm1b_diseng_amp)$coefficients["age", "Pr(>|t|)"],
  summary(lm1b_facil_lat)$coefficients["age", "Pr(>|t|)"],   # from lmerTest
  summary(lm1b_diseng_lat)$coefficients["age", "Pr(>|t|)"],
  summary(lm1b_facil_rt)$coefficients["age", "Pr(>|t|)"],
  summary(lm1b_diseng_rt)$coefficients["age", "Pr(>|t|)"]
)

q1b_fdr <- data.frame(
  Outcome   = c("Gap_amp", "Overlap_amp", "Gap_lat",
                "Overlap_lat", "Gap_rt",  "Overlap_rt"),
  p_age     = round(q1b_age_pvals, 4),
  p_age_FDR = round(p.adjust(q1b_age_pvals, method = "BH"), 4),
  sig_FDR   = ifelse(p.adjust(q1b_age_pvals, method = "BH") < .05, "*", "ns")
)
cat("\n=== Q1b BH-FDR corrected age effects ===\n")
print(q1b_fdr)
write.csv(q1b_fdr, "output/tables/Q1b_age_BH_FDR.csv", row.names = FALSE)




# =============================================================================
# Q1c. AGE BIN WELCH ANOVA
# =============================================================================
pairs_to_signif <- function(pairs_obj, bin_levels, p_thresh = 0.05) {
  df <- as.data.frame(pairs_obj)
  df$group1 <- trimws(sub("(?s) - .*", "", df$contrast, perl = TRUE))
  df$group2 <- trimws(sub("(?s).* - ", "", df$contrast, perl = TRUE))
  df <- df[df$p.value < p_thresh, ]
  if (nrow(df) == 0) return(NULL)
  
  # Normalise whitespace/newlines AND strip emmeans outer parens (e.g. "(Older Adults (45+))")
  clean <- function(x) {
    x <- gsub("\\s+", " ", trimws(x))
    x <- gsub("^\\((.*)\\)$", "\\1", x)
    x
  }
  clean_levels <- clean(bin_levels)
  
  df$pos1 <- match(clean(df$group1), clean_levels)
  df$pos2 <- match(clean(df$group2), clean_levels)
  
  # Drop any pair where a group wasn't found
  df <- df[!is.na(df$pos1) & !is.na(df$pos2), ]
  if (nrow(df) == 0) return(NULL)
  
  df$sig_label <- ifelse(df$p.value < 0.001, "***",
                         ifelse(df$p.value < 0.01,  "**",
                                ifelse(df$p.value < 0.05,  "*", "")))
  df
}
make_agebin_plot <- function(data_v, outcome_name, pairs_obj = NULL) {
  summ_v <- data_v %>%
    group_by(age_bin) %>%
    summarise(M  = mean(.data[[outcome_name]], na.rm = TRUE),
              SE = sd(.data[[outcome_name]], na.rm = TRUE) / sqrt(n()),
              N  = n(), .groups = "drop")
  
  bin_levels <- levels(data_v$age_bin)
  y_max      <- max(summ_v$M + 1.96 * summ_v$SE, na.rm = TRUE)
  y_min      <- min(summ_v$M - 1.96 * summ_v$SE, na.rm = TRUE)
  y_range    <- y_max - y_min
  step       <- y_range * 0.14
  
  # Clean outcome label for title (strip underscores)
  outcome_label <- gsub("_", " ", outcome_name)
  outcome_label <- tools::toTitleCase(outcome_label)
  
  p <- ggplot(summ_v, aes(x = age_bin, y = M, group = 1)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey60") +
    geom_ribbon(aes(ymin = M - 1.96*SE, ymax = M + 1.96*SE),
                fill = "steelblue", alpha = 0.15) +
    geom_line(color = "steelblue", linewidth = 1.2) +
    geom_point(color = "steelblue", size = 3.5, shape = 19) +
    geom_errorbar(aes(ymin = M - 1.96*SE, ymax = M + 1.96*SE),
                  width = 0.12, color = "steelblue4", linewidth = 0.8) +
    geom_text(aes(label = paste0("n=", N), y = M + 1.96*SE + y_range * 0.05),
              size = 3, color = "grey40") +
    labs(
      title    = outcome_label,
      subtitle = "Mean ± 95% CI; Tukey post-hoc: * p<.05  ** p<.01  *** p<.001",
      x        = "Age Group",
      y        = outcome_label
    ) +
    theme_bw(base_size = 12) +
    theme(
      panel.grid.major   = element_blank(),
      panel.grid.minor   = element_blank(),
      panel.border       = element_rect(color = "grey30", linewidth = 0.6),
      axis.text.x        = element_text(angle = 30, hjust = 1, size = 10),
      plot.title         = element_text(face = "bold", size = 13),
      plot.subtitle      = element_text(size = 9, color = "grey40")
    )
  
  if (!is.null(pairs_obj)) {
    sig_df <- pairs_to_signif(pairs_obj, bin_levels, p_thresh = 0.05)
    if (!is.null(sig_df) && nrow(sig_df) > 0) {
      sig_df <- sig_df[order(abs(sig_df$pos2 - sig_df$pos1)), ]
      sig_df$bracket_y <- y_max + step * seq_len(nrow(sig_df))
      
      # ggsignif on discrete axis requires numeric positions
      p <- p +
        ggsignif::geom_signif(
          xmin        = sig_df$pos1,
          xmax        = sig_df$pos2,
          annotations = sig_df$sig_label,
          y_position  = sig_df$bracket_y,
          tip_length  = 0.02,
          textsize    = 4.5,
          vjust       = 0.3,
          color       = "grey20"
        ) +
        # Expand y axis to accommodate brackets
        scale_y_continuous(
          expand = expansion(mult = c(0.05,
                                      0.1 + 0.15 * nrow(sig_df)))
          
        )
    }
  }
  
  p
}
cat("\n=== Q1c Welch ANOVAs ===\n")

cat("\n--- Gap_amp ---\n")
welch_facil_amp   <- oneway.test(Gap_amp  ~ age_bin, data = data_facil_amp,  var.equal = FALSE)
eta2_facil_amp    <- effectsize::eta_squared(aov(Gap_amp  ~ age_bin, data = data_facil_amp),  partial = FALSE)
emm_facil_amp_bin <- emmeans(lm(Gap_amp  ~ age_bin, data = data_facil_amp),  ~ age_bin)
pairs_facil_amp   <- pairs(emm_facil_amp_bin, adjust = "tukey")
print(welch_facil_amp); cat("η² =", round(as.numeric(eta2_facil_amp$Eta2), 4), "\n"); print(pairs_facil_amp)

cat("\n--- Overlap_amp ---\n")
welch_diseng_amp   <- oneway.test(Overlap_amp ~ age_bin, data = data_diseng_amp, var.equal = FALSE)
eta2_diseng_amp    <- effectsize::eta_squared(aov(Overlap_amp ~ age_bin, data = data_diseng_amp), partial = FALSE)
emm_diseng_amp_bin <- emmeans(lm(Overlap_amp ~ age_bin, data = data_diseng_amp), ~ age_bin)
pairs_diseng_amp   <- pairs(emm_diseng_amp_bin, adjust = "tukey")
print(welch_diseng_amp); cat("η² =", round(as.numeric(eta2_diseng_amp$Eta2), 4), "\n"); print(pairs_diseng_amp)

cat("\n---Gap_lat ---\n")
welch_facil_lat   <- oneway.test(Gap_lat  ~ age_bin, data = data_facil_lat,  var.equal = FALSE)
eta2_facil_lat    <- effectsize::eta_squared(aov(Gap_lat  ~ age_bin, data = data_facil_lat),  partial = FALSE)
emm_facil_lat_bin <- emmeans(lm(Gap_lat  ~ age_bin, data = data_facil_lat),  ~ age_bin)
pairs_facil_lat   <- pairs(emm_facil_lat_bin, adjust = "tukey")
print(welch_facil_lat); cat("η² =", round(as.numeric(eta2_facil_lat$Eta2), 4), "\n"); print(pairs_facil_lat)

cat("\n--- Overlap_lat ---\n")
welch_diseng_lat   <- oneway.test(Overlap_lat ~ age_bin, data = data_diseng_lat, var.equal = FALSE)
eta2_diseng_lat    <- effectsize::eta_squared(aov(Overlap_lat ~ age_bin, data = data_diseng_lat), partial = FALSE)
emm_diseng_lat_bin <- emmeans(lm(Overlap_lat ~ age_bin, data = data_diseng_lat), ~ age_bin)
pairs_diseng_lat   <- pairs(emm_diseng_lat_bin, adjust = "tukey")
print(welch_diseng_lat); cat("η² =", round(as.numeric(eta2_diseng_lat$Eta2), 4), "\n"); print(pairs_diseng_lat)

cat("\n--- Gap_rt ---\n")
welch_facil_rt   <- oneway.test(Gap_rt   ~ age_bin, data = data_facil_rt,   var.equal = FALSE)
eta2_facil_rt    <- effectsize::eta_squared(aov(Gap_rt   ~ age_bin, data = data_facil_rt),   partial = FALSE)
emm_facil_rt_bin <- emmeans(lm(Gap_rt   ~ age_bin, data = data_facil_rt),   ~ age_bin)
pairs_facil_rt   <- pairs(emm_facil_rt_bin, adjust = "tukey")
print(welch_facil_rt); cat("η² =", round(as.numeric(eta2_facil_rt$Eta2), 4), "\n"); print(pairs_facil_rt)

cat("\n--- Overlap_rt ---\n")
welch_diseng_rt   <- oneway.test(Overlap_rt  ~ age_bin, data = data_diseng_rt,  var.equal = FALSE)
eta2_diseng_rt    <- effectsize::eta_squared(aov(Overlap_rt  ~ age_bin, data = data_diseng_rt),  partial = FALSE)
emm_diseng_rt_bin <- emmeans(lm(Overlap_rt  ~ age_bin, data = data_diseng_rt),  ~ age_bin)
pairs_diseng_rt   <- pairs(emm_diseng_rt_bin, adjust = "tukey")
print(welch_diseng_rt); cat("η² =", round(as.numeric(eta2_diseng_rt$Eta2), 4), "\n"); print(pairs_diseng_rt)

write.csv(as.data.frame(pairs_facil_amp),  "output/tables/Q1c_posthoc_Gap_amp.csv",  row.names = FALSE)
write.csv(as.data.frame(pairs_diseng_amp), "output/tables/Q1c_posthoc_Overlap_amp.csv", row.names = FALSE)
write.csv(as.data.frame(pairs_facil_lat),  "output/tables/Q1c_posthoc_Gap_lat.csv",  row.names = FALSE)
write.csv(as.data.frame(pairs_diseng_lat), "output/tables/Q1c_posthoc_Overlap_lat.csv", row.names = FALSE)
write.csv(as.data.frame(pairs_facil_rt),   "output/tables/Q1c_posthoc_Gap_rt.csv",   row.names = FALSE)
write.csv(as.data.frame(pairs_diseng_rt),  "output/tables/Q1c_posthoc_Overlap_rt.csv",  row.names = FALSE)

p1c_facil_amp  <- make_agebin_plot(data_facil_amp,  "Gap_amp",  pairs_facil_amp)
p1c_diseng_amp <- make_agebin_plot(data_diseng_amp, "Overlap_amp", pairs_diseng_amp)
p1c_facil_lat  <- make_agebin_plot(data_facil_lat,  "Gap_lat",  pairs_facil_lat)
p1c_diseng_lat <- make_agebin_plot(data_diseng_lat, "Overlap_lat", pairs_diseng_lat)
p1c_facil_rt   <- make_agebin_plot(data_facil_rt,   "Gap_rt",   pairs_facil_rt)
p1c_diseng_rt  <- make_agebin_plot(data_diseng_rt,  "Overlap_rt",  pairs_diseng_rt)


print(p1c_facil_amp);  ggsave("output/figures/Q1c_agebin_Gap_amp.png",  p1c_facil_amp,  width=8, height=6, dpi=300, bg="white")
print(p1c_diseng_amp); ggsave("output/figures/Q1c_agebin_Overlap_amp.png", p1c_diseng_amp, width=8, height=6, dpi=300, bg="white")
print(p1c_facil_lat);  ggsave("output/figures/Q1c_agebin_Gap_lat.png",  p1c_facil_lat,  width=8, height=6, dpi=300, bg="white")
print(p1c_diseng_lat); ggsave("output/figures/Q1c_agebin_Overlap_lat.png", p1c_diseng_lat, width=8, height=6, dpi=300, bg="white")
print(p1c_facil_rt);   ggsave("output/figures/Q1c_agebin_Gap_rt.png",   p1c_facil_rt,   width=8, height=6, dpi=300, bg="white")
print(p1c_diseng_rt);  ggsave("output/figures/Q1c_agebin_Overlap_rt.png",  p1c_diseng_rt,  width=8, height=6, dpi=300, bg="white")

# =============================================================================
# Q2: NDD MODERATION — Linear Models
# Outcomes: Gap_amp, Overlap_amp,Gap_lat, Overlap_lat


# ── Gap_amp ─────────────────────────────────────────────────────────
cat("\n\n========== Q2 NDD: Gap_amp ==========\n")
cat("N =", nrow(data_facil_amp), "\n")

lm_q2_facil_amp_m1 <- lm(Gap_amp ~ age + sex + site,             data = data_facil_amp)
lm_q2_facil_amp_m2 <- lm(Gap_amp ~ ndd + age + sex + site,       data = data_facil_amp)
lm_q2_facil_amp_m3 <- lm(Gap_amp ~ ndd * age + sex + site,       data = data_facil_amp)

cat("\nM1 vs M2 (NDD main effect):\n");   print(anova(lm_q2_facil_amp_m1, lm_q2_facil_amp_m2))
cat("\nM2 vs M3 (NDD × age interaction):\n"); print(anova(lm_q2_facil_amp_m2, lm_q2_facil_amp_m3))

cat("\nM2 summary:\n"); print(summary(lm_q2_facil_amp_m2))
cat("\nM3 summary:\n"); print(summary(lm_q2_facil_amp_m3))

emm_q2_facil_amp   <- emmeans(lm_q2_facil_amp_m2, ~ ndd, at = list(age = median(data_facil_amp$age, na.rm = TRUE)))
pairs_q2_facil_amp <- pairs(emm_q2_facil_amp, adjust = "tukey")
eff_q2_facil_amp   <- eff_size(emm_q2_facil_amp, sigma = sigma(lm_q2_facil_amp_m2), edf = df.residual(lm_q2_facil_amp_m2))
cat("\nMarginal means (M2, at median age):\n"); print(emm_q2_facil_amp)
cat("\nPairwise:\n"); print(pairs_q2_facil_amp)
cat("\nEffect sizes:\n"); print(eff_q2_facil_amp)
write.csv(as.data.frame(summary(pairs_q2_facil_amp)), "output/tables/Q2_NDD_pairwise_Gap_amp.csv", row.names = FALSE)

# ── Overlap_amp ─────────────────────────────────────────────────────────
cat("\n\n========== Q2 NDD: Overlap_amp ==========\n")
cat("N =", nrow(data_diseng_amp), "\n")

lm_q2_diseng_amp_m1 <- lm(Overlap_amp ~ age + sex + site,           data = data_diseng_amp)
lm_q2_diseng_amp_m2 <- lm(Overlap_amp ~ ndd + age + sex + site,     data = data_diseng_amp)
lm_q2_diseng_amp_m3 <- lm(Overlap_amp ~ ndd * age + sex + site,     data = data_diseng_amp)

cat("\nM1 vs M2 (NDD main effect):\n");       print(anova(lm_q2_diseng_amp_m1, lm_q2_diseng_amp_m2))
cat("\nM2 vs M3 (NDD × age interaction):\n"); print(anova(lm_q2_diseng_amp_m2, lm_q2_diseng_amp_m3))

cat("\nM2 summary:\n"); print(summary(lm_q2_diseng_amp_m2))
cat("\nM3 summary:\n"); print(summary(lm_q2_diseng_amp_m3))

emm_q2_diseng_amp   <- emmeans(lm_q2_diseng_amp_m2, ~ ndd, at = list(age = median(data_diseng_amp$age, na.rm = TRUE)))
pairs_q2_diseng_amp <- pairs(emm_q2_diseng_amp, adjust = "tukey")
eff_q2_diseng_amp   <- eff_size(emm_q2_diseng_amp, sigma = sigma(lm_q2_diseng_amp_m2), edf = df.residual(lm_q2_diseng_amp_m2))
cat("\nMarginal means (M2, at median age):\n"); print(emm_q2_diseng_amp)
cat("\nPairwise:\n"); print(pairs_q2_diseng_amp)
cat("\nEffect sizes:\n"); print(eff_q2_diseng_amp)
write.csv(as.data.frame(summary(pairs_q2_diseng_amp)), "output/tables/Q2_NDD_pairwise_Overlap_amp.csv", row.names = FALSE)

# ──Gap_lat (lmer: family ICC = 2.9%, non-singular) ─────────────────
cat("\n\n========== Q2 NDD:Gap_lat ==========\n")
cat("N =", nrow(data_facil_lat), "\n")

# ML for LRT comparisons
lmer_q2_facil_lat_m1 <- lmer(Gap_lat ~ age + sex + site + (1|fam_id),
                             data = data_facil_lat, REML = FALSE)
lmer_q2_facil_lat_m2 <- lmer(Gap_lat ~ ndd + age + sex + site + (1|fam_id),
                             data = data_facil_lat, REML = FALSE)
lmer_q2_facil_lat_m3 <- lmer(Gap_lat ~ ndd * age + sex + site + (1|fam_id),
                             data = data_facil_lat, REML = FALSE)

cat("\nM1 vs M2 (NDD main effect):\n");       print(anova(lmer_q2_facil_lat_m1, lmer_q2_facil_lat_m2))
cat("\nM2 vs M3 (NDD × age interaction):\n"); print(anova(lmer_q2_facil_lat_m2, lmer_q2_facil_lat_m3))

# Refit with REML for final summaries
lmer_q2_facil_lat_m2_reml <- lmer(Gap_lat ~ ndd + age + sex + site + (1|fam_id),
                                  data = data_facil_lat, REML = TRUE)
lmer_q2_facil_lat_m3_reml <- lmer(Gap_lat ~ ndd * age + sex + site + (1|fam_id),
                                  data = data_facil_lat, REML = TRUE)
cat("\nM2 summary (REML):\n"); print(summary(lmer_q2_facil_lat_m2_reml))
cat("\nM3 summary (REML):\n"); print(summary(lmer_q2_facil_lat_m3_reml))

emm_q2_facil_lat   <- emmeans(lmer_q2_facil_lat_m2_reml, ~ ndd,
                              at = list(age = median(data_facil_lat$age, na.rm = TRUE)))
pairs_q2_facil_lat <- pairs(emm_q2_facil_lat, adjust = "tukey")
eff_q2_facil_lat   <- eff_size(emm_q2_facil_lat,
                               sigma = sigma(lmer_q2_facil_lat_m2_reml),
                               edf   = df.residual(lmer_q2_facil_lat_m2_reml))
cat("\nMarginal means (M2, at median age):\n"); print(emm_q2_facil_lat)
cat("\nPairwise:\n"); print(pairs_q2_facil_lat)
cat("\nEffect sizes:\n"); print(eff_q2_facil_lat)
write.csv(as.data.frame(summary(pairs_q2_facil_lat)), "output/tables/Q2_NDD_pairwise_Gap_lat.csv", row.names = FALSE)

# ── Overlap_lat ─────────────────────────────────────────────────────────
cat("\n\n========== Q2 NDD: Overlap_lat ==========\n")
cat("N =", nrow(data_diseng_lat), "\n")

lm_q2_diseng_lat_m1 <- lm(Overlap_lat ~ age + sex + site,           data = data_diseng_lat)
lm_q2_diseng_lat_m2 <- lm(Overlap_lat ~ ndd + age + sex + site,     data = data_diseng_lat)
lm_q2_diseng_lat_m3 <- lm(Overlap_lat ~ ndd * age + sex + site,     data = data_diseng_lat)

cat("\nM1 vs M2 (NDD main effect):\n");       print(anova(lm_q2_diseng_lat_m1, lm_q2_diseng_lat_m2))
cat("\nM2 vs M3 (NDD × age interaction):\n"); print(anova(lm_q2_diseng_lat_m2, lm_q2_diseng_lat_m3))

cat("\nM2 summary:\n"); print(summary(lm_q2_diseng_lat_m2))
cat("\nM3 summary:\n"); print(summary(lm_q2_diseng_lat_m3))

emm_q2_diseng_lat   <- emmeans(lm_q2_diseng_lat_m2, ~ ndd, at = list(age = median(data_diseng_lat$age, na.rm = TRUE)))
pairs_q2_diseng_lat <- pairs(emm_q2_diseng_lat, adjust = "tukey")
eff_q2_diseng_lat   <- eff_size(emm_q2_diseng_lat, sigma = sigma(lm_q2_diseng_lat_m2), edf = df.residual(lm_q2_diseng_lat_m2))
cat("\nMarginal means (M2, at median age):\n"); print(emm_q2_diseng_lat)
cat("\nPairwise:\n"); print(pairs_q2_diseng_lat)
cat("\nEffect sizes:\n"); print(eff_q2_diseng_lat)
write.csv(as.data.frame(summary(pairs_q2_diseng_lat)), "output/tables/Q2_NDD_pairwise_Overlap_lat.csv", row.names = FALSE)

# =============================================================================
# BH-FDR correction across Q2 NDD tests
# =============================================================================

q2_p_main <- c(
  anova(lm_q2_facil_amp_m1,  lm_q2_facil_amp_m2)$`Pr(>F)`[2],
  anova(lm_q2_diseng_amp_m1, lm_q2_diseng_amp_m2)$`Pr(>F)`[2],
  anova(lmer_q2_facil_lat_m1,  lmer_q2_facil_lat_m2)$`Pr(>Chisq)`[2],
  anova(lm_q2_diseng_lat_m1, lm_q2_diseng_lat_m2)$`Pr(>F)`[2]
)
q2_p_int <- c(
  anova(lm_q2_facil_amp_m2,  lm_q2_facil_amp_m3)$`Pr(>F)`[2],
  anova(lm_q2_diseng_amp_m2, lm_q2_diseng_amp_m3)$`Pr(>F)`[2],
  anova(lmer_q2_facil_lat_m2,  lmer_q2_facil_lat_m3)$`Pr(>Chisq)`[2],
  anova(lm_q2_diseng_lat_m2, lm_q2_diseng_lat_m3)$`Pr(>F)`[2]
)

q2_fdr_table <- data.frame(
  Outcome         = c("Gap_amp", "Overlap_amp",
                      "Gap_lat", "Overlap_lat"),
  p_NDD_main      = round(q2_p_main, 4),
  p_NDD_main_FDR  = round(p.adjust(q2_p_main, method = "BH"), 4),
  sig_main_FDR    = ifelse(p.adjust(q2_p_main, method = "BH") < .05, "*", "ns"),
  p_NDD_x_age     = round(q2_p_int, 4),
  p_NDD_x_age_FDR = round(p.adjust(q2_p_int, method = "BH"), 4),
  sig_int_FDR     = ifelse(p.adjust(q2_p_int, method = "BH") < .05, "*", "ns")
)
cat("\n\n=== Q2 BH-FDR Summary Table ===\n")
print(q2_fdr_table)
write.csv(q2_fdr_table, "output/tables/Q2_NDD_BH_FDR_summary.csv", row.names = FALSE)

# =============================================================================
# Q2 Trajectory plots (lm/lmer, NDD × age)
# =============================================================================

make_ndd_traj_plot_lm <- function(model, data_v, outcome_name) {
  age_seq  <- seq(min(data_v$age, na.rm = TRUE),
                  max(data_v$age, na.rm = TRUE), length.out = 200)
  ref_site <- levels(data_v$site)[1]
  ref_sex  <- levels(data_v$sex)[1]
  
  build_pred <- function(ndd_level) {
    pg <- data.frame(age    = age_seq,
                     ndd    = factor(ndd_level, levels = levels(data_v$ndd)),
                     sex    = factor(ref_sex,   levels = levels(data_v$sex)),
                     site   = factor(ref_site,  levels = levels(data_v$site)),
                     fam_id = levels(data_v$fam_id)[1])
    if (inherits(model, "lmerMod")) {
      mm  <- model.matrix(~ ndd + age + sex + site, data = pg)
      fe  <- lme4::fixef(model)
      vcv <- as.matrix(vcov(model))
      fit <- as.numeric(mm %*% fe)
      se  <- sqrt(rowSums((mm %*% vcv) * mm))
    } else {
      pr  <- predict(model, newdata = pg, se.fit = TRUE)
      fit <- pr$fit
      se  <- pr$se.fit
    }
    pg$fit <- fit
    pg$lwr <- fit - 1.96 * se
    pg$upr <- fit + 1.96 * se
    pg
  }
  
  pred_df <- rbind(build_pred("Non-NDD"), build_pred("NDD"))
  
  ggplot() +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
    geom_jitter(data = data_v,
                aes(x = age, y = .data[[outcome_name]], color = ndd),
                alpha = 0.25, size = 1.2, width = 0.15) +
    geom_ribbon(data = pred_df,
                aes(x = age, ymin = lwr, ymax = upr, fill = ndd), alpha = 0.2) +
    geom_line(data = pred_df,
              aes(x = age, y = fit, color = ndd), linewidth = 1.3) +
    scale_color_manual(values = group_colors) +
    scale_fill_manual(values  = group_colors) +
    labs(title    = paste("Q2: NDD Moderation —", outcome_name),
         subtitle = paste0("Linear model ± 95% CI (marginal: ", ref_sex, ", ", ref_site, ")"),
         x = "Age (years)", y = outcome_name, color = "Group", fill = "Group") +
    theme_pub
}

p2_facil_amp  <- make_ndd_traj_plot_lm(lm_q2_facil_amp_m3,         data_facil_amp,  "Gap_amp")
p2_diseng_amp <- make_ndd_traj_plot_lm(lm_q2_diseng_amp_m3,        data_diseng_amp, "Overlap_amp")
p2_facil_lat  <- make_ndd_traj_plot_lm(lmer_q2_facil_lat_m3_reml,  data_facil_lat,  "Gap_lat")
p2_diseng_lat <- make_ndd_traj_plot_lm(lm_q2_diseng_lat_m3,        data_diseng_lat, "Overlap_lat")

print(p2_facil_amp);  ggsave("output/figures/Q2_NDD_trajectory_Gap_amp.png",  p2_facil_amp,  width=10, height=6, dpi=300, bg="white")
print(p2_diseng_amp); ggsave("output/figures/Q2_NDD_trajectory_Overlap_amp.png", p2_diseng_amp, width=10, height=6, dpi=300, bg="white")
print(p2_facil_lat);  ggsave("output/figures/Q2_NDD_trajectory_Gap_lat.png",  p2_facil_lat,  width=10, height=6, dpi=300, bg="white")
print(p2_diseng_lat); ggsave("output/figures/Q2_NDD_trajectory_Overlap_lat.png", p2_diseng_lat, width=10, height=6, dpi=300, bg="white")


# =============================================================================
# Q2 SECONDARY: AFFECTED GROUP (3-LEVEL) — Linear Models
# Outcomes: Gap_amp, Overlap_amp,Gap_lat, Overlap_lat
# (rt outcomes dropped: non-significant in Q1b/Q1c)
# M0 = age+sex+site; M1 = +affected_group; M2 = +affected_group*age
# BH-FDR across 4 outcomes per test family
# =============================================================================

data_3lvl_facil_amp  <- data[!is.na(data$Gap_amp)  & !is.na(data$affected_group), ]
data_3lvl_diseng_amp <- data[!is.na(data$Overlap_amp) & !is.na(data$affected_group), ]
data_3lvl_facil_lat  <- data[!is.na(data$Gap_lat)  & !is.na(data$affected_group), ]
data_3lvl_diseng_lat <- data[!is.na(data$Overlap_lat) & !is.na(data$affected_group), ]

# ── Gap_amp ──────────────────────────────────────────────────────────
cat("\n\n===== Q2 SECONDARY: AFFECTED GROUP — Gap_amp =====\n")
cat("N =", nrow(data_3lvl_facil_amp), "| Group counts:\n")
print(table(data_3lvl_facil_amp$affected_group))

lm_3lvl_facil_amp_m0 <- lm(Gap_amp ~ age + sex + site,
                           data = data_3lvl_facil_amp)
lm_3lvl_facil_amp_m1 <- lm(Gap_amp ~ affected_group + age + sex + site,
                           data = data_3lvl_facil_amp)
lm_3lvl_facil_amp_m2 <- lm(Gap_amp ~ affected_group * age + sex + site,
                           data = data_3lvl_facil_amp)

cat("\nLRT M0 vs M1 (group main effect):\n")
print(anova(lm_3lvl_facil_amp_m0, lm_3lvl_facil_amp_m1))
cat("\nLRT M1 vs M2 (group × age interaction):\n")
print(anova(lm_3lvl_facil_amp_m1, lm_3lvl_facil_amp_m2))
cat("\nM1 summary:\n"); print(summary(lm_3lvl_facil_amp_m1))

emm_3lvl_facil_amp   <- emmeans(lm_3lvl_facil_amp_m1, ~ affected_group,
                                at = list(age = median(data_3lvl_facil_amp$age, na.rm = TRUE)))
pairs_3lvl_facil_amp <- pairs(emm_3lvl_facil_amp, adjust = "tukey")
eff_3lvl_facil_amp   <- eff_size(emm_3lvl_facil_amp,
                                 sigma = sigma(lm_3lvl_facil_amp_m1),
                                 edf   = df.residual(lm_3lvl_facil_amp_m1))
cat("\nMarginal means (M1, at median age):\n"); print(emm_3lvl_facil_amp)
cat("\nPairwise (Tukey):\n");                   print(pairs_3lvl_facil_amp)
cat("\nEffect sizes:\n");                        print(eff_3lvl_facil_amp)
write.csv(as.data.frame(summary(pairs_3lvl_facil_amp)),
          "output/tables/Q2_3lvl_pairwise_Gap_amp.csv", row.names = FALSE)

# ── Overlap_amp ─────────────────────────────────────────────────────────
cat("\n\n===== Q2 SECONDARY: AFFECTED GROUP — Overlap_amp =====\n")
cat("N =", nrow(data_3lvl_diseng_amp), "| Group counts:\n")
print(table(data_3lvl_diseng_amp$affected_group))

lm_3lvl_diseng_amp_m0 <- lm(Overlap_amp ~ age + sex + site,
                            data = data_3lvl_diseng_amp)
lm_3lvl_diseng_amp_m1 <- lm(Overlap_amp ~ affected_group + age + sex + site,
                            data = data_3lvl_diseng_amp)
lm_3lvl_diseng_amp_m2 <- lm(Overlap_amp ~ affected_group * age + sex + site,
                            data = data_3lvl_diseng_amp)

cat("\nLRT M0 vs M1 (group main effect):\n")
print(anova(lm_3lvl_diseng_amp_m0, lm_3lvl_diseng_amp_m1))
cat("\nLRT M1 vs M2 (group × age interaction):\n")
print(anova(lm_3lvl_diseng_amp_m1, lm_3lvl_diseng_amp_m2))
cat("\nM1 summary:\n"); print(summary(lm_3lvl_diseng_amp_m1))

emm_3lvl_diseng_amp   <- emmeans(lm_3lvl_diseng_amp_m1, ~ affected_group,
                                 at = list(age = median(data_3lvl_diseng_amp$age, na.rm = TRUE)))
pairs_3lvl_diseng_amp <- pairs(emm_3lvl_diseng_amp, adjust = "tukey")
eff_3lvl_diseng_amp   <- eff_size(emm_3lvl_diseng_amp,
                                  sigma = sigma(lm_3lvl_diseng_amp_m1),
                                  edf   = df.residual(lm_3lvl_diseng_amp_m1))
cat("\nMarginal means (M1, at median age):\n"); print(emm_3lvl_diseng_amp)
cat("\nPairwise (Tukey):\n");                   print(pairs_3lvl_diseng_amp)
cat("\nEffect sizes:\n");                        print(eff_3lvl_diseng_amp)
write.csv(as.data.frame(summary(pairs_3lvl_diseng_amp)),
          "output/tables/Q2_3lvl_pairwise_Overlap_amp.csv", row.names = FALSE)

# ──Gap_lat (lmer: ICC = 2.9%, non-singular) ────────────────────────
cat("\n\n===== Q2 SECONDARY: AFFECTED GROUP —Gap_lat =====\n")
cat("N =", nrow(data_3lvl_facil_lat), "| Group counts:\n")
print(table(data_3lvl_facil_lat$affected_group))

# ML for LRT comparisons
lmer_3lvl_facil_lat_m0 <- lmer(Gap_lat ~ age + sex + site + (1|fam_id),
                               data = data_3lvl_facil_lat, REML = FALSE)
lmer_3lvl_facil_lat_m1 <- lmer(Gap_lat ~ affected_group + age + sex + site + (1|fam_id),
                               data = data_3lvl_facil_lat, REML = FALSE)
lmer_3lvl_facil_lat_m2 <- lmer(Gap_lat ~ affected_group * age + sex + site + (1|fam_id),
                               data = data_3lvl_facil_lat, REML = FALSE)

cat("\nLRT M0 vs M1 (group main effect):\n")
print(anova(lmer_3lvl_facil_lat_m0, lmer_3lvl_facil_lat_m1))
cat("\nLRT M1 vs M2 (group × age interaction):\n")
print(anova(lmer_3lvl_facil_lat_m1, lmer_3lvl_facil_lat_m2))

# REML for final summary and emmeans
lmer_3lvl_facil_lat_m1r <- lmer(Gap_lat ~ affected_group + age + sex + site + (1|fam_id),
                                data = data_3lvl_facil_lat, REML = TRUE)
cat("\nM1 summary (REML):\n"); print(summary(lmer_3lvl_facil_lat_m1r))

emm_3lvl_facil_lat   <- emmeans(lmer_3lvl_facil_lat_m1r, ~ affected_group,
                                at = list(age = median(data_3lvl_facil_lat$age, na.rm = TRUE)))
pairs_3lvl_facil_lat <- pairs(emm_3lvl_facil_lat, adjust = "tukey")
eff_3lvl_facil_lat   <- eff_size(emm_3lvl_facil_lat,
                                 sigma = sigma(lmer_3lvl_facil_lat_m1r),
                                 edf   = df.residual(lmer_3lvl_facil_lat_m1r))
cat("\nMarginal means (M1, at median age):\n"); print(emm_3lvl_facil_lat)
cat("\nPairwise (Tukey):\n");                   print(pairs_3lvl_facil_lat)
cat("\nEffect sizes:\n");                        print(eff_3lvl_facil_lat)
write.csv(as.data.frame(summary(pairs_3lvl_facil_lat)),
          "output/tables/Q2_3lvl_pairwise_Gap_lat.csv", row.names = FALSE)

# ── Overlap_lat ─────────────────────────────────────────────────────────
cat("\n\n===== Q2 SECONDARY: AFFECTED GROUP — Overlap_lat =====\n")
cat("N =", nrow(data_3lvl_diseng_lat), "| Group counts:\n")
print(table(data_3lvl_diseng_lat$affected_group))

lm_3lvl_diseng_lat_m0 <- lm(Overlap_lat ~ age + sex + site,
                            data = data_3lvl_diseng_lat)
lm_3lvl_diseng_lat_m1 <- lm(Overlap_lat ~ affected_group + age + sex + site,
                            data = data_3lvl_diseng_lat)
lm_3lvl_diseng_lat_m2 <- lm(Overlap_lat ~ affected_group * age + sex + site,
                            data = data_3lvl_diseng_lat)

cat("\nLRT M0 vs M1 (group main effect):\n")
print(anova(lm_3lvl_diseng_lat_m0, lm_3lvl_diseng_lat_m1))
cat("\nLRT M1 vs M2 (group × age interaction):\n")
print(anova(lm_3lvl_diseng_lat_m1, lm_3lvl_diseng_lat_m2))
cat("\nM1 summary:\n"); print(summary(lm_3lvl_diseng_lat_m1))

emm_3lvl_diseng_lat   <- emmeans(lm_3lvl_diseng_lat_m1, ~ affected_group,
                                 at = list(age = median(data_3lvl_diseng_lat$age, na.rm = TRUE)))
pairs_3lvl_diseng_lat <- pairs(emm_3lvl_diseng_lat, adjust = "tukey")
eff_3lvl_diseng_lat   <- eff_size(emm_3lvl_diseng_lat,
                                  sigma = sigma(lm_3lvl_diseng_lat_m1),
                                  edf   = df.residual(lm_3lvl_diseng_lat_m1))
cat("\nMarginal means (M1, at median age):\n"); print(emm_3lvl_diseng_lat)
cat("\nPairwise (Tukey):\n");                   print(pairs_3lvl_diseng_lat)
cat("\nEffect sizes:\n");                        print(eff_3lvl_diseng_lat)
write.csv(as.data.frame(summary(pairs_3lvl_diseng_lat)),
          "output/tables/Q2_3lvl_pairwise_Overlap_lat.csv", row.names = FALSE)

# =============================================================================
# BH-FDR across Q2 3-level tests (4 outcomes × 2 families)
# =============================================================================

q2_3lvl_p_main <- c(
  anova(lm_3lvl_facil_amp_m0,    lm_3lvl_facil_amp_m1)$`Pr(>F)`[2],
  anova(lm_3lvl_diseng_amp_m0,   lm_3lvl_diseng_amp_m1)$`Pr(>F)`[2],
  anova(lmer_3lvl_facil_lat_m0,  lmer_3lvl_facil_lat_m1)$`Pr(>Chisq)`[2],
  anova(lm_3lvl_diseng_lat_m0,   lm_3lvl_diseng_lat_m1)$`Pr(>F)`[2]
)

q2_3lvl_p_int <- c(
  anova(lm_3lvl_facil_amp_m1,    lm_3lvl_facil_amp_m2)$`Pr(>F)`[2],
  anova(lm_3lvl_diseng_amp_m1,   lm_3lvl_diseng_amp_m2)$`Pr(>F)`[2],
  anova(lmer_3lvl_facil_lat_m1,  lmer_3lvl_facil_lat_m2)$`Pr(>Chisq)`[2],
  anova(lm_3lvl_diseng_lat_m1,   lm_3lvl_diseng_lat_m2)$`Pr(>F)`[2]
)

q2_3lvl_fdr <- data.frame(
  Outcome           = c("Gap_amp", "Overlap_amp",
                        "Gap_lat", "Overlap_lat"),
  p_group_main      = round(q2_3lvl_p_main, 4),
  p_group_main_FDR  = round(p.adjust(q2_3lvl_p_main, method = "BH"), 4),
  sig_main_FDR      = ifelse(p.adjust(q2_3lvl_p_main, method = "BH") < .05, "*", "ns"),
  p_group_x_age     = round(q2_3lvl_p_int, 4),
  p_group_x_age_FDR = round(p.adjust(q2_3lvl_p_int, method = "BH"), 4),
  sig_int_FDR       = ifelse(p.adjust(q2_3lvl_p_int, method = "BH") < .05, "*", "ns")
)
cat("\n\n=== Q2 3-level BH-FDR Summary ===\n")
print(q2_3lvl_fdr)
write.csv(q2_3lvl_fdr, "output/tables/Q2_3lvl_BH_FDR_summary.csv", row.names = FALSE)

# =============================================================================
# Q2 SENSITIVITY: PEDIATRIC (age ≤ 18)
# Run for all 4 outcomes; interpret only those with p_group_main_FDR < .05 above
# Model: M1 only (group main effect + age + sex + site)
# No group×age interaction: age range is narrow (2–18), sensitivity check only
# =============================================================================

data_u18 <- data %>% filter(age <= 18)
cat("\n\n===== Q2 SENSITIVITY: PEDIATRIC (age ≤ 18) — N =", nrow(data_u18), "=====\n")
print(table(data_u18$affected_group))

data_u18_facil_amp  <- data_u18[!is.na(data_u18$Gap_amp)  & !is.na(data_u18$affected_group), ]
data_u18_diseng_amp <- data_u18[!is.na(data_u18$Overlap_amp) & !is.na(data_u18$affected_group), ]
data_u18_facil_lat  <- data_u18[!is.na(data_u18$Gap_lat)  & !is.na(data_u18$affected_group), ]
data_u18_diseng_lat <- data_u18[!is.na(data_u18$Overlap_lat) & !is.na(data_u18$affected_group), ]

# ── Gap_amp (pediatric) ──────────────────────────────────────────────
cat("\n--- Sensitivity (≤18): Gap_amp  N =", nrow(data_u18_facil_amp), "---\n")
print(table(data_u18_facil_amp$affected_group))

lm_sens_facil_amp_m0 <- lm(Gap_amp ~ age + sex + site,
                           data = data_u18_facil_amp)
lm_sens_facil_amp_m1 <- lm(Gap_amp ~ affected_group + age + sex + site,
                           data = data_u18_facil_amp)

cat("\nLRT M0 vs M1 (group main effect):\n")
print(anova(lm_sens_facil_amp_m0, lm_sens_facil_amp_m1))
cat("\nM1 summary:\n"); print(summary(lm_sens_facil_amp_m1))

emm_sens_facil_amp   <- emmeans(lm_sens_facil_amp_m1, ~ affected_group,
                                at = list(age = median(data_u18_facil_amp$age, na.rm = TRUE)))
pairs_sens_facil_amp <- pairs(emm_sens_facil_amp, adjust = "tukey")
eff_sens_facil_amp   <- eff_size(emm_sens_facil_amp,
                                 sigma = sigma(lm_sens_facil_amp_m1),
                                 edf   = df.residual(lm_sens_facil_amp_m1))
cat("\nPairwise:\n"); print(pairs_sens_facil_amp)
cat("\nEffect sizes:\n"); print(eff_sens_facil_amp)
write.csv(as.data.frame(summary(pairs_sens_facil_amp)),
          "output/tables/Q2_sens_u18_pairwise_Gap_amp.csv", row.names = FALSE)

# ── Overlap_amp (pediatric) ─────────────────────────────────────────────
cat("\n--- Sensitivity (≤18): Overlap_amp  N =", nrow(data_u18_diseng_amp), "---\n")
print(table(data_u18_diseng_amp$affected_group))

lm_sens_diseng_amp_m0 <- lm(Overlap_amp ~ age + sex + site,
                            data = data_u18_diseng_amp)
lm_sens_diseng_amp_m1 <- lm(Overlap_amp ~ affected_group + age + sex + site,
                            data = data_u18_diseng_amp)

cat("\nLRT M0 vs M1 (group main effect):\n")
print(anova(lm_sens_diseng_amp_m0, lm_sens_diseng_amp_m1))
cat("\nM1 summary:\n"); print(summary(lm_sens_diseng_amp_m1))

emm_sens_diseng_amp   <- emmeans(lm_sens_diseng_amp_m1, ~ affected_group,
                                 at = list(age = median(data_u18_diseng_amp$age, na.rm = TRUE)))
pairs_sens_diseng_amp <- pairs(emm_sens_diseng_amp, adjust = "tukey")
eff_sens_diseng_amp   <- eff_size(emm_sens_diseng_amp,
                                  sigma = sigma(lm_sens_diseng_amp_m1),
                                  edf   = df.residual(lm_sens_diseng_amp_m1))
cat("\nPairwise:\n"); print(pairs_sens_diseng_amp)
cat("\nEffect sizes:\n"); print(eff_sens_diseng_amp)
write.csv(as.data.frame(summary(pairs_sens_diseng_amp)),
          "output/tables/Q2_sens_u18_pairwise_Overlap_amp.csv", row.names = FALSE)

# ──Gap_lat (pediatric, lmer) ────────────────────────────────────────
cat("\n--- Sensitivity (≤18):Gap_lat  N =", nrow(data_u18_facil_lat), "---\n")
print(table(data_u18_facil_lat$affected_group))

lmer_sens_facil_lat_m0 <- lmer(Gap_lat ~ age + sex + site + (1|fam_id),
                               data = data_u18_facil_lat, REML = FALSE)
lmer_sens_facil_lat_m1 <- lmer(Gap_lat ~ affected_group + age + sex + site + (1|fam_id),
                               data = data_u18_facil_lat, REML = FALSE)

cat("\nLRT M0 vs M1 (group main effect):\n")
print(anova(lmer_sens_facil_lat_m0, lmer_sens_facil_lat_m1))

lmer_sens_facil_lat_m1r <- lmer(Gap_lat ~ affected_group + age + sex + site + (1|fam_id),
                                data = data_u18_facil_lat, REML = TRUE)
cat("\nM1 summary (REML):\n"); print(summary(lmer_sens_facil_lat_m1r))

emm_sens_facil_lat   <- emmeans(lmer_sens_facil_lat_m1r, ~ affected_group,
                                at = list(age = median(data_u18_facil_lat$age, na.rm = TRUE)))
pairs_sens_facil_lat <- pairs(emm_sens_facil_lat, adjust = "tukey")
eff_sens_facil_lat   <- eff_size(emm_sens_facil_lat,
                                 sigma = sigma(lmer_sens_facil_lat_m1r),
                                 edf   = df.residual(lmer_sens_facil_lat_m1r))
cat("\nPairwise:\n"); print(pairs_sens_facil_lat)
cat("\nEffect sizes:\n"); print(eff_sens_facil_lat)
write.csv(as.data.frame(summary(pairs_sens_facil_lat)),
          "output/tables/Q2_sens_u18_pairwise_Gap_lat.csv", row.names = FALSE)

# ── Overlap_lat (pediatric) ─────────────────────────────────────────────
cat("\n--- Sensitivity (≤18): Overlap_lat  N =", nrow(data_u18_diseng_lat), "---\n")
print(table(data_u18_diseng_lat$affected_group))

lm_sens_diseng_lat_m0 <- lm(Overlap_lat ~ age + sex + site,
                            data = data_u18_diseng_lat)
lm_sens_diseng_lat_m1 <- lm(Overlap_lat ~ affected_group + age + sex + site,
                            data = data_u18_diseng_lat)

cat("\nLRT M0 vs M1 (group main effect):\n")
print(anova(lm_sens_diseng_lat_m0, lm_sens_diseng_lat_m1))
cat("\nM1 summary:\n"); print(summary(lm_sens_diseng_lat_m1))

emm_sens_diseng_lat   <- emmeans(lm_sens_diseng_lat_m1, ~ affected_group,
                                 at = list(age = median(data_u18_diseng_lat$age, na.rm = TRUE)))
pairs_sens_diseng_lat <- pairs(emm_sens_diseng_lat, adjust = "tukey")
eff_sens_diseng_lat   <- eff_size(emm_sens_diseng_lat,
                                  sigma = sigma(lm_sens_diseng_lat_m1),
                                  edf   = df.residual(lm_sens_diseng_lat_m1))
cat("\nPairwise:\n"); print(pairs_sens_diseng_lat)
cat("\nEffect sizes:\n"); print(eff_sens_diseng_lat)
write.csv(as.data.frame(summary(pairs_sens_diseng_lat)),
          "output/tables/Q2_sens_u18_pairwise_Overlap_lat.csv", row.names = FALSE)

# ── BH-FDR for pediatric sensitivity ─────────────────────────────────────────
q2_sens_p_main <- c(
  anova(lm_sens_facil_amp_m0,    lm_sens_facil_amp_m1)$`Pr(>F)`[2],
  anova(lm_sens_diseng_amp_m0,   lm_sens_diseng_amp_m1)$`Pr(>F)`[2],
  anova(lmer_sens_facil_lat_m0,  lmer_sens_facil_lat_m1)$`Pr(>Chisq)`[2],
  anova(lm_sens_diseng_lat_m0,   lm_sens_diseng_lat_m1)$`Pr(>F)`[2]
)

q2_sens_fdr <- data.frame(
  Outcome          = c("Gap_amp", "Overlap_amp",
                       "Gap_lat", "Overlap_lat"),
  p_group_main     = round(q2_sens_p_main, 4),
  p_group_main_FDR = round(p.adjust(q2_sens_p_main, method = "BH"), 4),
  sig_FDR          = ifelse(p.adjust(q2_sens_p_main, method = "BH") < .05, "*", "ns")
)
cat("\n\n=== Q2 Pediatric Sensitivity BH-FDR Summary ===\n")
print(q2_sens_fdr)
write.csv(q2_sens_fdr, "output/tables/Q2_sens_u18_BH_FDR_summary.csv", row.names = FALSE)

# =============================================================================
# Q3 EXTENSION: RAW ERP → RAW RT (Gap andOverlap conditions)
# Gap     →  Gap analog  (Gap_amp, Gap_lat, Gap_rt)
#Overlap → Overlap analog (Overlap_amp,Overlap_lat,Overlap_rt)
# =============================================================================

data_q3_raw_facil <- data %>%
  filter(!is.na(Gap_rt), !is.na(Gap_amp), !is.na(Gap_lat))
data_q3_raw_diseng <- data %>%
  filter(!is.na(Overlap_rt), !is.na(Overlap_amp), !is.na(Overlap_lat))

cat("\n\n========== Q3 EXTENSION: RAW — GAP CONDITION ==========\n")
cat("N (complete on Gap_rt + Gap_amp + Gap_lat):", nrow(data_q3_raw_facil), "\n")

# ── Bivariate correlations ────────────────────────────────────────────────────
cat("\n-- Bivariate correlations (raw Gap) --\n")
r_gap_amp_rt  <- cor.test(data_q3_raw_facil$Gap_amp, data_q3_raw_facil$Gap_rt,  method = "pearson")
r_gap_lat_rt  <- cor.test(data_q3_raw_facil$Gap_lat, data_q3_raw_facil$Gap_rt,  method = "pearson")
r_gap_amp_lat <- cor.test(data_q3_raw_facil$Gap_amp, data_q3_raw_facil$Gap_lat, method = "pearson")
cat("Gap_amp × Gap_rt:  r =", round(r_gap_amp_rt$estimate,  3), "p =", round(r_gap_amp_rt$p.value,  4), "\n")
cat("Gap_lat × Gap_rt:  r =", round(r_gap_lat_rt$estimate,  3), "p =", round(r_gap_lat_rt$p.value,  4), "\n")
cat("Gap_amp × Gap_lat: r =", round(r_gap_amp_lat$estimate, 3), "p =", round(r_gap_amp_lat$p.value, 4), "\n")

# ── Scatter plots ─────────────────────────────────────────────────────────────
p3_gap_amp <- ggplot(data_q3_raw_facil, aes(x = Gap_amp, y = Gap_rt, color = ndd)) +
  geom_point(alpha = 0.5, size = 2) +
  geom_smooth(method = "lm", se = TRUE, color = "grey20", linewidth = 1, aes(group = 1)) +
  scale_color_manual(values = group_colors) +
  annotate("text", x = Inf, y = Inf, hjust = 1.1, vjust = 1.5,
           label = paste0("r = ", round(r_gap_amp_rt$estimate, 2),
                          ", p = ", round(r_gap_amp_rt$p.value, 4))) +
  labs(title = "Q3 Raw: Gap — AMP × RT", x = "Gap_amp", y = "Gap_rt") + theme_pub

p3_gap_lat <- ggplot(data_q3_raw_facil, aes(x = Gap_lat, y = Gap_rt, color = ndd)) +
  geom_point(alpha = 0.5, size = 2) +
  geom_smooth(method = "lm", se = TRUE, color = "grey20", linewidth = 1, aes(group = 1)) +
  scale_color_manual(values = group_colors) +
  annotate("text", x = Inf, y = Inf, hjust = 1.1, vjust = 1.5,
           label = paste0("r = ", round(r_gap_lat_rt$estimate, 2),
                          ", p = ", round(r_gap_lat_rt$p.value, 4))) +
  labs(title = "Q3 Raw: Gap — LAT × RT", x = "Gap_lat", y = "Gap_rt") + theme_pub

print(p3_gap_amp + p3_gap_lat)
ggsave("output/figures/Q3_scatter_raw_gap.png", p3_gap_amp + p3_gap_lat,
       width = 14, height = 6, dpi = 300, bg = "white")

# ── Linear models: raw ERP → Gap_rt above age ─────────────────────────────────
cat("\n-- Linear models: raw Gap ERP → Gap_rt above age --\n")
lm3_gap_null <- lm(Gap_rt ~ age + sex + site,                           data = data_q3_raw_facil)
lm3_gap_amp  <- lm(Gap_rt ~ Gap_amp + age + sex + site,                 data = data_q3_raw_facil)
lm3_gap_lat  <- lm(Gap_rt ~ Gap_lat + age + sex + site,                 data = data_q3_raw_facil)
lm3_gap_both <- lm(Gap_rt ~ Gap_amp + Gap_lat + age + sex + site,       data = data_q3_raw_facil)

cat("\nLRT Null vs AMP+LAT:\n"); print(anova(lm3_gap_null, lm3_gap_both))
cat("\nLRT AMP only vs AMP+LAT:\n"); print(anova(lm3_gap_amp, lm3_gap_both))
cat("\nLRT LAT only vs AMP+LAT:\n"); print(anova(lm3_gap_lat, lm3_gap_both))

aic_q3_gap <- data.frame(
  Model     = c("Null (age+sex+site)", "AMP only", "LAT only", "AMP + LAT"),
  AIC       = c(AIC(lm3_gap_null), AIC(lm3_gap_amp), AIC(lm3_gap_lat), AIC(lm3_gap_both)),
  Adj_R2    = c(summary(lm3_gap_null)$adj.r.squared, summary(lm3_gap_amp)$adj.r.squared,
                summary(lm3_gap_lat)$adj.r.squared,  summary(lm3_gap_both)$adj.r.squared)
)
aic_q3_gap$Delta_AIC <- aic_q3_gap$AIC - min(aic_q3_gap$AIC)
print(aic_q3_gap)
cat("\nBest model (AMP + LAT) summary:\n"); print(summary(lm3_gap_both))
cat("\nVIF:\n"); print(car::vif(lm3_gap_both))
write.csv(aic_q3_gap, "output/tables/Q3_AIC_raw_gap.csv", row.names = FALSE)

# ── NDD × raw ERP interaction — Gap ───────────────────────────────────────────
cat("\n-- NDD × raw ERP interaction — Gap --\n")
lm3_gap_int <- lm(Gap_rt ~ ndd * Gap_amp + ndd * Gap_lat + age + sex + site,
                  data = data_q3_raw_facil)
cat("LRT: NDD×ERP vs ERP only:\n"); print(anova(lm3_gap_both, lm3_gap_int))
cat("Summary:\n"); print(summary(lm3_gap_int))

slope_data_gap <- data_q3_raw_facil %>%
  group_by(ndd) %>%
  summarise(r_amp = cor(Gap_amp, Gap_rt, use = "complete.obs"),
            r_lat = cor(Gap_lat, Gap_rt, use = "complete.obs"),
            n = n(), .groups = "drop")
cat("Raw Gap ERP–RT correlation by NDD group:\n"); print(slope_data_gap)

# ── Comparison: relative vs raw coupling —  Gap ───────────────────────
cat("\n-- Relative vs Raw coupling ( Gap condition) --\n")
cat("Relative: Gap_amp × Gap_rt  r =", round(r_facil_amp_rt$estimate, 3),
    "p =", round(r_facil_amp_rt$p.value, 4), "\n")
cat("Raw:      Gap_amp × Gap_rt                    r =", round(r_gap_amp_rt$estimate, 3),
    "p =", round(r_gap_amp_rt$p.value, 4), "\n")
cat("Relative:Gap_lat × Gap_rt  r =", round(r_facil_lat_rt$estimate, 3),
    "p =", round(r_facil_lat_rt$p.value, 4), "\n")
cat("Raw:      Gap_lat × Gap_rt                    r =", round(r_gap_lat_rt$estimate, 3),
    "p =", round(r_gap_lat_rt$p.value, 4), "\n")


cat("\n\n========== Q3 EXTENSION: RAW —Overlap CONDITION ==========\n")
cat("N (complete onOverlap_rt +Overlap_amp +Overlap_lat):", nrow(data_q3_raw_diseng), "\n")

# ── Bivariate correlations ────────────────────────────────────────────────────
cat("\n-- Bivariate correlations (rawOverlap) --\n")
r_ovlp_amp_rt  <- cor.test(data_q3_raw_diseng$Overlap_amp, data_q3_raw_diseng$Overlap_rt,  method = "pearson")
r_ovlp_lat_rt  <- cor.test(data_q3_raw_diseng$Overlap_lat, data_q3_raw_diseng$Overlap_rt,  method = "pearson")
r_ovlp_amp_lat <- cor.test(data_q3_raw_diseng$Overlap_amp, data_q3_raw_diseng$Overlap_lat, method = "pearson")
cat("Overlap_amp ×Overlap_rt:  r =", round(r_ovlp_amp_rt$estimate,  3), "p =", round(r_ovlp_amp_rt$p.value,  4), "\n")
cat("Overlap_lat ×Overlap_rt:  r =", round(r_ovlp_lat_rt$estimate,  3), "p =", round(r_ovlp_lat_rt$p.value,  4), "\n")
cat("Overlap_amp ×Overlap_lat: r =", round(r_ovlp_amp_lat$estimate, 3), "p =", round(r_ovlp_amp_lat$p.value, 4), "\n")

# ── Scatter plots ─────────────────────────────────────────────────────────────
p3_ovlp_amp <- ggplot(data_q3_raw_diseng, aes(x =Overlap_amp, y =Overlap_rt, color = ndd)) +
  geom_point(alpha = 0.5, size = 2) +
  geom_smooth(method = "lm", se = TRUE, color = "grey20", linewidth = 1, aes(group = 1)) +
  scale_color_manual(values = group_colors) +
  annotate("text", x = Inf, y = Inf, hjust = 1.1, vjust = 1.5,
           label = paste0("r = ", round(r_ovlp_amp_rt$estimate, 2),
                          ", p = ", round(r_ovlp_amp_rt$p.value, 4))) +
  labs(title = "Q3 Raw:Overlap — AMP × RT", x = "Overlap_amp", y = "Overlap_rt") + theme_pub

p3_ovlp_lat <- ggplot(data_q3_raw_diseng, aes(x =Overlap_lat, y =Overlap_rt, color = ndd)) +
  geom_point(alpha = 0.5, size = 2) +
  geom_smooth(method = "lm", se = TRUE, color = "grey20", linewidth = 1, aes(group = 1)) +
  scale_color_manual(values = group_colors) +
  annotate("text", x = Inf, y = Inf, hjust = 1.1, vjust = 1.5,
           label = paste0("r = ", round(r_ovlp_lat_rt$estimate, 2),
                          ", p = ", round(r_ovlp_lat_rt$p.value, 4))) +
  labs(title = "Q3 Raw:Overlap — LAT × RT", x = "Overlap_lat", y = "Overlap_rt") + theme_pub

print(p3_ovlp_amp + p3_ovlp_lat)
ggsave("output/figures/Q3_scatter_raw_overlap.png", p3_ovlp_amp + p3_ovlp_lat,
       width = 14, height = 6, dpi = 300, bg = "white")

# ── Linear models: raw ERP →Overlap_rt above age ─────────────────────────────
cat("\n-- Linear models: rawOverlap ERP →Overlap_rt above age --\n")
lm3_ovlp_null <- lm(Overlap_rt ~ age + sex + site,                               data = data_q3_raw_diseng)
lm3_ovlp_amp  <- lm(Overlap_rt ~Overlap_amp + age + sex + site,                 data = data_q3_raw_diseng)
lm3_ovlp_lat  <- lm(Overlap_rt ~Overlap_lat + age + sex + site,                 data = data_q3_raw_diseng)
lm3_ovlp_both <- lm(Overlap_rt ~Overlap_amp +Overlap_lat + age + sex + site,   data = data_q3_raw_diseng)

cat("\nLRT Null vs AMP+LAT:\n"); print(anova(lm3_ovlp_null, lm3_ovlp_both))
cat("\nLRT AMP only vs AMP+LAT:\n"); print(anova(lm3_ovlp_amp, lm3_ovlp_both))
cat("\nLRT LAT only vs AMP+LAT:\n"); print(anova(lm3_ovlp_lat, lm3_ovlp_both))

aic_q3_ovlp <- data.frame(
  Model     = c("Null (age+sex+site)", "AMP only", "LAT only", "AMP + LAT"),
  AIC       = c(AIC(lm3_ovlp_null), AIC(lm3_ovlp_amp), AIC(lm3_ovlp_lat), AIC(lm3_ovlp_both)),
  Adj_R2    = c(summary(lm3_ovlp_null)$adj.r.squared, summary(lm3_ovlp_amp)$adj.r.squared,
                summary(lm3_ovlp_lat)$adj.r.squared,  summary(lm3_ovlp_both)$adj.r.squared)
)
aic_q3_ovlp$Delta_AIC <- aic_q3_ovlp$AIC - min(aic_q3_ovlp$AIC)
print(aic_q3_ovlp)
cat("\nBest model (AMP + LAT) summary:\n"); print(summary(lm3_ovlp_both))
cat("\nVIF:\n"); print(car::vif(lm3_ovlp_both))
write.csv(aic_q3_ovlp, "output/tables/Q3_AIC_raw_overlap.csv", row.names = FALSE)

# ── NDD × raw ERP interaction —Overlap ───────────────────────────────────────
cat("\n-- NDD × raw ERP interaction —Overlap --\n")
lm3_ovlp_int <- lm(Overlap_rt ~ ndd *Overlap_amp + ndd *Overlap_lat + age + sex + site,
                   data = data_q3_raw_diseng)
cat("LRT: NDD×ERP vs ERP only:\n"); print(anova(lm3_ovlp_both, lm3_ovlp_int))
cat("Summary:\n"); print(summary(lm3_ovlp_int))

slope_data_ovlp <- data_q3_raw_diseng %>%
  group_by(ndd) %>%
  summarise(r_amp = cor(Overlap_amp,Overlap_rt, use = "complete.obs"),
            r_lat = cor(Overlap_lat,Overlap_rt, use = "complete.obs"),
            n = n(), .groups = "drop")
cat("RawOverlap ERP–RT correlation by NDD group:\n"); print(slope_data_ovlp)

# ── Comparison: relative vs raw coupling — Overlap ──────────────────────
cat("\n-- Relative vs Raw coupling (Overlap condition) --\n")
cat("Relative: Overlap_amp × Overlap_rt  r =", round(r_diseng_amp_rt$estimate, 3),
    "p =", round(r_diseng_amp_rt$p.value, 4), "\n")
cat("Raw:     Overlap_amp ×Overlap_rt              r =", round(r_ovlp_amp_rt$estimate, 3),
    "p =", round(r_ovlp_amp_rt$p.value, 4), "\n")
cat("Relative: Overlap_lat × Overlap_rt  r =", round(r_diseng_lat_rt$estimate, 3),
    "p =", round(r_diseng_lat_rt$p.value, 4), "\n")
cat("Raw:     Overlap_lat ×Overlap_rt              r =", round(r_ovlp_lat_rt$estimate, 3),
    "p =", round(r_ovlp_lat_rt$p.value, 4), "\n")


# =============================================================================
# Q3 EXTENSION: EXTENDED FULL CORRELATION MATRIX
# Relative measures + raw condition measures
# =============================================================================

data_fullcor_ext <- data %>%
  select(
    Gap_amp, Overlap_amp,
   Gap_lat, Overlap_lat,
    Gap_rt,  
   Overlap_amp,Overlap_lat,Overlap_rt,facilitation_rt ,disengagement_rt ,
  ) %>%
  na.omit()

cat("\nN for extended correlation matrix:", nrow(data_fullcor_ext), "\n")
fullcor_ext_mat <- cor(data_fullcor_ext)
cat("\nExtended full correlation matrix:\n"); print(round(fullcor_ext_mat, 3))
write.csv(round(as.data.frame(fullcor_ext_mat), 3),
          "output/tables/Q3_extended_correlation_matrix.csv")

# ── BH-FDR across all 12-variable pairwise correlations ──────────────────────
# Extract upper triangle p-values
cor_vars <- names(data_fullcor_ext)
n_cor    <- nrow(data_fullcor_ext)
cor_pairs <- data.frame(
  var1 = character(), var2 = character(),
  r = numeric(), p = numeric(),
  stringsAsFactors = FALSE
)
for (i in seq_along(cor_vars)) {
  for (j in seq(i + 1, length(cor_vars))) {
    ct <- cor.test(data_fullcor_ext[[cor_vars[i]]], data_fullcor_ext[[cor_vars[j]]])
    cor_pairs <- rbind(cor_pairs,
                       data.frame(var1 = cor_vars[i], var2 = cor_vars[j],
                                  r = round(ct$estimate, 3), p = round(ct$p.value, 4),
                                  stringsAsFactors = FALSE))
  }
}
cor_pairs$p_FDR <- round(p.adjust(cor_pairs$p, method = "BH"), 4)
cor_pairs$sig   <- ifelse(cor_pairs$p_FDR < .001, "***",
                          ifelse(cor_pairs$p_FDR < .01,  "**",
                                 ifelse(cor_pairs$p_FDR < .05,  "*", "ns")))
cor_pairs <- cor_pairs[order(cor_pairs$p_FDR), ]
cat("\nExtended correlation table (BH-FDR corrected, sorted by FDR p):\n")
print(cor_pairs)
write.csv(cor_pairs, "output/tables/Q3_extended_correlations_BH_FDR.csv", row.names = FALSE)




# =============================================================================
# Q3: ERP → RT (NEURAL-BEHAVIORAL COUPLING)
# =============================================================================

data_q3_facil <- data %>%
  filter(!is.na(Gap_rt), !is.na(Gap_amp), !is.na(Gap_lat))
data_q3_diseng <- data %>%
  filter(!is.na(Overlap_rt), !is.na(Overlap_amp), !is.na(Overlap_lat))

cat("\n\n========== Q3:  Gap ==========\n")
cat("N (complete on RT + AMP + LAT):", nrow(data_q3_facil), "\n")

cat("\n-- Bivariate correlations --\n")
r_facil_amp_rt  <- cor.test(data_q3_facil$Gap_amp, data_q3_facil$Gap_rt,  method="pearson")
r_facil_lat_rt  <- cor.test(data_q3_facil$Gap_lat, data_q3_facil$Gap_rt,  method="pearson")
r_facil_amp_lat <- cor.test(data_q3_facil$Gap_amp, data_q3_facil$Gap_lat, method="pearson")
cat("Gap_amp × Gap_rt:  r =", round(r_facil_amp_rt$estimate,  3), "p =", round(r_facil_amp_rt$p.value,  4), "\n")
cat("Gap_lat × Gap_rt:  r =", round(r_facil_lat_rt$estimate,  3), "p =", round(r_facil_lat_rt$p.value,  4), "\n")
cat("Gap_amp ×Gap_lat: r =", round(r_facil_amp_lat$estimate, 3), "p =", round(r_facil_amp_lat$p.value, 4), "\n")

p3_facil_amp <- ggplot(data_q3_facil, aes(x=Gap_amp, y=Gap_rt, color=ndd)) +
  geom_point(alpha=0.5, size=2) +
  geom_smooth(method="lm", se=TRUE, color="grey20", linewidth=1, aes(group=1)) +
  scale_color_manual(values=group_colors) +
  annotate("text", x=Inf, y=Inf, hjust=1.1, vjust=1.5,
           label=paste0("r = ", round(r_facil_amp_rt$estimate, 2), ", p = ", round(r_facil_amp_rt$p.value, 4))) +
  labs(title="Q3:  Gap — AMP × RT", x="Gap_amp", y="Gap_rt") + theme_pub

p3_facil_lat <- ggplot(data_q3_facil, aes(x=Gap_lat, y=Gap_rt, color=ndd)) +
  geom_point(alpha=0.5, size=2) +
  geom_smooth(method="lm", se=TRUE, color="grey20", linewidth=1, aes(group=1)) +
  scale_color_manual(values=group_colors) +
  annotate("text", x=Inf, y=Inf, hjust=1.1, vjust=1.5,
           label=paste0("r = ", round(r_facil_lat_rt$estimate, 2), ", p = ", round(r_facil_lat_rt$p.value, 4))) +
  labs(title="Q3:  Gap — LAT × RT", x="Gap_lat", y="Gap_rt") + theme_pub

print(p3_facil_amp + p3_facil_lat)
ggsave("output/figures/Q3_scatter_ Gap.png", p3_facil_amp + p3_facil_lat, width=14, height=6, dpi=300, bg="white")

cat("\n-- GAM: ERP → RT above age —  Gap --\n")
gam3_facil_null <- gam(Gap_rt ~ s(age, k=5) + site + s(fam_id, bs="re"),                                    data=data_q3_facil, method="REML")
gam3_facil_amp  <- gam(Gap_rt ~ Gap_amp + s(age, k=5) + site + s(fam_id, bs="re"),                 data=data_q3_facil, method="REML")
gam3_facil_lat  <- gam(Gap_rt ~Gap_lat + s(age, k=5) + site + s(fam_id, bs="re"),                 data=data_q3_facil, method="REML")
gam3_facil_both <- gam(Gap_rt ~ Gap_amp +Gap_lat + s(age, k=5) + site + s(fam_id, bs="re"), data=data_q3_facil, method="REML")

aic_q3_facil <- data.frame(
  Model   = c("Null (age only)", "AMP only", "LAT only", "AMP + LAT"),
  AIC     = c(AIC(gam3_facil_null), AIC(gam3_facil_amp), AIC(gam3_facil_lat), AIC(gam3_facil_both)),
  DevExpl = c(summary(gam3_facil_null)$dev.expl, summary(gam3_facil_amp)$dev.expl,
              summary(gam3_facil_lat)$dev.expl,  summary(gam3_facil_both)$dev.expl)
)
aic_q3_facil$Delta_AIC <- aic_q3_facil$AIC - min(aic_q3_facil$AIC)
print(aic_q3_facil)
cat("\nLRT: AMP+LAT vs Null:\n"); print(anova(gam3_facil_null, gam3_facil_both, test="Chisq"))
cat("\nBest model summary:\n");    print(summary(gam3_facil_both))
lm_vif_facil <- lm(Gap_rt ~ Gap_amp +Gap_lat + age + site, data=data_q3_facil)
cat("VIF:\n"); print(car::vif(lm_vif_facil))
write.csv(aic_q3_facil, "output/tables/Q3_AIC_ Gap.csv", row.names=FALSE)

cat("\n-- NDD × ERP interaction —  Gap --\n")
gam3_facil_int <- gam(Gap_rt ~ ndd * Gap_amp + ndd *Gap_lat + s(age, k=5) + site + s(fam_id, bs="re"), data=data_q3_facil, method="REML")
cat("LRT: NDD×ERP vs ERP only:\n"); print(anova(gam3_facil_both, gam3_facil_int, test="Chisq"))
cat("Summary:\n"); print(summary(gam3_facil_int))
slope_data_facil <- data_q3_facil %>%
  group_by(ndd) %>%
  summarise(r_amp = cor(Gap_amp, Gap_rt, use="complete.obs"),
            r_lat = cor(Gap_lat, Gap_rt, use="complete.obs"),
            n = n(), .groups="drop")
cat("ERP–RT correlation by NDD group:\n"); print(slope_data_facil)

cat("\n-- Cross-construct specificity:  Gap AMP × Overlap RT --\n")
data_cross_facil <- data %>% filter(!is.na(Gap_amp), !is.na(Overlap_rt))
r_cross_facil    <- cor.test(data_cross_facil$Gap_amp, data_cross_facil$Overlap_rt)
cat("Gap_amp × Overlap_rt (cross): r =", round(r_cross_facil$estimate, 3),
    "p =", round(r_cross_facil$p.value, 4), "\n")
cat("Matched r (Gap_amp × Gap_rt) =", round(r_facil_amp_rt$estimate, 3), "\n")


cat("\n\n========== Q3: Overlap ==========\n")
cat("N (complete on RT + AMP + LAT):", nrow(data_q3_diseng), "\n")

cat("\n-- Bivariate correlations --\n")
r_diseng_amp_rt  <- cor.test(data_q3_diseng$Overlap_amp, data_q3_diseng$Overlap_rt,  method="pearson")
r_diseng_lat_rt  <- cor.test(data_q3_diseng$Overlap_lat, data_q3_diseng$Overlap_rt,  method="pearson")
r_diseng_amp_lat <- cor.test(data_q3_diseng$Overlap_amp, data_q3_diseng$Overlap_lat, method="pearson")
cat("Overlap_amp × Overlap_rt:  r =", round(r_diseng_amp_rt$estimate,  3), "p =", round(r_diseng_amp_rt$p.value,  4), "\n")
cat("Overlap_lat × Overlap_rt:  r =", round(r_diseng_lat_rt$estimate,  3), "p =", round(r_diseng_lat_rt$p.value,  4), "\n")
cat("Overlap_amp × Overlap_lat: r =", round(r_diseng_amp_lat$estimate, 3), "p =", round(r_diseng_amp_lat$p.value, 4), "\n")

p3_diseng_amp <- ggplot(data_q3_diseng, aes(x=Overlap_amp, y=Overlap_rt, color=ndd)) +
  geom_point(alpha=0.5, size=2) +
  geom_smooth(method="lm", se=TRUE, color="grey20", linewidth=1, aes(group=1)) +
  scale_color_manual(values=group_colors) +
  annotate("text", x=Inf, y=Inf, hjust=1.1, vjust=1.5,
           label=paste0("r = ", round(r_diseng_amp_rt$estimate, 2), ", p = ", round(r_diseng_amp_rt$p.value, 4))) +
  labs(title="Q3: Overlap — AMP × RT", x="Overlap_amp", y="Overlap_rt") + theme_pub

p3_diseng_lat <- ggplot(data_q3_diseng, aes(x=Overlap_lat, y=Overlap_rt, color=ndd)) +
  geom_point(alpha=0.5, size=2) +
  geom_smooth(method="lm", se=TRUE, color="grey20", linewidth=1, aes(group=1)) +
  scale_color_manual(values=group_colors) +
  annotate("text", x=Inf, y=Inf, hjust=1.1, vjust=1.5,
           label=paste0("r = ", round(r_diseng_lat_rt$estimate, 2), ", p = ", round(r_diseng_lat_rt$p.value, 4))) +
  labs(title="Q3: Overlap — LAT × RT", x="Overlap_lat", y="Overlap_rt") + theme_pub

print(p3_diseng_amp + p3_diseng_lat)
ggsave("output/figures/Q3_scatter_Overlap.png", p3_diseng_amp + p3_diseng_lat, width=14, height=6, dpi=300, bg="white")

cat("\n-- GAM: ERP → RT above age — Overlap --\n")
gam3_diseng_null <- gam(Overlap_rt ~ s(age, k=5) + site + s(fam_id, bs="re"),                                          data=data_q3_diseng, method="REML")
gam3_diseng_amp  <- gam(Overlap_rt ~ Overlap_amp + s(age, k=5) + site + s(fam_id, bs="re"),                     data=data_q3_diseng, method="REML")
gam3_diseng_lat  <- gam(Overlap_rt ~ Overlap_lat + s(age, k=5) + site + s(fam_id, bs="re"),                     data=data_q3_diseng, method="REML")
gam3_diseng_both <- gam(Overlap_rt ~ Overlap_amp + Overlap_lat + s(age, k=5) + site + s(fam_id, bs="re"), data=data_q3_diseng, method="REML")

aic_q3_diseng <- data.frame(
  Model   = c("Null (age only)", "AMP only", "LAT only", "AMP + LAT"),
  AIC     = c(AIC(gam3_diseng_null), AIC(gam3_diseng_amp), AIC(gam3_diseng_lat), AIC(gam3_diseng_both)),
  DevExpl = c(summary(gam3_diseng_null)$dev.expl, summary(gam3_diseng_amp)$dev.expl,
              summary(gam3_diseng_lat)$dev.expl,  summary(gam3_diseng_both)$dev.expl)
)
aic_q3_diseng$Delta_AIC <- aic_q3_diseng$AIC - min(aic_q3_diseng$AIC)
print(aic_q3_diseng)
cat("\nLRT: AMP+LAT vs Null:\n"); print(anova(gam3_diseng_null, gam3_diseng_both, test="Chisq"))
cat("\nBest model summary:\n");    print(summary(gam3_diseng_both))
lm_vif_diseng <- lm(Overlap_rt ~ Overlap_amp + Overlap_lat + age + site, data=data_q3_diseng)
cat("VIF:\n"); print(car::vif(lm_vif_diseng))
write.csv(aic_q3_diseng, "output/tables/Q3_AIC_Overlap.csv", row.names=FALSE)

cat("\n-- NDD × ERP interaction — Overlap --\n")
gam3_diseng_int <- gam(Overlap_rt ~ ndd * Overlap_amp + ndd * Overlap_lat + s(age, k=5) + site + s(fam_id, bs="re"), data=data_q3_diseng, method="REML")
cat("LRT: NDD×ERP vs ERP only:\n"); print(anova(gam3_diseng_both, gam3_diseng_int, test="Chisq"))
cat("Summary:\n"); print(summary(gam3_diseng_int))
slope_data_diseng <- data_q3_diseng %>%
  group_by(ndd) %>%
  summarise(r_amp = cor(Overlap_amp, Overlap_rt, use="complete.obs"),
            r_lat = cor(Overlap_lat, Overlap_rt, use="complete.obs"),
            n = n(), .groups="drop")
cat("ERP–RT correlation by NDD group:\n"); print(slope_data_diseng)

cat("\n-- Cross-construct specificity: Overlap AMP ×  Gap RT --\n")
data_cross_diseng <- data %>% filter(!is.na(Overlap_amp), !is.na(Gap_rt))
r_cross_diseng    <- cor.test(data_cross_diseng$Overlap_amp, data_cross_diseng$Gap_rt)
cat("Overlap_amp × Gap_rt (cross): r =", round(r_cross_diseng$estimate, 3),
    "p =", round(r_cross_diseng$p.value, 4), "\n")
cat("Matched r (Overlap_amp × Overlap_rt) =", round(r_diseng_amp_rt$estimate, 3), "\n")

data_fullcor <- data %>%
  select(Gap_amp, Overlap_amp,Gap_lat, Overlap_lat,
         Gap_rt, Overlap_rt) %>%
  na.omit()
fullcor_mat <- cor(data_fullcor)
write.csv(round(as.data.frame(fullcor_mat), 3), "output/tables/Q3_full_correlation_matrix.csv")
cat("\nFull correlation matrix:\n"); print(round(fullcor_mat, 3))

 