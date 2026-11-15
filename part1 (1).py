
# PharmEasy Healthcare Decision Intelligence
# Part 1: Estimation, Inference and Regression

import numpy as np
import pandas as pd
import math
from pathlib import Path

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "data" / "records.csv")

n = len(df)
results = {}

# --------------------------------------------------
# TASK 1.1: ESTIMATION
# --------------------------------------------------

Y = df["care_plan_cost_inr_thousands"].to_numpy()

mean_value = np.mean(Y)
sd_value = np.std(Y, ddof=1)
se_value = sd_value / np.sqrt(n)

ci95_low = mean_value - 1.96 * se_value
ci95_high = mean_value + 1.96 * se_value

results["n"] = n
results["mean_value"] = mean_value
results["sd_value"] = sd_value
results["se_value"] = se_value
results["ci95_low"] = ci95_low
results["ci95_high"] = ci95_high

# --------------------------------------------------
# TASK 1.2: Z-TEST
# --------------------------------------------------

hypothesized_mean = 8.0
z_stat = (mean_value - hypothesized_mean) / se_value
reject_h0 = "yes" if abs(z_stat) > 1.96 else "no"

results["z_stat"] = z_stat
results["reject_h0"] = reject_h0

# --------------------------------------------------
# TASK 1.3: SAMPLE SIZE
# --------------------------------------------------

E = 0.5
n_required = math.ceil((1.96 * sd_value / E) ** 2)

results["n_required"] = n_required

# --------------------------------------------------
# TASK 1.4: SIMPLE LINEAR REGRESSION
# --------------------------------------------------

X = df["avg_bill_inr_thousands"].to_numpy()

X_mean = np.mean(X)
Y_mean = np.mean(Y)

reg_slope = np.sum((X - X_mean) * (Y - Y_mean)) / np.sum((X - X_mean) ** 2)
reg_intercept = Y_mean - reg_slope * X_mean

fitted = reg_slope * X + reg_intercept

tss = np.sum((Y - Y_mean) ** 2)
ess = np.sum((fitted - Y_mean) ** 2)
rss = np.sum((Y - fitted) ** 2)

tss_equals_ess_plus_rss = (
    "yes" if abs(tss - ess - rss) < 1e-6 * tss else "no"
)

r2 = 1 - rss / tss
rmse = np.sqrt(rss / n)
mae = np.mean(np.abs(Y - fitted))

results["reg_slope"] = reg_slope
results["reg_intercept"] = reg_intercept
results["tss"] = tss
results["ess"] = ess
results["rss"] = rss
results["tss_equals_ess_plus_rss"] = tss_equals_ess_plus_rss
results["r2"] = r2
results["rmse"] = rmse
results["mae"] = mae

# --------------------------------------------------
# TASK 1.5: GRADIENT DESCENT
# --------------------------------------------------

X_sd = np.std(X, ddof=1)
X_std = (X - X_mean) / X_sd

m = 0.0
c = 0.0
learning_rate = 0.1

# Initial cost before updates
gd_J0 = np.mean((Y - (m * X_std + c)) ** 2)

cost_values = {}

for i in range(1, 501):
    Y_pred = m * X_std + c
    error = Y - Y_pred

    dm = (-2 / n) * np.sum(X_std * error)
    dc = (-2 / n) * np.sum(error)

    m = m - learning_rate * dm
    c = c - learning_rate * dc

    J = np.mean((Y - (m * X_std + c)) ** 2)

    if i in [1, 10, 100, 500]:
        cost_values[i] = J

gd_slope = m / X_sd
gd_intercept = c - (m * X_mean / X_sd)

gd_slope_gap = abs(gd_slope - reg_slope)

results["gd_J0"] = gd_J0
results["gd_J1"] = cost_values[1]
results["gd_J10"] = cost_values[10]
results["gd_J100"] = cost_values[100]
results["gd_J500"] = cost_values[500]
results["gd_slope"] = gd_slope
results["gd_intercept"] = gd_intercept
results["gd_slope_gap"] = gd_slope_gap

# --------------------------------------------------
# TASK 1.6: RESIDUAL FUNNEL AND LOG TRANSFORMATION
# --------------------------------------------------

residuals = Y - fitted

sorted_indices = np.argsort(fitted, kind="stable")
lower_indices = sorted_indices[:400]
upper_indices = sorted_indices[400:]

resid_sd_lower_raw = np.std(residuals[lower_indices], ddof=1)
resid_sd_upper_raw = np.std(residuals[upper_indices], ddof=1)
resid_sd_ratio_raw = resid_sd_upper_raw / resid_sd_lower_raw

# Log-log regression
log_X = np.log(X)
log_Y = np.log(Y)

log_X_mean = np.mean(log_X)
log_Y_mean = np.mean(log_Y)

loglog_slope = (
    np.sum((log_X - log_X_mean) * (log_Y - log_Y_mean))
    / np.sum((log_X - log_X_mean) ** 2)
)

loglog_intercept = log_Y_mean - loglog_slope * log_X_mean
log_fitted = loglog_intercept + loglog_slope * log_X

log_tss = np.sum((log_Y - log_Y_mean) ** 2)
log_rss = np.sum((log_Y - log_fitted) ** 2)

loglog_r2 = 1 - log_rss / log_tss
loglog_rmse = np.sqrt(log_rss / n)

log_residuals = log_Y - log_fitted

log_sorted_indices = np.argsort(log_fitted, kind="stable")
log_lower_indices = log_sorted_indices[:400]
log_upper_indices = log_sorted_indices[400:]

resid_sd_lower_log = np.std(log_residuals[log_lower_indices], ddof=1)
resid_sd_upper_log = np.std(log_residuals[log_upper_indices], ddof=1)
resid_sd_ratio_log = resid_sd_upper_log / resid_sd_lower_log

results["resid_sd_lower_raw"] = resid_sd_lower_raw
results["resid_sd_upper_raw"] = resid_sd_upper_raw
results["resid_sd_ratio_raw"] = resid_sd_ratio_raw
results["loglog_intercept"] = loglog_intercept
results["loglog_slope"] = loglog_slope
results["loglog_r2"] = loglog_r2
results["loglog_rmse"] = loglog_rmse
results["resid_sd_lower_log"] = resid_sd_lower_log
results["resid_sd_upper_log"] = resid_sd_upper_log
results["resid_sd_ratio_log"] = resid_sd_ratio_log

# --------------------------------------------------
# TASK 1.7: MULTIPLE LINEAR REGRESSION
# --------------------------------------------------

X1 = df["months_registered"].to_numpy()
X2 = df["missed_visit_count_12m"].to_numpy()
X3 = df["copay_burden_ratio"].to_numpy()
X4_log = np.log(df["avg_bill_inr_thousands"].to_numpy())
X5 = df["adherence_score"].to_numpy()
X6 = df["chronic_condition_flag"].to_numpy()
X7 = df["insured_flag"].to_numpy()

A = np.column_stack((
    np.ones(n),
    X4_log,
    X1,
    X2,
    X3,
    X5,
    X6,
    X7
))

beta = np.linalg.solve(A.T @ A, A.T @ log_Y)

Y_multi_pred = A @ beta

mr_tss = np.sum((log_Y - np.mean(log_Y)) ** 2)
mr_rss = np.sum((log_Y - Y_multi_pred) ** 2)

mr_r2 = 1 - mr_rss / mr_tss

p = 7
mr_adj_r2 = 1 - (1 - mr_r2) * (n - 1) / (n - p - 1)

loglog_adj_r2 = 1 - (1 - loglog_r2) * (n - 1) / (n - 2)

for i in range(8):
    results[f"mr_beta_{i}"] = beta[i]

results["mr_r2"] = mr_r2
results["mr_adj_r2"] = mr_adj_r2
results["loglog_adj_r2"] = loglog_adj_r2

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

output_dir = BASE_DIR / "outputs"
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "part1_results.txt"

with open(output_file, "w") as f:
    for key, value in results.items():
        if isinstance(value, (int, np.integer)):
            f.write(f"{key} = {int(value)}\n")
        elif isinstance(value, str):
            f.write(f"{key} = {value}\n")
        else:
            f.write(f"{key} = {float(value):.4f}\n")

print("Part 1 completed successfully!")
print("Total records:", n)
print("Total result keys:", len(results))
print("Results saved to:", output_file)
print("\nMULTIPLE REGRESSION SUMMARY")
print("mr_r2:", round(mr_r2, 4))
print("mr_adj_r2:", round(mr_adj_r2, 4))
