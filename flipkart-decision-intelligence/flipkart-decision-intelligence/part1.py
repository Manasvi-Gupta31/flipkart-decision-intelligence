"""Part 1: estimation, inference and regression baseline (all 800 rows)."""
import math

import numpy as np

from common import Results, load_data, sd, sd_cols

X, y, value = load_data()
n = len(value)
R = Results()
X4 = X[:, 3]

# 1.1
mean_value = float(value.mean())
sd_value = sd(value)
se_value = sd_value / math.sqrt(n)
R.add("n", n)
R.add("mean_value", mean_value)
R.add("sd_value", sd_value)
R.add("se_value", se_value)
R.add("ci95_low", mean_value - 1.96 * se_value)
R.add("ci95_high", mean_value + 1.96 * se_value)

# 1.2
z_stat = (mean_value - 8.0) / se_value
R.add("z_stat", z_stat)
R.add("reject_h0", abs(z_stat) > 1.96)

# 1.3
E = 0.5
n_required = int(math.ceil((1.96 * sd_value / E) ** 2 - 1e-12))
R.add("n_required", n_required)


def simple_ols(x, yy):
    mx, my = x.mean(), yy.mean()
    cov = np.sum((x - mx) * (yy - my)) / (len(x) - 1)
    var = np.sum((x - mx) ** 2) / (len(x) - 1)
    slope = cov / var
    return slope, my - slope * mx


# 1.4
reg_slope, reg_intercept = simple_ols(X4, value)
fitted = reg_intercept + reg_slope * X4
tss = float(np.sum((value - value.mean()) ** 2))
ess = float(np.sum((fitted - value.mean()) ** 2))
resid = value - fitted
rss = float(np.sum(resid ** 2))
R.add("reg_slope", reg_slope)
R.add("reg_intercept", reg_intercept)
R.add("tss", tss)
R.add("ess", ess)
R.add("rss", rss)
R.add("tss_equals_ess_plus_rss", abs(tss - ess - rss) < 1e-6 * tss)
R.add("r2", 1 - rss / tss)
R.add("rmse", math.sqrt(rss / n))
R.add("mae", float(np.mean(np.abs(resid))))

# 1.5
mx, sx = X4.mean(), sd(X4)
z = (X4 - mx) / sx
m, c = 0.0, 0.0
eta = 0.1
Js = {}
for t in range(0, 501):
    e = value - (m * z + c)
    if t in (0, 1, 10, 100, 500):
        Js[t] = float(np.mean(e ** 2))
    if t == 500:
        break
    gm = -2.0 / n * np.sum(z * e)
    gc = -2.0 / n * np.sum(e)
    m -= eta * gm
    c -= eta * gc
for t, name in ((0, "gd_J0"), (1, "gd_J1"), (10, "gd_J10"), (100, "gd_J100"), (500, "gd_J500")):
    R.add(name, Js[t])
gd_slope = m / sx
gd_intercept = c - m * mx / sx
R.add("gd_slope", gd_slope)
R.add("gd_intercept", gd_intercept)
R.add("gd_slope_gap", abs(gd_slope - reg_slope))


# 1.6
def halves_sd(fit, res):
    order = np.argsort(fit, kind="stable")
    lo, hi = res[order[:400]], res[order[400:]]
    return sd(lo), sd(hi)


lo, hi = halves_sd(fitted, resid)
R.add("resid_sd_lower_raw", lo)
R.add("resid_sd_upper_raw", hi)
R.add("resid_sd_ratio_raw", hi / lo)

lx, ly = np.log(X4), np.log(value)
ll_slope, ll_int = simple_ols(lx, ly)
ll_fit = ll_int + ll_slope * lx
ll_res = ly - ll_fit
ll_rss = float(np.sum(ll_res ** 2))
ll_tss = float(np.sum((ly - ly.mean()) ** 2))
ll_r2 = 1 - ll_rss / ll_tss
R.add("loglog_intercept", ll_int)
R.add("loglog_slope", ll_slope)
R.add("loglog_r2", ll_r2)
R.add("loglog_rmse", math.sqrt(ll_rss / n))
lo, hi = halves_sd(ll_fit, ll_res)
R.add("resid_sd_lower_log", lo)
R.add("resid_sd_upper_log", hi)
R.add("resid_sd_ratio_log", hi / lo)

# 1.7
D = np.column_stack([np.ones(n), np.log(X4), X[:, 0], X[:, 1], X[:, 2], X[:, 4], X[:, 5], X[:, 6]])
beta = np.linalg.solve(D.T @ D, D.T @ ly)
for i, b in enumerate(beta):
    R.add("mr_beta_%d" % i, float(b))
mr_res = ly - D @ beta
mr_r2 = 1 - float(np.sum(mr_res ** 2)) / ll_tss
p = 7
R.add("mr_r2", mr_r2)
R.add("mr_adj_r2", 1 - (1 - mr_r2) * (n - 1) / (n - p - 1))
R.add("loglog_adj_r2", 1 - (1 - ll_r2) * (n - 1) / (n - 1 - 1))

R.write("part1_results.txt")
