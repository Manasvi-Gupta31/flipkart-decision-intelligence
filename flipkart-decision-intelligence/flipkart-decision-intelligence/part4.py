"""Part 4: split purity, model combination, AdaBoost by hand, network sizing."""
import math

import numpy as np

from common import (Results, choose_k, confusion, fit_logistic_main, get_split, knn_predict,
                    metrics, sigmoid, write_csv)

sp = get_split()
X, Xtr, Xte, ytr, yte, tr, te = sp["X"], sp["Xtr"], sp["Xte"], sp["ytr"], sp["yte"], sp["tr"], sp["te"]
R = Results()


def entropy(pos, n):
    if n == 0:
        return 0.0
    e = 0.0
    for c in (pos, n - pos):
        if c > 0:
            q = c / n
            e -= q * math.log2(q)
    return e


def gini(pos, n):
    if n == 0:
        return 0.0
    q = pos / n
    return 1 - q * q - (1 - q) ** 2


# 4.1  (original units, training rows)
Xo = X[tr]
n = len(ytr)
P = int(ytr.sum())
R.add("root_entropy", entropy(P, n))
R.add("root_gini", gini(P, n))
splits = [("S1", Xo[:, 5] == 1), ("S2", Xo[:, 6] == 1), ("S3", Xo[:, 2] > 0.5), ("S4", Xo[:, 0] <= 24)]
rows, gi, gg = [], {}, {}
for name, yes in splits:
    ny, py = int(yes.sum()), int(ytr[yes].sum())
    nn, pn = n - ny, P - py
    ig = entropy(P, n) - (ny / n * entropy(py, ny) + nn / n * entropy(pn, nn))
    gn = gini(P, n) - (ny / n * gini(py, ny) + nn / n * gini(pn, nn))
    gi[name], gg[name] = ig, gn
    rows.append([name, ny, py, nn, pn, ig, gn])
write_csv("part4_splits.csv", ["split", "n_yes", "positives_yes", "n_no", "positives_no",
                              "info_gain", "gini_gain"], rows)
best_ig = max(gi, key=lambda k: (gi[k], -int(k[1])))
best_gg = max(gg, key=lambda k: (gg[k], -int(k[1])))
R.add("best_split_by_info_gain", best_ig)
R.add("best_split_by_gini_gain", best_gg)

# 4.2  stump = best split by information gain
yes_tr = dict(splits)[best_ig]
rate_yes = float(ytr[yes_tr].mean())
rate_no = float(ytr[~yes_tr].mean())
R.add("stump_rate_yes", rate_yes)
R.add("stump_rate_no", rate_no)
Xo_te = X[te]
cond = {"S1": lambda a: a[:, 5] == 1, "S2": lambda a: a[:, 6] == 1,
        "S3": lambda a: a[:, 2] > 0.5, "S4": lambda a: a[:, 0] <= 24}[best_ig]
p_stump = np.where(cond(Xo_te), rate_yes, rate_no)

w, b, _ = fit_logistic_main(sp)
p_log = sigmoid(Xte @ w + b)
_, K = choose_k(Xtr, ytr)
share, knn_pos = knn_predict(Xtr, ytr, Xte, K)

TAU = 0.3
pred = {
    "logistic": p_log >= TAU - 1e-12,
    "knn": knn_pos,
    "stump": p_stump >= TAU - 1e-12,
}
votes = pred["logistic"].astype(int) + pred["knn"].astype(int) + pred["stump"].astype(int)
pred["hard_vote"] = votes >= 2
pred["soft_vote"] = (p_log + share + p_stump) / 3 >= TAU - 1e-12
vrows = []
for name in ("logistic", "knn", "stump", "hard_vote", "soft_vote"):
    tp, fp, tn, fn = confusion(yte, pred[name])
    m = metrics(tp, fp, tn, fn)
    vrows.append([name, tp, fp, tn, fn, m["precision"], m["recall"], m["f1"]])
    # extra (not required): test cost of each model with the Task 2.8 rule
    R.add("extra_cost_test_" + name, 50 * fp + 400 * fn)
write_csv("part4_voting.csv", ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"], vrows)

# 4.3  AdaBoost by hand
y8 = np.array([1, 1, 1, -1, -1, -1, 1, -1])
h = [np.array([1, 1, -1, -1, -1, -1, -1, -1]), np.array([-1, 1, 1, -1, 1, -1, 1, -1])]
wts = np.full(8, 1 / 8)
ab_rows, alphas = [], []
for r in range(2):
    eps = float(wts[h[r] != y8].sum())
    alpha = 0.5 * math.log((1 - eps) / eps)
    unnorm = wts * np.exp(-alpha * y8 * h[r])
    after = unnorm / unnorm.sum()
    for i in range(8):
        ab_rows.append([r + 1, i + 1, float(wts[i]), float(unnorm[i]), float(after[i])])
    R.add("round%d_error" % (r + 1), eps)
    R.add("round%d_alpha" % (r + 1), alpha)
    alphas.append(alpha)
    wts = after
write_csv("part4_adaboost.csv", ["round", "point", "weight_before", "weight_unnormalised",
                                 "weight_after"], ab_rows)
score = alphas[0] * h[0] + alphas[1] * h[1]
R.add("ensemble_accuracy", float(np.mean(np.sign(score) == y8)))

# 4.4
R.add("oob_fraction", (1 - 1 / 560) ** 560)

# 4.5
mlp = [784 * 16 + 16, 16 * 16 + 16, 16 * 10 + 10]
c1 = 5 * 5 * 1 * 6 + 6
s1 = (28 - 5 + 2 * 2) // 1 + 1           # 28
s1p = (s1 - 2) // 2 + 1                  # 14
c2 = 5 * 5 * 6 * 16 + 16
s2 = (s1p - 5 + 0) // 1 + 1              # 10
s2p = (s2 - 2) // 2 + 1                  # 5
flat = s2p * s2p * 16                    # 400
d1 = flat * 32 + 32
d2 = 32 * 10 + 10
cnn = [c1, c2, d1, d2]
names = ["mlp_hidden1", "mlp_hidden2", "mlp_output", "cnn_conv1", "cnn_conv2", "cnn_dense1", "cnn_dense2"]
write_csv("part4_architecture.csv", ["layer", "parameters"], [[a, b_] for a, b_ in zip(names, mlp + cnn)])
R.add("mlp_total", sum(mlp))
R.add("cnn_total", sum(cnn))
R.add("fc_layer_params", 128 * 128 * 3 * 1024 + 1024)
R.add("conv_layer_params", 3 * 3 * 3 * 64 + 64)
R.add("rnn_total", 128 * 64 + 128 * 128 + 16 * 128 + 128 + 16)
# shapes for the README (extra, not required)
R.add("extra_cnn_shapes", "%dx%dx6 -> %dx%dx6 -> %dx%dx16 -> %dx%dx16 -> %d" %
      (s1, s1, s1p, s1p, s2, s2, s2p, s2p, flat))

R.write("part4_results.txt")
