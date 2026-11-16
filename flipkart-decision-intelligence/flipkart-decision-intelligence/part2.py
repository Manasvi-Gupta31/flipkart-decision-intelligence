"""Part 2: classification and evaluation (logistic regression, KNN, cost-based threshold)."""
import math

import numpy as np

from common import (Results, bce, choose_k, confusion, fit_logistic_main, get_split, knn_predict,
                    metrics, sigmoid, train_logistic, write_csv)

sp = get_split()
Xtr, Xte, ytr, yte = sp["Xtr"], sp["Xte"], sp["ytr"], sp["yte"]
R = Results()

# 2.1
R.add("train_rows", len(ytr))
R.add("train_positives", int(ytr.sum()))
R.add("test_rows", len(yte))
R.add("test_positives", int(yte.sum()))

# 2.2
w, b, L = fit_logistic_main(sp)
R.add("loss_epoch_0", L[0])
R.add("loss_epoch_1", L[1])
R.add("loss_epoch_10", L[10])
R.add("loss_epoch_100", L[100])
R.add("loss_epoch_1000", L[1000])
for j in range(7):
    R.add("weight_X%d" % (j + 1), float(w[j]))
R.add("bias", float(b))
p_te = sigmoid(Xte @ w + b)
p_tr = sigmoid(Xtr @ w + b)
R.add("test_bce", bce(yte, p_te))

# 2.3
for eta in (0.01, 0.1, 1.0):
    _, _, l = train_logistic(Xtr, ytr, eta, 200, record=(200,))
    R.add("loss_epoch_200_eta_%s" % ("%g" % eta if eta != 1.0 else "1.0"), l[200])

# 2.4
for j in range(7):
    R.add("odds_ratio_X%d" % (j + 1), math.exp(w[j]))

# 2.5
for tau in (0.5, 0.3):
    tp, fp, tn, fn = confusion(yte, p_te >= tau)
    m = metrics(tp, fp, tn, fn)
    pre = "tau_%s_" % tau
    R.add(pre + "tp", tp)
    R.add(pre + "fp", fp)
    R.add(pre + "tn", tn)
    R.add(pre + "fn", fn)
    for k in ("accuracy", "precision", "recall", "f1", "specificity"):
        R.add(pre + k, m[k])
R.add("blind_accuracy", float(np.mean(yte == 0)))

# 2.6
rows = []
for k in range(1, 10):
    tau = k / 10
    tp, fp, tn, fn = confusion(yte, p_te >= tau - 1e-12)
    m = metrics(tp, fp, tn, fn)
    rows.append([tau, tp, fp, tn, fn, m["recall"], m["fpr"]])
write_csv("part2_thresholds.csv", ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], rows)

# 2.7  5-fold CV for K (routine in common.py, shared with Part 4)
cv_rows, best_k = choose_k(Xtr, ytr)
write_csv("part2_knn_cv.csv", ["K", "mean_cv_f1"], [list(r) for r in cv_rows])
R.add("knn_best_k", best_k)
_, kpred = knn_predict(Xtr, ytr, Xte, best_k)
tp, fp, tn, fn = confusion(yte, kpred)
m = metrics(tp, fp, tn, fn)
for k, v in (("tp", tp), ("fp", fp), ("tn", tn), ("fn", fn)):
    R.add("knn_" + k, v)
for k in ("accuracy", "precision", "recall", "f1"):
    R.add("knn_" + k, m[k])

# 2.8
def cost(y, pred):
    tp, fp, tn, fn = confusion(y, pred)
    return 50 * fp + 400 * fn

best_tau, best_cost = None, None
for k in range(1, 20):
    tau = k / 20
    c = cost(ytr, p_tr >= tau - 1e-12)
    if best_cost is None or c < best_cost:
        best_tau, best_cost = tau, c
R.add("cost_tau", best_tau)
R.add("cost_train_at_tau", best_cost)
R.add("cost_test_at_tau", cost(yte, p_te >= best_tau - 1e-12))
R.add("cost_test_at_0.5", cost(yte, p_te >= 0.5))
R.add("cost_test_flag_nobody", cost(yte, np.zeros(len(yte), bool)))

R.write("part2_results.txt")
