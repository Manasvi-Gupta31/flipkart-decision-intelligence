"""Shared helpers: data loading, formatting, split, standardisation, logistic regression, metrics.
Importing this module runs nothing."""
import csv
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")


def load_data():
    with open(os.path.join(ROOT, "data", "records.csv")) as f:
        rd = csv.reader(f)
        next(rd)
        rows = list(rd)
    X = np.array([[float(v) for v in r[1:8]] for r in rows])
    y = np.array([int(r[8]) for r in rows])
    value = np.array([float(r[9]) for r in rows])
    return X, y, value


def fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        s = "%.4f" % float(v)
        return "0.0000" if s == "-0.0000" else s
    return str(v)


class Results:
    def __init__(self):
        self.items = []

    def add(self, key, value):
        self.items.append((key, value))

    def write(self, name):
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, name), "w") as f:
            for k, v in self.items:
                f.write("%s = %s\n" % (k, fmt(v)))

    def get(self, key):
        return dict(self.items)[key]


def write_csv(name, header, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(fmt(v) for v in r) + "\n")


def sd(a):
    return float(np.sqrt(np.sum((a - a.mean(axis=0)) ** 2, axis=0) / (len(a) - 1)))


def sd_cols(a):
    return np.sqrt(np.sum((a - a.mean(axis=0)) ** 2, axis=0) / (len(a) - 1))


def stratified_split(y):
    """Indices of train and test rows (file order kept)."""
    train = []
    for cls in (0, 1):
        idx = [i for i in range(len(y)) if y[i] == cls]
        k = (7 * len(idx)) // 10
        train += idx[:k]
    train = np.array(sorted(train))
    mask = np.ones(len(y), bool)
    mask[train] = False
    test = np.where(mask)[0]
    return train, test


def get_split():
    X, y, value = load_data()
    tr, te = stratified_split(y)
    mu = X[tr].mean(axis=0)
    s = sd_cols(X[tr])
    Xtr = (X[tr] - mu) / s
    Xte = (X[te] - mu) / s
    return dict(X=X, y=y, value=value, tr=tr, te=te, Xtr=Xtr, Xte=Xte,
                ytr=y[tr], yte=y[te], mu=mu, s=s)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce(y, p):
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train_logistic(X, y, eta, updates, record=()):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    losses = {}
    if 0 in record:
        losses[0] = bce(y, sigmoid(X @ w + b))
    for t in range(1, updates + 1):
        p = sigmoid(X @ w + b)
        err = p - y
        gw = X.T @ err / n
        gb = err.mean()
        w = w - eta * gw
        b = b - eta * gb
        if t in record:
            losses[t] = bce(y, sigmoid(X @ w + b))
    return w, b, losses


def confusion(y, pred):
    y = np.asarray(y)
    pred = np.asarray(pred).astype(bool)
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    return tp, fp, tn, fn


def _div(a, b):
    return a / b if b else 0.0


def metrics(tp, fp, tn, fn):
    acc = _div(tp + tn, tp + fp + tn + fn)
    prec = _div(tp, tp + fp)
    rec = _div(tp, tp + fn)
    spec = _div(tn, tn + fp)
    f1 = _div(2 * prec * rec, prec + rec)
    return dict(accuracy=acc, precision=prec, recall=rec, specificity=spec, f1=f1,
                fpr=_div(fp, fp + tn))


def fit_logistic_main(sp):
    """The Task 2.2 model (eta 0.1, 1000 updates) on the training rows."""
    return train_logistic(sp["Xtr"], sp["ytr"], 0.1, 1000, record=(0, 1, 10, 100, 1000))


def knn_predict(Xref, yref, Xq, K):
    """Returns (neighbour positive share, predicted-positive boolean). Ties by file order."""
    shares = np.empty(len(Xq))
    for i, q in enumerate(Xq):
        d = np.sum((Xref - q) ** 2, axis=1)
        order = np.argsort(d, kind="stable")[:K]
        shares[i] = yref[order].sum() / K
    return shares, shares * K >= (K + 1) / 2 - 1e-12


def choose_k(Xtr, ytr):
    """Task 2.7: 5-fold CV (fold = training position mod 5), odd K 1..21.
    Returns (list of (K, mean F1), best K); ties go to the smaller K."""
    fold = np.arange(len(ytr)) % 5
    out, best_k, best_f1 = [], None, -1.0
    for K in range(1, 22, 2):
        f1s = []
        for f in range(5):
            ref, q = fold != f, fold == f
            _, pred = knn_predict(Xtr[ref], ytr[ref], Xtr[q], K)
            f1s.append(metrics(*confusion(ytr[q], pred))["f1"])
        mf = float(np.mean(f1s))
        out.append((K, mf))
        if mf > best_f1 + 1e-12:
            best_k, best_f1 = K, mf
    return out, best_k
