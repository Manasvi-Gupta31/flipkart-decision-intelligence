"""Part 3: K-means, hierarchical clustering on a supplied table, PCA by the seven steps."""
import numpy as np

from common import Results, load_data, sd_cols, write_csv

X, y, value = load_data()
X5 = X[:, :5]
n = len(X5)
Z = (X5 - X5.mean(axis=0)) / sd_cols(X5)
R = Results()


def kmeans(Z, K):
    step = len(Z) // K
    cent = Z[[i * step for i in range(K)]].copy()
    labels = None
    passes = 0
    while True:
        d = np.sum((Z[:, None, :] - cent[None, :, :]) ** 2, axis=2)
        new = np.argmin(d, axis=1)  # first minimum -> lower index wins ties
        passes += 1
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        for k in range(K):
            mem = Z[labels == k]
            if len(mem):
                cent[k] = mem.mean(axis=0)
    wcss = float(np.sum((Z - cent[labels]) ** 2))
    return labels, cent, wcss, passes


# 3.1
res = {}
for K in range(2, 7):
    res[K] = kmeans(Z, K)
    R.add("wcss_k%d" % K, res[K][2])

# 3.2
W = {K: res[K][2] for K in res}
elbow = None
for K in (3, 4, 5):
    if W[K] - W[K + 1] < 0.5 * (W[K - 1] - W[K]):
        elbow = K
        break
R.add("elbow_k", elbow if elbow is not None else "none")

# 3.3
labels = res[3][0]
mx3 = [X[labels == k, 2].mean() for k in range(3)]
order = np.argsort(mx3)
seg_names = ["A", "B", "C"]
seg_of = np.empty(n, int)
rows = []
for si, k in enumerate(order):
    m = labels == k
    seg_of[m] = si
    R.add("size_segment_%s" % seg_names[si], int(m.sum()))
    rows.append([seg_names[si], int(m.sum())] + [float(X[m, j].mean()) for j in range(5)]
                + [float(y[m].mean()), float(value[m].mean())])
write_csv("part3_segments.csv",
          ["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5",
           "outcome_rate", "mean_value"], rows)

# 3.4
names = ["E1", "E2", "E3", "E4", "E5", "E6"]
D = np.array([[0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
              [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
              [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
              [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
              [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
              [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]])


def agglomerate(link):
    clusters = [[i] for i in range(6)]
    merges = []
    while len(clusters) > 1:
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                ds = [D[i, j] for i in clusters[a] for j in clusters[b]]
                h = min(ds) if link == "single" else (max(ds) if link == "complete"
                                                       else sum(ds) / len(ds))
                if best is None or h < best[0]:
                    best = (h, a, b)
        h, a, b = best
        ca, cb = clusters[a], clusters[b]
        if min(cb) < min(ca):
            ca, cb = cb, ca
        merges.append(("+".join(names[i] for i in sorted(ca)),
                       "+".join(names[i] for i in sorted(cb)), h))
        merged = sorted(clusters[a] + clusters[b])
        clusters = [c for i, c in enumerate(clusters) if i not in (a, b)] + [merged]
    return merges


hrows = []
for link in ("single", "complete", "average"):
    ms = agglomerate(link)
    for i, (ca, cb, h) in enumerate(ms):
        hrows.append([link, i + 1, ca, cb, h])
    hs = [m[2] for m in ms]
    gaps = [hs[i + 1] - hs[i] for i in range(4)]
    g = int(np.argmax(gaps))  # gap between merge g+1 and g+2 -> g+1 merges below the cut
    R.add("%s_clusters_at_cut" % link, 6 - (g + 1))
write_csv("part3_hierarchical.csv", ["linkage", "merge_no", "cluster_a", "cluster_b", "height"], hrows)

# 3.5 PCA
mean_vec = Z.mean(axis=0)            # step 1 (zeros)
Xc = Z - mean_vec                    # step 2
S = Xc.T @ Xc / (n - 1)              # step 3
evals, evecs = np.linalg.eigh(S)     # step 4
idx = np.argsort(-evals)             # step 5
evals, evecs = evals[idx], evecs[:, idx]
for j in range(5):                   # sign convention
    if evecs[np.argmax(np.abs(evecs[:, j])), j] < 0:
        evecs[:, j] = -evecs[:, j]
Wm = evecs[:, :2]                    # step 6
P = Xc @ Wm                          # step 7
tot = float(evals.sum())
R.add("eigenvalue_sum", tot)
for j in range(5):
    R.add("eigenvalue_%d" % (j + 1), float(evals[j]))
for j in range(5):
    R.add("explained_ratio_%d" % (j + 1), float(evals[j] / tot))
cum = np.cumsum(evals / tot)
for j in range(5):
    R.add("cumulative_ratio_%d" % (j + 1), float(cum[j]))
R.add("components_for_80pct", int(np.argmax(cum >= 0.8 - 1e-12)) + 1)
for j in range(5):
    R.add("pc1_loading_X%d" % (j + 1), float(evecs[j, 0]))
for j in range(5):
    R.add("pc2_loading_X%d" % (j + 1), float(evecs[j, 1]))
for r in range(3):
    R.add("row%d_pc1" % (r + 1), float(P[r, 0]))
    R.add("row%d_pc2" % (r + 1), float(P[r, 1]))
for si in range(3):
    R.add("mean_pc1_segment_%s" % seg_names[si], float(P[seg_of == si, 0].mean()))

R.write("part3_results.txt")
print({K: res[K][3] for K in res})
