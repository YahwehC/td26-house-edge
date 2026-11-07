"""Tick-level logistic regression on TRAIN (all non-short episodes, ticks before any possible flip):
y_t ~ Bernoulli(sigmoid(offset + X beta)), offset = logit(leave-one-out plug-in cell rate)."""
import sys, math, json
from collections import defaultdict
sys.path.insert(0, '.')
from he.model import load_split
PAIRS = [(1, 3), (2, 5), (4, 7)]
FEATS = ["oth_win_%d" % k for k in range(1, 9)] + ["oth_net_8", "oth_net_9_16", "oth_win_9_16"]
def pairfit(seq, tmax):
    best = None
    for A, B in PAIRS:
        n = [[0, 0], [0, 0]]
        for t in seq:
            if t < tmax and (t - A) in seq and (t - B) in seq:
                n[int((seq[t - A] > 0) == (seq[t - B] > 0))][int(seq[t] > 0)] += 1
        r = [(n[z][1] + .5) / (sum(n[z]) + 1) for z in (0, 1)]
        if best is None or abs(r[1] - r[0]) > best[0]:
            best = (abs(r[1] - r[0]), (A, B), n)
    return best
def rows_for(split, max_eps=None):
    qs, ex = load_split('kit/public', split)
    cond = {r["query_id"]: r["condition"] for r in ex["conditions"]}
    ep = defaultdict(list)
    for q in qs: ep[q["episode_id"]].append(q)
    X, Y, O = [], [], []
    for e, ms in sorted(ep.items()):
        cnd = cond[ms[0]["query_id"]]
        if cnd == "short_history": continue
        T = min(m["cutoff"] for m in ms)
        tmax = T if cnd == "stable" else T - 110
        if tmax < 60: continue
        W = [[int(m["seq"].get(t, 0) > 0) for t in range(T)] for m in ms]
        NET = [[m["seq"].get(t, 1) - 1 for t in range(T)] for m in ms]
        for i, m in enumerate(ms):
            sep, (A, B), n = pairfit(m["seq"], tmax); s = m["seq"]
            if sep < 0.3: continue
            for t in range(20, tmax):
                z = int((s[t - A] > 0) == (s[t - B] > 0)); y = int(s[t] > 0)
                w_, tot = n[z][1] - y, sum(n[z]) - 1
                r = (w_ + .5) / (tot + 1); off = math.log(r / (1 - r))
                oth = [j for j in range(len(ms)) if j != i]
                f = [sum(W[j][t - k] for j in oth) for k in range(1, 9)]
                f.append(sum(NET[j][t - k] for j in oth for k in range(1, 9)) / 8)
                f.append(sum(NET[j][t - k] for j in oth for k in range(9, 17)) / 8)
                f.append(sum(W[j][t - k] for j in oth for k in range(9, 17)) / 8)
                X.append(f); Y.append(y); O.append(off)
    return X, Y, O
def logreg(X, Y, O, names, iters=8, ridge=1e-6):
    d = len(names) + 1
    mu = [sum(x[i] for x in X) / len(X) for i in range(len(names))]
    Z = [[1.0] + [x[i] - mu[i] for i in range(len(names))] for x in X]
    b = [0.0] * d
    for _ in range(iters):
        H = [[0.0] * d for _ in range(d)]; g = [0.0] * d
        for z, y, o in zip(Z, Y, O):
            eta = o + sum(bi * zi for bi, zi in zip(b, z)); p = 1 / (1 + math.exp(-eta)); w = p * (1 - p)
            for i in range(d):
                g[i] += (y - p) * z[i]
                for j in range(i, d): H[i][j] += w * z[i] * z[j]
        for i in range(d):
            H[i][i] += ridge
            for j in range(i): H[i][j] = H[j][i]
        # solve H step = g
        M = [H[i][:] + [g[i]] for i in range(d)]
        for i in range(d):
            piv = M[i][i]
            for j in range(i, d + 1): M[i][j] /= piv
            for r2 in range(d):
                if r2 != i:
                    fct = M[r2][i]
                    for j in range(i, d + 1): M[r2][j] -= fct * M[i][j]
        b = [bi + M[i][d] for i, bi in enumerate(b)]
    # SEs from inverse Hessian diag
    inv = [[float(i == j) for j in range(d)] for i in range(d)]
    M = [H[i][:] + inv[i] for i in range(d)]
    for i in range(d):
        piv = M[i][i]
        for j in range(2 * d): M[i][j] /= piv
        for r2 in range(d):
            if r2 != i:
                fct = M[r2][i]
                for j in range(2 * d): M[r2][j] -= fct * M[i][j]
    se = [math.sqrt(M[i][d + i]) for i in range(d)]
    return b, se, mu
X, Y, O = rows_for("train")
print("train ticks", len(X))
for sel in (FEATS, ["oth_win_3", "oth_net_8", "oth_net_9_16"], ["oth_win_3", "oth_win_9_16"] + ["oth_win_%d" % k for k in (1, 2, 4, 5, 6, 7, 8)]):
    idx = [FEATS.index(n) for n in sel]
    b, se, mu = logreg([[x[i] for i in idx] for x in X], Y, O, sel)
    print("intercept %+.3f (%.3f)" % (b[0], se[0]))
    for n, bi, si, m in zip(sel, b[1:], se[1:], mu):
        print("  %-14s beta %+.4f  se %.4f  z %+5.1f  mean %.2f" % (n, bi, si, bi / si, m))
    print()
