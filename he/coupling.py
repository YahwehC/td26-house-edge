"""Cross-machine coupling fitted on per-tick outcomes (pure stdlib, deterministic).

Model per tick t of machine i (non-short episodes; ticks before any possible flip):
    y_it ~ Bernoulli(sigmoid(offset_it + sum_f beta_f * (x_f,it - mu_f)))
offset = logit of the leave-one-out plug-in cell rate of the machine's best XNOR pair. Candidate features:
    oth_win_k  (k = 1..8): number of OTHER machines in the episode that won at t-k
    oth_net_8            : sum over the other machines of net payout over t-1..t-8, divided by 8 ("heat")
Main kit (train ticks): oth_win_3 beta +0.087 (z=9), oth_net_8 +0.010 (z=2); all other lags |z| < 2.5.
Output is a he/corr.py-compatible dict {"names", "beta", "mu", "z"} applied at the query cutoff.
"""
import math
from collections import defaultdict

PAIRS = ((1, 3), (2, 5), (4, 7))
CANDIDATES = ["oth_win_%d" % k for k in range(1, 9)] + ["oth_net_8"]


def _best_pair(seq, tmax, pairs):
    best = None
    for A, B in pairs:
        n = [[0, 0], [0, 0]]
        for t in seq:
            if t < tmax and (t - A) in seq and (t - B) in seq:
                n[int((seq[t - A] > 0) == (seq[t - B] > 0))][int(seq[t] > 0)] += 1
        r = [(n[z][1] + .5) / (sum(n[z]) + 1) for z in (0, 1)]
        sep = abs(r[1] - r[0])
        if best is None or sep > best[0]:
            best = (sep, (A, B), n)
    return best


def tick_rows(queries, conditions=None, pairs=PAIRS, flip_margin=110, min_sep=0.3):
    """Feature rows (X over CANDIDATES), outcomes Y and offsets O from per-tick histories."""
    cond = conditions or {}
    by_ep = defaultdict(list)
    for q in queries:
        by_ep[q["episode_id"]].append(q)
    X, Y, O = [], [], []
    for e in sorted(by_ep):
        ms = sorted(by_ep[e], key=lambda q: q["machine_id"])
        c0 = cond.get(ms[0]["query_id"])
        if c0 == "short_history" or any(len(m["seq"]) < 0.6 * m["cutoff"] for m in ms):
            continue
        T = min(m["cutoff"] for m in ms)
        tmax = T if c0 == "stable" else T - flip_margin
        if tmax < 60:
            continue
        W = [[int(m["seq"].get(t, 0) > 0) for t in range(T)] for m in ms]
        NET = [[m["seq"].get(t, 1) - 1 for t in range(T)] for m in ms]
        sw = [sum(W[j][t] for j in range(len(ms))) for t in range(T)]
        sn = [sum(NET[j][t] for j in range(len(ms))) for t in range(T)]
        for i, m in enumerate(ms):
            sep, (A, B), n = _best_pair(m["seq"], tmax, pairs)
            if sep < min_sep:
                continue
            s = m["seq"]
            for t in range(20, tmax):
                if t not in s or (t - A) not in s or (t - B) not in s:
                    continue
                z = int((s[t - A] > 0) == (s[t - B] > 0))
                y = int(s[t] > 0)
                r = (n[z][1] - y + .5) / (sum(n[z]) - 1 + 1)
                O.append(math.log(r / (1 - r)))
                Y.append(y)
                f = [float(sw[t - k] - W[i][t - k]) for k in range(1, 9)]
                f.append(math.fsum(sn[t - k] - NET[i][t - k] for k in range(1, 9)) / 8.0)
                X.append(f)
    return X, Y, O


def _logreg(X, Y, O, idx, iters=8, ridge=1e-6):
    d = len(idx) + 1
    mu = [math.fsum(x[i] for x in X) / len(X) for i in idx]
    b = [0.0] * d
    H = None
    for _ in range(iters):
        H = [[0.0] * d for _ in range(d)]
        g = [0.0] * d
        for x, y, o in zip(X, Y, O):
            zr = [1.0] + [x[i] - m for i, m in zip(idx, mu)]
            eta = o + math.fsum(bi * zi for bi, zi in zip(b, zr))
            p = 1 / (1 + math.exp(-eta))
            w = p * (1 - p)
            for a in range(d):
                g[a] += (y - p) * zr[a]
                wa = w * zr[a]
                for c in range(a, d):
                    H[a][c] += wa * zr[c]
        for a in range(d):
            H[a][a] += ridge
            for c in range(a):
                H[a][c] = H[c][a]
        b = [bi + si for bi, si in zip(b, _solve(H, g))]
    se = [math.sqrt(max(v, 1e-18)) for v in _inv_diag(H)]
    return b, se, mu


def _solve(A, rhs):
    n = len(rhs)
    M = [A[i][:] + [rhs[i]] for i in range(n)]
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[piv] = M[piv], M[i]
        d = M[i][i]
        for j in range(i, n + 1):
            M[i][j] /= d
        for r in range(n):
            if r != i:
                f = M[r][i]
                for j in range(i, n + 1):
                    M[r][j] -= f * M[i][j]
    return [M[i][n] for i in range(n)]


def _inv_diag(A):
    n = len(A)
    return [_solve(A, [float(i == j) for j in range(n)])[i] for i in range(n)]


def fit_coupling(query_lists, condition_maps=None, z_keep=3.0, log=print):
    """Fit all candidates, keep |z| >= z_keep, refit the kept set. Returns corr dict or None if nothing kept."""
    X, Y, O = [], [], []
    for k, qs in enumerate(query_lists):
        cm = (condition_maps or [None] * len(query_lists))[k]
        x, y, o = tick_rows(qs, cm)
        X += x
        Y += y
        O += o
    if len(X) < 2000:
        log(f"[coupling] only {len(X)} ticks - skipped")
        return None
    idx = list(range(len(CANDIDATES)))
    b, se, mu = _logreg(X, Y, O, idx)
    log(f"[coupling] {len(X)} ticks; all candidates: " + ", ".join(
        f"{CANDIDATES[i]} {b[j + 1]:+.3f}(z{b[j + 1] / se[j + 1]:+.1f})" for j, i in enumerate(idx)))
    keep = [i for j, i in enumerate(idx) if abs(b[j + 1] / se[j + 1]) >= z_keep]
    if not keep:
        log("[coupling] no feature with |z| >= %.1f" % z_keep)
        return None
    b, se, mu = _logreg(X, Y, O, keep)
    names = [CANDIDATES[i] for i in keep]
    log("[coupling] kept: " + ", ".join(f"{n} {bi:+.4f}(z{bi / si:+.1f})" for n, bi, si in zip(names, b[1:], se[1:])))
    return {"names": names, "beta": b[1:], "mu": mu, "z": [bi / si for bi, si in zip(b[1:], se[1:])],
            "n_ticks": len(X)}
