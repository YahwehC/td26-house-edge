"""Episode-level logit corrections of the win probability (fitted on validation targets, CV-gated in r2.py).

logit(w') = logit(w) + sum_i beta_i * (x_i - mu_i), size distribution given a win unchanged. Features:
- "ep_net_K":  mean net payout (payout - 1) of the episode's machines over the K ticks before cutoff ("heat").
- "oth_win_k": number of OTHER machines of the episode that won at tick cutoff - k (cross-machine coupling;
               k = 3 is a strong spike in the main kit: +0.09 logit per winner, z=9 on train ticks).
- "oth_wdev_k": oth_win_k minus this machine's history mean of the same count (mu should then be 0).
Old params format {"K", "beta", "mu"} (single ep_net_K feature) is still accepted. Pure stdlib, deterministic.
"""
import math
from collections import defaultdict


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _by_episode(queries):
    by_ep = defaultdict(list)
    for q in queries:
        by_ep[q["episode_id"]].append(q)
    return by_ep


def feature(queries, name):
    """{query_id: value} for one feature name."""
    by_ep = _by_episode(queries)
    kind, _, num = name.rpartition("_")
    k = int(num)
    out = {}
    for q in queries:
        c = q["cutoff"]
        mates = by_ep[q["episode_id"]]
        if kind == "ep_net":
            tot = math.fsum(m["seq"].get(c - j, 1) - 1 for m in mates for j in range(1, k + 1))
            out[q["query_id"]] = tot / (5.0 * k)
        elif kind == "oth_win":
            out[q["query_id"]] = float(sum(1 for m in mates if m is not q and m["seq"].get(c - k, 0) > 0))
        elif kind == "oth_net":
            # sum over OTHER machines of net payout over cutoff-1..cutoff-k, divided by k (as in he/coupling.py)
            tot = math.fsum(m["seq"].get(c - j, 1) - 1 for m in mates if m is not q for j in range(1, k + 1))
            out[q["query_id"]] = tot / float(k)
        elif kind == "oth_wdev":
            # others' winners at cutoff-k minus this machine's own history mean of that count (cells absorb the mean)
            now = sum(1 for m in mates if m is not q and m["seq"].get(c - k, 0) > 0)
            ticks = [t for t in q["seq"] if t - k >= 0 and all((t - k) in m["seq"] for m in mates if m is not q)]
            if ticks:
                mean = math.fsum(sum(1 for m in mates if m is not q and m["seq"][t - k] > 0) for t in ticks) / len(ticks)
            else:
                mean = now
            out[q["query_id"]] = now - mean
        else:
            raise ValueError("unknown feature " + name)
    return out


def episode_feature(queries, K=8):
    """Backward-compatible single heat feature."""
    return feature(queries, "ep_net_%d" % K)


def _normalize(corr):
    if "names" in corr:
        return corr["names"], corr["beta"], corr["mu"]
    return ["ep_net_%d" % corr.get("K", 8)], [corr["beta"]], [corr["mu"]]


def apply_corr(queries, pred, corr):
    """Return a new prediction dict with the win logit shifted; size distribution given a win is unchanged."""
    if not corr:
        return pred
    names, beta, mu = _normalize(corr)
    F = [feature(queries, n) for n in names]
    out = {}
    for q in queries:
        k = q["query_id"]
        p = pred[k]
        w = 1 - p[0]
        if w <= 1e-12:
            out[k] = p
            continue
        z = _logit(w) + math.fsum(b * (f[k] - m) for b, f, m in zip(beta, F, mu))
        w2 = 1 / (1 + math.exp(-z))
        out[k] = [1 - w2] + [w2 * x / w for x in p[1:]]
    return out


def _solve(A, b):
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
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


def fit_corr(queries, pred, targets, K=8, names=None, ridge=1e-3, iters=12):
    """Least-squares fit (win-prob error vs targets) by Gauss-Newton; mu = mean of each feature."""
    names = list(names) if names else ["ep_net_%d" % K]
    F = [feature(queries, n) for n in names]
    mu = [math.fsum(f.values()) / len(f) for f in F]
    tw = {t["query_id"]: 1 - float(t["p_0"]) for t in targets}
    keys = [q["query_id"] for q in queries if q["query_id"] in tw]
    d = len(names)
    beta = [0.0] * d
    for _ in range(iters):
        JJ = [[0.0] * d for _ in range(d)]
        Jr = [0.0] * d
        for k in keys:
            w = 1 - pred[k][0]
            x = [f[k] - m for f, m in zip(F, mu)]
            w2 = 1 / (1 + math.exp(-(_logit(w) + math.fsum(b * xi for b, xi in zip(beta, x)))))
            g = [w2 * (1 - w2) * xi for xi in x]
            r = w2 - tw[k]
            for i in range(d):
                Jr[i] += g[i] * r
                for j in range(d):
                    JJ[i][j] += g[i] * g[j]
        for i in range(d):
            JJ[i][i] += ridge
        step = _solve(JJ, Jr)
        beta = [b - s for b, s in zip(beta, step)]
    return {"names": names, "beta": beta, "mu": mu}
