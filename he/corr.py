"""Episode "heat" correction: shift the win-probability logit by beta * (recent episode net payout - mu).

Found on validation: the true win probability exceeds our model's when the episode's 8 machines paid out more
than usual over the last K ticks (2-fold CV by episode: +0.10 points). Pure standard library, deterministic.
"""
import math
from collections import defaultdict


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def episode_feature(queries, K=8):
    """{query_id: mean net payout (payout - 1) of the episode's machines over the K ticks before cutoff}."""
    by_ep = defaultdict(list)
    for q in queries:
        by_ep[q["episode_id"]].append(q)
    out = {}
    for q in queries:
        c = q["cutoff"]
        mates = by_ep[q["episode_id"]]
        tot = math.fsum(m["seq"].get(c - k, 1) - 1 for m in mates for k in range(1, K + 1))
        out[q["query_id"]] = tot / (5.0 * K)  # same scale as eda/ev29.py
    return out


def apply_corr(queries, pred, corr):
    """Return a new prediction dict with the win logit shifted; size distribution given a win is unchanged."""
    if not corr or not corr.get("beta"):
        return pred
    f = episode_feature(queries, corr.get("K", 8))
    out = {}
    for q in queries:
        k = q["query_id"]
        p = pred[k]
        w = 1 - p[0]
        if w <= 1e-12:
            out[k] = p
            continue
        z = _logit(w) + corr["beta"] * (f[k] - corr["mu"])
        w2 = 1 / (1 + math.exp(-z))
        out[k] = [1 - w2] + [w2 * x / w for x in p[1:]]
    return out


def fit_corr(queries, pred, targets, K=8, ridge=1e-3, iters=10):
    """Least-squares fit of beta (win-prob error vs targets) by Gauss-Newton; mu = mean feature."""
    f = episode_feature(queries, K)
    mu = math.fsum(f.values()) / len(f)
    tw = {t["query_id"]: 1 - float(t["p_0"]) for t in targets}
    keys = [q["query_id"] for q in queries if q["query_id"] in tw]
    beta = 0.0
    for _ in range(iters):
        jj = jr = 0.0
        for k in keys:
            w = 1 - pred[k][0]
            x = f[k] - mu
            w2 = 1 / (1 + math.exp(-(_logit(w) + beta * x)))
            g = w2 * (1 - w2) * x
            jj += g * g
            jr += g * (w2 - tw[k])
        beta -= jr / (jj + ridge)
    return {"K": K, "beta": beta, "mu": mu}
