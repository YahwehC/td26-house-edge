"""Episode-level pooling of cell levels: machines in an episode share a logit offset (hi and lo)."""
import math
from collections import defaultdict

from .model import WinMachine, _pair_prior, DEFAULT_CFG


def machine_residuals(q, cfg, plp):
    """Best (pair, flip) hypothesis -> smoothed hi/lo cell logits minus the type/signal prior mean.
    Returns ({'hi': r, 'lo': r}, {'hi': n, 'lo': n}) or None."""
    cp = (cfg.get("cell_prior") or {}).get(q["machine_type"])
    if not cp:
        return None
    m = WinMachine(dict(q, _cell_offset=None), dict(DEFAULT_CFG, **dict(cfg, cell_prior=None)), plp)
    best = None
    for p in m.pairs:
        cand = [(m.lml_stable[p], p, None)] + [(l, p, i) for i, l in enumerate(m.lml_tau[p])]
        for c in cand:
            if best is None or c[0] > best[0]:
                best = c
    _, (A, B), ti = best
    s, tau = q["seq"], (m.taus[ti] if ti is not None else None)
    n = {0: [0, 0], 1: [0, 0]}
    for t in s:
        if (t - A) in s and (t - B) in s:
            z = int((s[t - A] > 0) == (s[t - B] > 0))
            e = z if (tau is None or t < tau) else 1 - z
            n[e][int(s[t] > 0)] += 1
    rates = []
    for e in (0, 1):
        l, w = n[e]
        rates.append(((w + 0.5) / (w + l + 1), w + l))
    hi, lo = sorted(rates, reverse=True)
    lg = lambda x: math.log(x / (1 - x))
    out, cnt = {}, {}
    for nm, (r, k) in (("hi", hi), ("lo", lo)):
        mu, slope, _ = cp[nm]
        out[nm] = lg(r) - (mu + slope * q["signal"])
        cnt[nm] = k
    return out, cnt


def episode_offsets(queries, cfg, slopes, min_n=10):
    """Per query: offset[nm] = slopes[nm] * mean of episode-mates' residuals (mates with >= min_n obs)."""
    plp = _pair_prior(dict(DEFAULT_CFG, **cfg))
    res = {q["query_id"]: machine_residuals(q, cfg, plp) for q in queries}
    by_ep = defaultdict(list)
    for q in queries:
        by_ep[q["episode_id"]].append(q["query_id"])
    offs = {}
    for ep, ids in by_ep.items():
        for qid in ids:
            o = {}
            for nm in ("hi", "lo"):
                vals = [res[j][0][nm] for j in ids if j != qid and res[j] and res[j][1][nm] >= min_n]
                o[nm] = slopes[nm] * (math.fsum(vals) / len(vals)) if vals else 0.0
            offs[qid] = o
    return offs, res


def fit_slopes(splits, cfg, min_n=10):
    """Regression slope (through origin) of own residual on mates' mean residual, stable+drift+shared machines."""
    num = {"hi": 0.0, "lo": 0.0}
    den = {"hi": 0.0, "lo": 0.0}
    for queries, cond in splits:
        offs, res = episode_offsets(queries, cfg, {"hi": 1.0, "lo": 1.0}, min_n)
        for q in queries:
            r = res[q["query_id"]]
            if not r or cond.get(q["query_id"]) == "short_history":
                continue
            for nm in ("hi", "lo"):
                if r[1][nm] >= min_n:
                    x = offs[q["query_id"]][nm]
                    num[nm] += x * r[0][nm]
                    den[nm] += x * x
    return {nm: (num[nm] / den[nm] if den[nm] > 0 else 0.0) for nm in num}
