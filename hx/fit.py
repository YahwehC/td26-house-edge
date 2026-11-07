"""Empirical-Bayes fitting of priors from labelled splits (train/validation histories)."""
import math
from collections import Counter, defaultdict
from .model import WinMachine, _pair_prior, DEFAULT_CFG


def fit_cell_prior(splits, cfg=None):
    """splits: list of (queries, conditions dict). Uses 'stable' machines' full histories."""
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    cfg["cell_prior"] = None
    plp = _pair_prior(cfg)
    rows = defaultdict(list)
    for queries, cond in splits:
        for q in queries:
            if cond.get(q["query_id"]) != "stable":
                continue
            m = WinMachine(q, cfg, plp)
            A, B = max(m.pairs, key=lambda p: m.lml_stable[p])
            s = q["seq"]
            n = Counter()
            for t in s:
                if (t - A) in s and (t - B) in s:
                    n[((s[t - A] > 0) == (s[t - B] > 0), s[t] > 0)] += 1
            cells = []
            for z in (True, False):
                w, l = n[(z, True)], n[(z, False)]
                if w + l >= 10:
                    cells.append(((w + 0.5) / (w + l + 1), w + l))
            if len(cells) == 2:
                hi, lo = sorted(cells, reverse=True)
                rows[q["machine_type"]].append((q["signal"], hi, lo))
    out = {}
    lg = lambda p: math.log(p / (1 - p))
    for ty, R in rows.items():
        spec = {}
        for nm, ix in (("hi", 1), ("lo", 2)):
            xs = [r[0] for r in R]
            ys = [lg(r[ix][0]) for r in R]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            sxx = sum((x - mx) ** 2 for x in xs)
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else 0.0
            mu = my - slope * mx
            # kappa by moment matching in prob space, removing binomial noise
            ps = [r[ix][0] for r in R]
            ns = [r[ix][1] for r in R]
            # residual around signal-adjusted mean
            preds = [1 / (1 + math.exp(-(mu + slope * x))) for x in xs]
            var = sum((p - pr) ** 2 for p, pr in zip(ps, preds)) / len(ps)
            noise = sum(p * (1 - p) / n for p, n in zip(ps, ns)) / len(ps)
            m = sum(preds) / len(preds)
            v = max(var - noise, 1e-5)
            kappa = max(2.0, min(500.0, m * (1 - m) / v - 1))
            spec[nm] = [mu, slope, kappa]
        out[ty] = spec
    return out


def fit_structure_priors(splits, vecs, cfg=None, smooth=0.5, min_frac=0.03, lag_min_frac=0.1, alpha_joint=1.0):
    """Learn pair prior (all 28 lag pairs) and size-lag prior (lags 1..8) from 'stable' machines.

    Each stable machine votes for its best pair (by stable marginal likelihood under a flat pair prior)
    and its best size lag (by size likelihood). Returns cfg overrides."""
    from .model import MAXLAG, SIZE_IDX
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    allpairs = [(a, b) for a in range(1, MAXLAG + 1) for b in range(a + 1, MAXLAG + 1)]
    flat = dict(cfg, pairs={f"{a},{b}": 1.0 for a, b in allpairs}, pair_other=0.0)
    plp = _pair_prior(flat)
    pv, lv, jv = Counter(), Counter(), Counter()
    for queries, cond in splits:
        for q in queries:
            if cond.get(q["query_id"]) != "stable":
                continue
            m = WinMachine(q, flat, plp)
            ranked = sorted(m.pairs, key=lambda p: -m.lml_stable[p])
            pair_ok = m.lml_stable[ranked[0]] - m.lml_stable[ranked[1]] > 2.0   # decisive only
            if pair_ok:
                pv[ranked[0]] += 1
            # size lag vote
            V = vecs[q["machine_type"]]
            LV = [[math.log(x) for x in v] for v in V]
            s = q["seq"]
            wins = [(t, SIZE_IDX[v]) for t, v in s.items() if v > 0]
            best = []
            for L in range(1, MAXLAG + 1):
                xs = [(s[t - L] >= 2, k) for t, k in wins if (t - L) in s]
                ll = max(math.fsum(LV[int(b) ^ sg][k] for b, k in xs) for sg in (0, 1))
                best.append((ll, L))
            best.sort(reverse=True)
            if best[0][0] - best[1][0] > 3.0:
                lv[best[0][1]] += 1
                if pair_ok:
                    jv[(ranked[0], best[0][1])] += 1
    npair = sum(pv.values())
    pairs = {f"{a},{b}": c for (a, b), c in pv.items() if c >= min_frac * npair}
    nl = sum(lv.values())
    lags = {str(L): c + smooth for L, c in lv.items() if c >= lag_min_frac * nl}
    # P(L | pair): joint votes smoothed toward the marginal lag prior
    zl = sum(lags.values())
    table = {}
    for a, b in [tuple(int(x) for x in k.split(",")) for k in pairs]:
        row = {L: c for (p, L), c in jv.items() if p == (a, b)}
        n = sum(row.values())
        table[f"{a},{b}"] = {str(L): (row.get(L, 0) + alpha_joint * (lags.get(str(L), 0) / zl)) / (n + alpha_joint)
                             for L in range(1, MAXLAG + 1) if row.get(L, 0) > 0 or str(L) in lags}
    return {"pairs": pairs, "pair_other": 0.0, "size_lag_given_pair": table, "size_lags": lags, "size_lag_other": 0.05,
            "_votes": {"pairs": {f"{a},{b}": c for (a, b), c in pv.most_common()}, "lags": dict(lv.most_common()),
                       "n_pair": npair, "n_lag": nl,
                       "joint": {f"{p[0]},{p[1]}|{L}": c for (p, L), c in jv.most_common()}}}


def fit_tau_priors(splits, cfg, smooth=0.02, bw=8):
    """Histogram of flip ticks relative to cutoff (bins of bw), from posterior over taus in labelled
    drift / shared_change training episodes (under the current cfg, flat tau prior)."""
    from collections import defaultdict as dd
    from .model import episode_posterior, is_short_episode
    base = dict(cfg, tau_hist_drift=None, tau_hist_shared=None, tau_bin=bw)
    plp = _pair_prior(base)
    H = {"drift": Counter(), "shared_change": Counter()}
    short_counts = Counter()
    for queries, cond in splits:
        eps = dd(list)
        for q in queries:
            eps[q["episode_id"]].append(q)
        for e, qs in sorted(eps.items()):
            cd = cond.get(qs[0]["query_id"])
            ms = [WinMachine(q, base, plp) for q in sorted(qs, key=lambda q: q["machine_id"])]
            if is_short_episode(ms):
                short_counts[cd] += 1
                continue
            if cd not in H:
                continue
            fixed = dict(base, model_prior={"drift": 1.0} if cd == "drift" else {"shared": 1.0})
            _, d = episode_posterior(ms, fixed)
            c = ms[0].q["cutoff"]
            if cd == "drift":
                for tp in d["tpost_drift"]:
                    s = sum(tp)
                    for tau, w in zip(d["taus"], tp):
                        H[cd][(c - tau) // bw] += w / s if s > 0 else 0
            elif d["tpost_shared"]:
                for tau, w in zip(d["taus"], d["tpost_shared"]):
                    H[cd][(c - tau) // bw] += w
    out = {}
    nb = 260 // bw + 1
    for cd, key in (("drift", "tau_hist_drift"), ("shared_change", "tau_hist_shared")):
        h = H[cd]
        tot = sum(h.values())
        out[key] = {str(b): (h[b] / tot if tot else 0) + smooth / nb for b in range(nb)}
        out[key]["rest"] = smooth / nb
    return out, dict(short_counts)


def fit_lag_given_pair_em(splits, vecs, cfg, iters=50, floor=1e-3):
    """EM estimate of P(L | pair) using stable machines whose win pair is decisive.

    For each machine: size log-likelihood for each L (marginalised over sign s with 1/2 each, plus no
    constant-regime term); E-step posterior over L given P(L|pair); M-step average."""
    from .model import MAXLAG, SIZE_IDX, logsumexp
    plp = _pair_prior(cfg)
    data = defaultdict(list)
    eps = cfg["size_eps"]
    for queries, cond in splits:
        for q in queries:
            if cond.get(q["query_id"]) != "stable":
                continue
            m = WinMachine(q, cfg, plp)
            ranked = sorted(m.pairs, key=lambda p: -m.lml_stable[p])
            if len(ranked) > 1 and m.lml_stable[ranked[0]] - m.lml_stable[ranked[1]] <= 2.0:
                continue
            V = vecs[q["machine_type"]]
            LV = [[math.log((1 - eps) * V[r][k] + eps * V[1 - r][k]) for k in range(4)] for r in (0, 1)]
            s = q["seq"]
            wins = [(t, SIZE_IDX[v]) for t, v in s.items() if v > 0]
            lls = []
            for L in range(1, MAXLAG + 1):
                xs = [(s[t - L] >= 2, k) for t, k in wins if (t - L) in s]
                lls.append(logsumexp([math.log(0.5) + math.fsum(LV[int(b) ^ sg][k] for b, k in xs) for sg in (0, 1)]))
            data[ranked[0]].append(lls)
    table = {}
    for p, rows in data.items():
        th = [1.0 / MAXLAG] * MAXLAG
        for _ in range(iters):
            acc = [0.0] * MAXLAG
            for lls in rows:
                lw = [math.log(t) + l for t, l in zip(th, lls)]
                z = logsumexp(lw)
                for i, x in enumerate(lw):
                    acc[i] += math.exp(x - z)
            th = [max(a / len(rows), floor) for a in acc]
            zt = math.fsum(th)
            th = [t / zt for t in th]
        table[f"{p[0]},{p[1]}"] = {str(L + 1): th[L] for L in range(MAXLAG)}
    return table


def fit_lag_rule(splits, vecs, cfg, delta=0.01):
    """Choose how the size lag relates to the win pair: 'first' (L=A), 'second' (L=B) or 'none'
    (marginal lag prior), by summed per-machine size marginal likelihood on stable machines with a
    decisive pair. Returns (table or None, diagnostics)."""
    from .model import MAXLAG, SIZE_IDX, logsumexp
    from .joint import lag_prior_marginal
    plp = _pair_prior(cfg)
    marg = lag_prior_marginal(cfg)
    eps = cfg["size_eps"]
    pairs = list(plp)

    def table_for(rule):
        if rule == "none":
            return None
        t = {}
        for a, b in pairs:
            L0 = a if rule == "first" else b
            row = {str(L): delta * marg.get(L, 0.0) for L in range(1, MAXLAG + 1)}
            row[str(L0)] = row.get(str(L0), 0.0) + (1 - delta)
            t[f"{a},{b}"] = row
        return t

    totals = {"first": 0.0, "second": 0.0, "none": 0.0}
    for queries, cond in splits:
        for q in queries:
            if cond.get(q["query_id"]) != "stable":
                continue
            m = WinMachine(q, cfg, plp)
            ranked = sorted(m.pairs, key=lambda p: -m.lml_stable[p])
            if len(ranked) > 1 and m.lml_stable[ranked[0]] - m.lml_stable[ranked[1]] <= 2.0:
                continue
            a, b = ranked[0]
            V = vecs[q["machine_type"]]
            LV = [[math.log((1 - eps) * V[r][k] + eps * V[1 - r][k]) for k in range(4)] for r in (0, 1)]
            s = q["seq"]
            wins = [(t, SIZE_IDX[v]) for t, v in s.items() if v > 0]
            ll = {}
            for L in range(1, MAXLAG + 1):
                xs = [(s[t - L] >= 2, k) for t, k in wins if (t - L) in s]
                ll[L] = logsumexp([math.log(0.5) + math.fsum(LV[int(x) ^ sg][k] for x, k in xs) for sg in (0, 1)])
            for rule in totals:
                tab = table_for(rule)
                pri = marg if tab is None else {int(k): v for k, v in tab[f"{a},{b}"].items() if v > 0}
                totals[rule] += logsumexp([math.log(w) + ll[L] for L, w in pri.items()])
    best = max(totals, key=totals.get)
    return table_for(best), {"rule": best, "loglik": {k: round(v, 1) for k, v in totals.items()}}
