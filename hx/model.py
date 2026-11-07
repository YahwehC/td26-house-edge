"""House Edge structured Bayesian forecaster (pure standard library).

Generative picture recovered from the data (see CLAUDE.md):
  * win_t ~ Bernoulli(cell probability), cell = XNOR(win_{t-A}, win_{t-B}) for a machine-specific
    lag pair (A, B); the mapping cell -> probability can flip sign at a change tick tau
    (per machine in 'drift', common to the episode in 'shared_change').
  * given a win, the size mix is one of two fixed vectors per machine type; which one is chosen by
    (payout_{t-L} >= 2) XOR s for a machine-specific lag L and sign s.
Everything is a closed-form Bayesian mixture: Beta-Binomial cells, enumerated (pair, tau),
enumerated (L, s) for sizes.
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

PAYOUTS = (0, 1, 2, 5, 20)
SIZE_IDX = {1: 0, 2: 1, 5: 2, 20: 3}
MAXLAG = 8


# ----------------------------------------------------------------------------- io
def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_split(public, split):
    d = Path(public) / split
    queries = read_csv(d / "queries.csv")
    hist = defaultdict(dict)
    for r in read_csv(d / "history.csv"):
        hist[(r["episode_id"], r["machine_id"])][int(r["tick"])] = int(r["payout"])
    for q in queries:
        q["cutoff"] = int(q["cutoff"])
        q["signal"] = int(q["signal"])
        seq = hist.get((q["episode_id"], q["machine_id"]), {})
        q["seq"] = {t: v for t, v in seq.items() if t < q["cutoff"]}  # strictly before cutoff
    extra = {}
    for name in ("labels", "conditions", "targets", "reference"):
        f = d / f"{name}.csv"
        if f.exists():
            extra[name] = read_csv(f)
    return queries, extra


# ----------------------------------------------------------------------------- math helpers
def logsumexp(xs):
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    return m + math.log(math.fsum(math.exp(x - m) for x in xs))


def betaln(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


# ----------------------------------------------------------------------------- size vectors
def fit_size_vectors(queries, targets):
    """Two size-given-win vectors per machine type, learned from validation targets (2-means)."""
    qmap = {q["query_id"]: q for q in queries}
    by_type = defaultdict(list)
    for t in targets:
        p = [float(t[k]) for k in ("p_0", "p_1", "p_2", "p_5", "p_20")]
        w = 1 - p[0]
        if w < 1e-6:
            continue
        by_type[qmap[t["query_id"]]["machine_type"]].append(([x / w for x in p[1:]], w))
    vecs = {}
    for ty, rows in by_type.items():
        pts = [r[0] for r in rows]
        # init: min / max of p_1
        c = [min(pts, key=lambda v: v[0]), max(pts, key=lambda v: v[0])]
        for _ in range(50):
            groups = ([], [])
            for v in pts:
                d0 = sum((a - b) ** 2 for a, b in zip(v, c[0]))
                d1 = sum((a - b) ** 2 for a, b in zip(v, c[1]))
                groups[0 if d0 <= d1 else 1].append(v)
            c = [[math.fsum(v[i] for v in g) / len(g) for i in range(4)] if g else c[j]
                 for j, g in enumerate(groups)]
        c = [[x / math.fsum(v) for x in v] for v in c]
        vecs[ty] = c
    return vecs


# ----------------------------------------------------------------------------- default config
DEFAULT_CFG = {
    "pairs": {"1,3": 1.0, "2,5": 1.0, "4,7": 1.0},  # prior weights over lag pairs
    "pair_other": 0.0,          # total prior mass spread over all other pairs
    "beta_a": 0.5, "beta_b": 0.5,  # Beta prior for each cell
    "model_prior": {"stable": 0.4, "drift": 0.3, "shared": 0.3},
    "drift_rho": 0.9,           # prob. each machine flips in a drift episode
    "shared_rho": 0.9,          # prob. each machine follows the common flip in a shared episode
    "tau_bin": 8,
    "tau_lo": 8,                # earliest flip tick considered
    "size_lags": {"1": 1.0, "2": 1.0, "4": 1.0},
    "size_lag_other": 0.05,
    "size_const": 0.05,         # prior mass for constant size regime (each)
    "size_eps": 0.02,           # regime noise
}


# ----------------------------------------------------------------------------- win model
def _pair_prior(cfg):
    named = {tuple(int(x) for x in k.split(",")): v for k, v in cfg["pairs"].items()}
    allpairs = [(a, b) for a in range(1, MAXLAG + 1) for b in range(a + 1, MAXLAG + 1)]
    others = [p for p in allpairs if p not in named]
    pri = dict(named)
    if cfg["pair_other"] > 0:
        for p in others:
            pri[p] = cfg["pair_other"] / len(others)
    z = math.fsum(pri.values())
    return {p: math.log(v / z) for p, v in pri.items() if v > 0}


class WinMachine:
    """Per-machine sufficient statistics for every pair and every flip tick."""

    def __init__(self, q, cfg, pair_logprior):
        self.q = q
        seq, c = q["seq"], q["cutoff"]
        a0, b0 = cfg["beta_a"], cfg["beta_b"]
        cp = (cfg.get("cell_prior") or {}).get(q["machine_type"])
        if cp:
            def beta_of(spec):
                mu, slope, kappa = spec
                p = 1 / (1 + math.exp(-(mu + slope * q["signal"])))
                k = kappa * cfg.get("kappa_scale", 1.0)
                return (k * p, k * (1 - p))
            BH, BLO = beta_of(cp["hi"]), beta_of(cp["lo"])
        else:
            BH = BLO = None
        self.taus = list(range(cfg["tau_lo"], c + 1))  # tau == c: flip right now (no evidence)
        self.pairs = list(pair_logprior)
        self.lml_stable = {}   # pair -> log marginal likelihood (no flip)
        self.lml_tau = {}      # pair -> list over taus
        self.pred_stable = {}  # pair -> predicted win prob at cutoff (no flip)
        self.pred_tau = {}     # pair -> list over taus
        base_w = (sum(1 for v in seq.values() if v > 0) + 1) / (len(seq) + 2)
        for p in self.pairs:
            A, B = p
            ticks = sorted(t for t in seq if (t - A) in seq and (t - B) in seq)
            # cum counts: before index i -> counts[z][y]
            n = len(ticks)
            cz = [[[0, 0], [0, 0]]]  # cz[i][zidx][y]
            hl = cfg.get("half_life")
            for t in ticks:
                z = 1 if (seq[t - A] > 0) == (seq[t - B] > 0) else 0
                y = 1 if seq[t] > 0 else 0
                prev = cz[-1]
                nxt = [prev[0][:], prev[1][:]]
                nxt[z][y] += 2.0 ** (-(c - 1 - t) / hl) if hl else 1
                cz.append(nxt)
            tot = cz[-1]
            # state at cutoff
            if (c - A) in seq and (c - B) in seq:
                zc_probs = {1 if (seq[c - A] > 0) == (seq[c - B] > 0) else 0: 1.0}
            else:
                # marginalize unknown lag values with base win rate
                xa = [(seq[c - A] > 0, 1.0)] if (c - A) in seq else [(True, base_w), (False, 1 - base_w)]
                xb = [(seq[c - B] > 0, 1.0)] if (c - B) in seq else [(True, base_w), (False, 1 - base_w)]
                zc_probs = defaultdict(float)
                for va, pa in xa:
                    for vb, pb in xb:
                        zc_probs[1 if va == vb else 0] += pa * pb

            if BH is None:
                def lml(cells):
                    return math.fsum(betaln(a0 + w, b0 + l) - betaln(a0, b0) for w, l in cells)

                def pred(cells, flipped):
                    # cells[k] = [losses, wins] of effective cell k (effective = z if not flipped else 1-z)
                    out = 0.0
                    for zc, pz in zc_probs.items():
                        e = (1 - zc) if flipped else zc
                        l, w = cells[e]
                        out += pz * (a0 + w) / (a0 + b0 + w + l)
                    return out
            else:
                def bb(pr, w, l):
                    return betaln(pr[0] + w, pr[1] + l) - betaln(pr[0], pr[1])

                def assign(cells):
                    # cells given as [(w, l), (w, l)]; returns log ML and P(cell1 is hi)
                    (w0, l0), (w1, l1) = cells
                    x = bb(BH, w1, l1) + bb(BLO, w0, l0)   # cell1 hi
                    y = bb(BLO, w1, l1) + bb(BH, w0, l0)   # cell0 hi
                    m = max(x, y)
                    return m + math.log(0.5 * math.exp(x - m) + 0.5 * math.exp(y - m)), 1 / (1 + math.exp(y - x))

                def lml(cells):
                    return assign(cells)[0]

                def pred(cells, flipped):
                    (l0, w0), (l1, w1) = cells
                    _, p1hi = assign([(w0, l0), (w1, l1)])
                    out = 0.0
                    for zc, pz in zc_probs.items():
                        e = (1 - zc) if flipped else zc
                        l, w = cells[e]
                        mh = (BH[0] + w) / (BH[0] + BH[1] + w + l)
                        ml = (BLO[0] + w) / (BLO[0] + BLO[1] + w + l)
                        phi = p1hi if e == 1 else 1 - p1hi
                        out += pz * (phi * mh + (1 - phi) * ml)
                    return out

            st_cells = [tot[0], tot[1]]
            self.lml_stable[p] = lml([(st_cells[0][1], st_cells[0][0]), (st_cells[1][1], st_cells[1][0])])
            self.pred_stable[p] = pred(st_cells, False)
            lt, pt = [], []
            j = 0
            for tau in self.taus:
                while j < n and ticks[j] < tau:
                    j += 1
                pre = cz[j]
                post = [[tot[0][0] - pre[0][0], tot[0][1] - pre[0][1]], [tot[1][0] - pre[1][0], tot[1][1] - pre[1][1]]]
                # effective cell e: before tau e=z, after tau e=1-z
                cells = [[pre[0][0] + post[1][0], pre[0][1] + post[1][1]],
                         [pre[1][0] + post[0][0], pre[1][1] + post[0][1]]]
                lt.append(lml([(cells[0][1], cells[0][0]), (cells[1][1], cells[1][0])]))
                pt.append(pred(cells, True))
            self.lml_tau[p] = lt
            self.pred_tau[p] = pt
        self.pair_logprior = pair_logprior

    def mix_stable(self):
        """log L(stable) and predictive under stable."""
        ls = [self.pair_logprior[p] + self.lml_stable[p] for p in self.pairs]
        tot = logsumexp(ls)
        pr = math.fsum(math.exp(l - tot) * self.pred_stable[p] for l, p in zip(ls, self.pairs))
        return tot, pr

    def mix_tau(self, i):
        ls = [self.pair_logprior[p] + self.lml_tau[p][i] for p in self.pairs]
        tot = logsumexp(ls)
        pr = math.fsum(math.exp(l - tot) * self.pred_tau[p][i] for l, p in zip(ls, self.pairs))
        return tot, pr


def tau_logprior(taus, cutoff, hist, bw):
    """Prior over flip tick from a histogram over (cutoff - tau) // bw; flat if hist is None."""
    ws = []
    for tau in taus:
        ws.append(hist.get(str((cutoff - tau) // bw), hist.get("rest", 1e-3)) if hist else 1.0)
    z = math.fsum(ws)
    return [math.log(w / z) for w in ws]


def is_short_episode(machines):
    """Short-history episodes: observed ticks far fewer than cutoff (40 recent + 24 sampled)."""
    n = sum(len(m.q["seq"]) for m in machines)
    c = sum(m.q["cutoff"] for m in machines)
    return n < 0.6 * c


def episode_posterior(machines, cfg):
    """Episode-level mixture: stable / drift (independent flips) / shared (common flip).

    Returns (per-machine win predictions, diagnostics dict with model posteriors and tau posteriors)."""
    bw = cfg["tau_bin"]
    infos = []
    for m in machines:
        ls, ps = m.mix_stable()
        lt, pt = [], []
        for i in range(len(m.taus)):
            l, p = m.mix_tau(i)
            lt.append(l)
            pt.append(p)
        infos.append(dict(ls=ls, ps=ps, lt=lt, pt=pt))
    cut = machines[0].q["cutoff"]
    same_cut = all(m.q["cutoff"] == cut for m in machines)
    pri = cfg["model_prior"]
    if is_short_episode(machines) and cfg.get("model_prior_short"):
        pri = cfg["model_prior_short"]
    logs = {}
    preds = {}
    post = {}
    # stable
    logs["stable"] = math.fsum(i["ls"] for i in infos)
    preds["stable"] = [i["ps"] for i in infos]
    # drift: independent flip per machine
    rho = cfg["drift_rho"]
    ld, pd_, tpost_d = 0.0, [], []
    for m, i in zip(machines, infos):
        tl = tau_logprior(m.taus, m.q["cutoff"], cfg.get("tau_hist_drift"), bw)
        terms = [math.log(rho) + a + b for a, b in zip(tl, i["lt"])]
        lchange = logsumexp(terms)
        lst = math.log(1 - rho) + i["ls"] if rho < 1 else -math.inf
        lm = logsumexp([lst, lchange])
        ld += lm
        wst = math.exp(lst - lm) if lst > -math.inf else 0.0
        tp = [math.exp(t - lm) for t in terms]
        tpost_d.append(tp)
        pd_.append(wst * i["ps"] + math.fsum(w * p for w, p in zip(tp, i["pt"])))
    logs["drift"], preds["drift"] = ld, pd_
    # shared: common flip tick; each machine flips with prob rho_s
    if same_cut:
        rs = cfg["shared_rho"]
        tl = tau_logprior(machines[0].taus, cut, cfg.get("tau_hist_shared"), bw)
        per_tau = []  # per tau: (log joint, per-machine predictive)
        for k in range(len(tl)):
            tot = tl[k]
            pk = []
            for i in infos:
                a = math.log(rs) + i["lt"][k]
                b = math.log(1 - rs) + i["ls"] if rs < 1 else -math.inf
                lm = logsumexp([a, b])
                tot += lm
                wa = math.exp(a - lm)
                pk.append(wa * i["pt"][k] + (1 - wa) * i["ps"])
            per_tau.append((tot, pk))
        L = logsumexp([t for t, _ in per_tau])
        tpost_s = [math.exp(t - L) for t, _ in per_tau]
        logs["shared"] = L
        preds["shared"] = [math.fsum(w * pk[j] for w, (_, pk) in zip(tpost_s, per_tau)) for j in range(len(infos))]
    else:
        tpost_s = None
    names = [n for n in logs if pri.get(n, 0) > 0]
    lj = {n: math.log(pri[n]) + logs[n] for n in names}
    LE = logsumexp(list(lj.values()))
    for n in names:
        post[n] = math.exp(lj[n] - LE)
    out = [math.fsum(post[n] * preds[n][j] for n in names) for j in range(len(infos))]
    return out, {"post": post, "tpost_drift": tpost_d, "tpost_shared": tpost_s, "taus": machines[0].taus}


# ----------------------------------------------------------------------------- size model
def predict_size(q, vecs, cfg):
    seq, c = q["seq"], q["cutoff"]
    V = vecs[q["machine_type"]]
    eps = cfg["size_eps"]
    # regime-specific log probs with noise: P(size | r) = (1-eps) V_r + eps V_{1-r}
    Vn = [[(1 - eps) * V[r][k] + eps * V[1 - r][k] for k in range(4)] for r in (0, 1)]
    LV = [[math.log(x) for x in v] for v in Vn]
    wins = [(t, SIZE_IDX[v]) for t, v in seq.items() if v > 0]
    hyps = []  # (logprior + loglik, predictive regime-1 probability)
    # constant regimes
    for r in (0, 1):
        ll = math.fsum(LV[r][k] for _, k in wins)
        hyps.append((math.log(cfg["size_const"]) + ll, float(r)))
    lag_pri = {int(k): v for k, v in cfg["size_lags"].items()}
    others = [L for L in range(1, MAXLAG + 1) if L not in lag_pri]
    for L in others:
        lag_pri[L] = cfg["size_lag_other"] / max(1, len(others))
    z = math.fsum(lag_pri.values())
    for L, pw in lag_pri.items():
        if pw <= 0:
            continue
        xs = [(seq[t - L] >= 2, k) for t, k in wins if (t - L) in seq]
        if (c - L) in seq:
            big_c = {seq[c - L] >= 2: 1.0}
        else:
            nb = sum(1 for v in seq.values() if v >= 2)
            pb = (nb + 1) / (len(seq) + 2)
            big_c = {True: pb, False: 1 - pb}
        for s in (0, 1):
            ll = math.fsum(LV[int(b) ^ s][k] for b, k in xs)
            pr1 = math.fsum(pp for b, pp in big_c.items() if (int(b) ^ s) == 1)
            hyps.append((math.log(pw / z * 0.5) + ll, pr1))
    tot = logsumexp([h[0] for h in hyps])
    r1 = math.fsum(math.exp(h[0] - tot) * h[1] for h in hyps)
    # predictive size distribution: regime posterior mixed with noise
    return [(1 - r1) * Vn[0][k] + r1 * Vn[1][k] for k in range(4)]


# ----------------------------------------------------------------------------- full predictor
def predict(queries, vecs, cfg=None):
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    pair_lp = _pair_prior(cfg)
    by_ep = defaultdict(list)
    for q in queries:
        by_ep[q["episode_id"]].append(q)
    out = {}
    for ep, qs in sorted(by_ep.items()):
        qs = sorted(qs, key=lambda q: q["machine_id"])
        ms = [WinMachine(q, cfg, pair_lp) for q in qs]
        ws, _ = episode_posterior(ms, cfg)
        for q, w in zip(qs, ws):
            w = min(max(w, 1e-6), 1 - 1e-6)
            sz = predict_size(q, vecs, cfg)
            out[q["query_id"]] = [1 - w] + [w * s for s in sz]
    return out


# ----------------------------------------------------------------------------- scoring
def official_score(pred, extra, queries, max_points=60.0):
    qmap = {q["query_id"]: q for q in queries}
    cond = {r["query_id"]: r["condition"] for r in extra["conditions"]}
    ref = {r["query_id"]: [float(r[k]) for k in ("p_0", "p_1", "p_2", "p_5", "p_20")] for r in extra["reference"]}
    E = defaultdict(lambda: defaultdict(list))
    Bm = defaultdict(lambda: defaultdict(list))
    for t in extra["targets"]:
        qid = t["query_id"]
        tv = [float(t[k]) for k in ("p_0", "p_1", "p_2", "p_5", "p_20")]
        p = pred[qid]
        E[cond[qid]][t["episode_id"]].append(math.fsum((a - b) ** 2 for a, b in zip(p, tv)))
        Bm[cond[qid]][t["episode_id"]].append(math.fsum((a - b) ** 2 for a, b in zip(ref[qid], tv)))

    def agg(D):
        return {c: math.fsum(math.fsum(v) / len(v) for v in eps.values()) / len(eps) for c, eps in D.items()}
    e, b = agg(E), agg(Bm)
    ratios = {c: e[c] / max(b[c], 1e-9) for c in e}
    R = math.fsum(ratios.values()) / len(ratios)
    return {"points": max_points / (1 + R), "R": R, "ratios": {c: round(v, 4) for c, v in sorted(ratios.items())},
            "err": {c: round(v, 5) for c, v in sorted(e.items())}}
