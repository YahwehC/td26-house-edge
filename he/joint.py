"""Joint win/size model: the size lag is coupled to the win lag pair through P(L | pair).

Per machine, the pair prior is multiplied by the size-evidence factor sum_L P(L|pair) * ML_size(L);
the episode mixture (stable / drift / shared) then yields a pair posterior, which is used to mix
the size-regime predictions. Pure standard library.
"""
import math
from collections import defaultdict

from .model import (DEFAULT_CFG, MAXLAG, SIZE_IDX, WinMachine, _pair_prior, is_short_episode,
                    logsumexp, tau_logprior)


def _mix(m, lmls, preds):
    ls = [m.pair_logprior[p] + l for p, l in zip(m.pairs, lmls)]
    tot = logsumexp(ls)
    ws = [math.exp(l - tot) for l in ls]
    return tot, math.fsum(w * pr for w, pr in zip(ws, preds)), ws


def episode_posterior_joint(machines, cfg):
    """Like model.episode_posterior, additionally returns per-machine pair posteriors."""
    bw = cfg["tau_bin"]
    npair = len(machines[0].pairs)
    infos = []
    for m in machines:
        ls, ps, wst = _mix(m, [m.lml_stable[p] for p in m.pairs], [m.pred_stable[p] for p in m.pairs])
        lt, pt, wt = [], [], []
        for i in range(len(m.taus)):
            l, p, w = _mix(m, [m.lml_tau[pp][i] for pp in m.pairs], [m.pred_tau[pp][i] for pp in m.pairs])
            lt.append(l)
            pt.append(p)
            wt.append(w)
        infos.append(dict(ls=ls, ps=ps, wst=wst, lt=lt, pt=pt, wt=wt))

    def add(acc, ws, f):
        if f > 1e-15:
            for j in range(npair):
                acc[j] += f * ws[j]

    cut = machines[0].q["cutoff"]
    same_cut = all(m.q["cutoff"] == cut for m in machines)
    pri = cfg["model_prior"]
    if is_short_episode(machines) and cfg.get("model_prior_short"):
        pri = cfg["model_prior_short"]
    logs, preds, pws = {}, {}, {}
    logs["stable"] = math.fsum(i["ls"] for i in infos)
    preds["stable"] = [i["ps"] for i in infos]
    pws["stable"] = [i["wst"] for i in infos]
    if pri.get("drift", 0) > 0:
        rho = cfg["drift_rho"]
        ld, pd_, pw_d = 0.0, [], []
        for m, i in zip(machines, infos):
            tl = tau_logprior(m.taus, m.q["cutoff"], cfg.get("tau_hist_drift"), bw)
            terms = [math.log(rho) + a + b for a, b in zip(tl, i["lt"])]
            lst = math.log(1 - rho) + i["ls"] if rho < 1 else -math.inf
            lm = logsumexp([lst, logsumexp(terms)])
            ld += lm
            wst = math.exp(lst - lm) if lst > -math.inf else 0.0
            tp = [math.exp(t - lm) for t in terms]
            pd_.append(wst * i["ps"] + math.fsum(w * p for w, p in zip(tp, i["pt"])))
            acc = [0.0] * npair
            add(acc, i["wst"], wst)
            for w, ws in zip(tp, i["wt"]):
                add(acc, ws, w)
            pw_d.append(acc)
        logs["drift"], preds["drift"], pws["drift"] = ld, pd_, pw_d
    if same_cut and pri.get("shared", 0) > 0:
        rs = cfg["shared_rho"]
        tl = tau_logprior(machines[0].taus, cut, cfg.get("tau_hist_shared"), bw)
        per_tau = []
        for k in range(len(tl)):
            tot = tl[k]
            pk, wak = [], []
            for i in infos:
                a = math.log(rs) + i["lt"][k]
                b = math.log(1 - rs) + i["ls"] if rs < 1 else -math.inf
                lm = logsumexp([a, b])
                tot += lm
                wa = math.exp(a - lm)
                pk.append(wa * i["pt"][k] + (1 - wa) * i["ps"])
                wak.append(wa)
            per_tau.append((tot, pk, wak))
        L = logsumexp([t for t, _, _ in per_tau])
        tps = [math.exp(t - L) for t, _, _ in per_tau]
        logs["shared"] = L
        preds["shared"] = [math.fsum(w * pt[1][j] for w, pt in zip(tps, per_tau)) for j in range(len(infos))]
        pw_s = []
        for j, i in enumerate(infos):
            acc = [0.0] * npair
            for k, w in enumerate(tps):
                if w > 1e-15:
                    wa = per_tau[k][2][j]
                    add(acc, i["wt"][k], w * wa)
                    add(acc, i["wst"], w * (1 - wa))
            pw_s.append(acc)
        pws["shared"] = pw_s
    names = [n for n in logs if pri.get(n, 0) > 0]
    lj = {n: math.log(pri[n]) + logs[n] for n in names}
    LE = logsumexp(list(lj.values()))
    post = {n: math.exp(lj[n] - LE) for n in names}
    out, pair_post = [], []
    for j in range(len(infos)):
        out.append(math.fsum(post[n] * preds[n][j] for n in names))
        acc = [0.0] * npair
        for n in names:
            add(acc, pws[n][j], post[n])
        z = math.fsum(acc)
        pair_post.append({p: a / z for p, a in zip(machines[0].pairs, acc)})
    return out, {"post": post, "pair_post": pair_post}


class SizeEvidence:
    """Per-machine size log-likelihoods for constant regimes and every (lag L, sign s)."""

    def __init__(self, q, vecs, cfg):
        seq, c = q["seq"], q["cutoff"]
        V = vecs[q["machine_type"]]
        eps = cfg["size_eps"]
        self.Vn = [[(1 - eps) * V[r][k] + eps * V[1 - r][k] for k in range(4)] for r in (0, 1)]
        LV = [[math.log(x) for x in v] for v in self.Vn]
        wins = [(t, SIZE_IDX[v]) for t, v in seq.items() if v > 0]
        self.const = [math.fsum(LV[r][k] for _, k in wins) for r in (0, 1)]
        nb = sum(1 for v in seq.values() if v >= 2)
        pb = (nb + 1) / (len(seq) + 2)
        self.lag = {}
        for L in range(1, MAXLAG + 1):
            if cfg.get("size_chain_missing") and len(seq) < 0.6 * c:
                self.lag[L] = [self._chain(seq, c, L, s) for s in (0, 1)]
                continue
            xs = [(seq[t - L] >= 2, k) for t, k in wins if (t - L) in seq]
            big_c = {seq[c - L] >= 2: 1.0} if (c - L) in seq else {True: pb, False: 1 - pb}
            miss = [k for t, k in wins if (t - L) not in seq] if cfg.get("size_marg_missing") else []
            row = []
            for s in (0, 1):
                ll = math.fsum(LV[int(b) ^ s][k] for b, k in xs)
                if miss:
                    # lag payout unobserved (sampled old ticks): marginalize the regime bit with the big-payout rate
                    ll += math.fsum(math.log((1 - pb) * self.Vn[s][k] + pb * self.Vn[1 - s][k]) for k in miss)
                pr1 = math.fsum(pp for b, pp in big_c.items() if (int(b) ^ s) == 1)
                row.append((ll, pr1))
            self.lag[L] = row

    def _chain(self, seq, cutoff, lag, sign):
        """Integrate missing big-payout bits along each residue chain.

        Observed wins are conditioned on; missing wins use the smoothed
        marginal win rate. This is exact for that auxiliary size-chain model,
        not for the full XNOR win process.
        """
        win_rate = (sum(v > 0 for v in seq.values()) + 1) / (len(seq) + 2)
        a = win_rate * (1 - self.Vn[sign][0])
        b = win_rate * (1 - self.Vn[1 - sign][0])
        stationary = a / (1 - b + a)
        ll = 0.0
        pred = stationary
        for residue in range(lag):
            big = stationary
            for t in range(residue, cutoff, lag):
                value = seq.get(t)
                if value is None:
                    big = (1 - big) * a + big * b
                elif value == 0:
                    big = 0.0
                else:
                    k = SIZE_IDX[value]
                    ll += math.log((1 - big) * self.Vn[sign][k] + big * self.Vn[1 - sign][k])
                    big = float(value >= 2)
            if residue == cutoff % lag:
                pred = big
        return ll, pred if sign == 0 else 1 - pred

    def factor_and_r1(self, lag_prior, cfg):
        pc = cfg["size_const"]
        h = [(math.log(pc) + self.const[r], float(r)) for r in (0, 1)]
        for L, pw in lag_prior.items():
            if pw > 0:
                for ll, pr1 in self.lag[L]:
                    h.append((math.log((1 - 2 * pc) * pw * 0.5) + ll, pr1))
        tot = logsumexp([x[0] for x in h])
        return tot, math.fsum(math.exp(x[0] - tot) * x[1] for x in h)


def lag_prior_marginal(cfg):
    pri = {int(k): v for k, v in cfg["size_lags"].items()}
    others = [L for L in range(1, MAXLAG + 1) if L not in pri]
    for L in others:
        pri[L] = cfg["size_lag_other"] / max(1, len(others))
    z = math.fsum(pri.values())
    return {L: v / z for L, v in pri.items() if v > 0}


def lag_prior_for_pair(pair, cfg, marginal):
    row = (cfg.get("size_lag_given_pair") or {}).get(f"{pair[0]},{pair[1]}")
    if not row:
        return marginal
    z = math.fsum(row.values())
    return {int(k): v / z for k, v in row.items() if v > 0}


def predict_joint(queries, vecs, cfg=None):
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    base_lp = _pair_prior(cfg)
    marg = lag_prior_marginal(cfg)
    by_ep = defaultdict(list)
    for q in queries:
        by_ep[q["episode_id"]].append(q)
    out = {}
    for ep, qs in sorted(by_ep.items()):
        qs = sorted(qs, key=lambda q: q["machine_id"])
        short = sum(len(q["seq"]) for q in qs) < 0.6 * sum(q["cutoff"] for q in qs)
        pri = cfg.get("model_prior_short") if short else None
        pri = pri or cfg["model_prior"]
        stable_only = pri.get("stable", 0) > 0 and not any(pri.get(k, 0) > 0 for k in ("drift", "shared"))
        ms, per_pair = [], []
        for q in qs:
            se = SizeEvidence(q, vecs, cfg)
            pp = {p: se.factor_and_r1(lag_prior_for_pair(p, cfg, marg), cfg) for p in base_lp}
            lp = {p: base_lp[p] + pp[p][0] for p in base_lp}
            z = logsumexp(list(lp.values()))
            per_pair.append((se, pp))
            ms.append(WinMachine(q, cfg, {p: v - z for p, v in lp.items()}, stable_only=stable_only))
        ws, diag = episode_posterior_joint(ms, cfg)
        for j, (q, w) in enumerate(zip(qs, ws)):
            w = min(max(w, 1e-6), 1 - 1e-6)
            se, pp = per_pair[j]
            post = diag["pair_post"][j]
            r1 = math.fsum(post[p] * pp[p][1] for p in post)
            sz = [(1 - r1) * se.Vn[0][k] + r1 * se.Vn[1][k] for k in range(4)]
            out[q["query_id"]] = [1 - w] + [w * s for s in sz]
    return out
