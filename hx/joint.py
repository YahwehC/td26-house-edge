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
    return tot, math.fsum(w * pr for w, pr in zip(ws, preds)), ws, [w * pr for w, pr in zip(ws, preds)]


def episode_posterior_joint(machines, cfg):
    """Like model.episode_posterior, additionally returns per-machine pair posteriors."""
    bw = cfg["tau_bin"]
    npair = len(machines[0].pairs)
    infos = []
    for m in machines:
        ls, ps, wst, wstp = _mix(m, [m.lml_stable[p] for p in m.pairs], [m.pred_stable[p] for p in m.pairs])
        lt, pt, wt, wtp = [], [], [], []
        for i in range(len(m.taus)):
            l, p, w, wp = _mix(m, [m.lml_tau[pp][i] for pp in m.pairs], [m.pred_tau[pp][i] for pp in m.pairs])
            lt.append(l)
            pt.append(p)
            wt.append(w)
            wtp.append(wp)
        infos.append(dict(ls=ls, ps=ps, wst=wst, lt=lt, pt=pt, wt=wt, wstp=wstp, wtp=wtp))

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
    pwp = {"stable": [i["wstp"] for i in infos]}
    if pri.get("drift", 0) > 0:
        rho = cfg["drift_rho"]
        ld, pd_, pw_d, pwp_d = 0.0, [], [], []
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
            accp = [0.0] * npair
            add(accp, i["wstp"], wst)
            for w, ws in zip(tp, i["wtp"]):
                add(accp, ws, w)
            pwp_d.append(accp)
        logs["drift"], preds["drift"], pws["drift"] = ld, pd_, pw_d
        pwp["drift"] = pwp_d
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
        pw_s, pwp_s = [], []
        for j, i in enumerate(infos):
            acc = [0.0] * npair
            accp = [0.0] * npair
            for k, w in enumerate(tps):
                if w > 1e-15:
                    wa = per_tau[k][2][j]
                    add(acc, i["wt"][k], w * wa)
                    add(acc, i["wst"], w * (1 - wa))
                    add(accp, i["wtp"][k], w * wa)
                    add(accp, i["wstp"], w * (1 - wa))
            pw_s.append(acc)
            pwp_s.append(accp)
        pws["shared"] = pw_s
        pwp["shared"] = pwp_s
    names = [n for n in logs if pri.get(n, 0) > 0]
    lj = {n: math.log(pri[n]) + logs[n] for n in names}
    LE = logsumexp(list(lj.values()))
    post = {n: math.exp(lj[n] - LE) for n in names}
    out, pair_post, pair_wp = [], [], []
    for j in range(len(infos)):
        out.append(math.fsum(post[n] * preds[n][j] for n in names))
        acc = [0.0] * npair
        for n in names:
            add(acc, pws[n][j], post[n])
        z = math.fsum(acc)
        pair_post.append({p: a / z for p, a in zip(machines[0].pairs, acc)})
        accp = [0.0] * npair
        for n in names:
            add(accp, pwp[n][j], post[n])
        pair_wp.append({p: a / z for p, a in zip(machines[0].pairs, accp)})
    return out, {"post": post, "pair_post": pair_post, "pair_wp": pair_wp}


class SizeEvidence:
    """Per-machine size log-likelihoods for constant regimes and every (lag L, sign s)."""

    def __init__(self, q, vecs, cfg):
        seq, c = q["seq"], q["cutoff"]
        V = vecs[q["machine_type"]]
        eps = cfg["size_eps"]
        self.Vn = [[(1 - eps) * V[r][k] + eps * V[1 - r][k] for k in range(4)] for r in (0, 1)]
        LV = [[math.log(x) for x in v] for v in self.Vn]
        wins = [(t, SIZE_IDX[v]) for t, v in seq.items() if v > 0]
        if cfg.get("common_support_size"):
            wins = [(t, k) for t, k in wins if all((t - L) in seq for L in range(1, MAXLAG + 1))]
        self.const = [math.fsum(LV[r][k] for _, k in wins) for r in (0, 1)]
        nb = sum(1 for v in seq.values() if v >= 2)
        pb = (nb + 1) / (len(seq) + 2)
        self.lag = {}
        for L in range(1, MAXLAG + 1):
            xs = [(seq[t - L] >= 2, k) for t, k in wins if (t - L) in seq]
            big_c = {seq[c - L] >= 2: 1.0} if (c - L) in seq else {True: pb, False: 1 - pb}
            miss = [k for t, k in wins if (t - L) not in seq] if cfg.get("size_marg_missing") else []
            row = []
            for s in (0, 1):
                ll = math.fsum(LV[int(b) ^ s][k] for b, k in xs)
                if miss:
                    # lag payout unobserved: marginalize the regime bit with the machine's big-payout rate
                    ll += math.fsum(math.log((1 - pb) * self.Vn[s][k] + pb * self.Vn[1 - s][k]) for k in miss)
                pr1 = math.fsum(pp for b, pp in big_c.items() if (int(b) ^ s) == 1)
                row.append((ll, pr1))
            self.lag[L] = row

    def factor_and_r1(self, lag_prior, cfg):
        pc = cfg["size_const"]
        h = [(math.log(pc) + self.const[r], float(r)) for r in (0, 1)]
        for L, pw in lag_prior.items():
            if pw > 0:
                p1 = cfg.get("size_s1_prior", 0.5)
                for s, (ll, pr1) in enumerate(self.lag[L]):
                    h.append((math.log((1 - 2 * pc) * pw * (p1 if s == 1 else 1 - p1)) + ll, pr1))
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
        ms, per_pair = [], []
        for q in qs:
            se = SizeEvidence(q, vecs, cfg)
            pp = {p: se.factor_and_r1(lag_prior_for_pair(p, cfg, marg), cfg) for p in base_lp}
            lp = {p: base_lp[p] + pp[p][0] for p in base_lp}
            z = logsumexp(list(lp.values()))
            per_pair.append((se, pp))
            ms.append(WinMachine(q, cfg, {p: v - z for p, v in lp.items()}))
        ws, diag = episode_posterior_joint(ms, cfg)
        for j, (q, w) in enumerate(zip(qs, ws)):
            w = min(max(w, 1e-6), 1 - 1e-6)
            se, pp = per_pair[j]
            post = diag["pair_post"][j]
            if cfg.get("joint_cov"):
                wp = diag["pair_wp"][j]
                zz = math.fsum(wp.values())
                r1 = math.fsum(wp[p] * pp[p][1] for p in wp) / zz if zz > 0 else 0.5
            else:
                r1 = math.fsum(post[p] * pp[p][1] for p in post)
            sz = [(1 - r1) * se.Vn[0][k] + r1 * se.Vn[1][k] for k in range(4)]
            out[q["query_id"]] = [1 - w] + [w * s for s in sz]
    return out
