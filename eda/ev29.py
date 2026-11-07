"""2-fold CV (by episode) of a logit correction w' = sigmoid(logit(w) + b.x) using payout-history features."""
import sys, math
from collections import defaultdict
sys.path.insert(0, '.')
from he.model import load_split, official_score
from he.api import load_params, predict_queries
qv, ev = load_split('kit/public', 'validation')
P = load_params('runs/main_v4/params.json')
pred = predict_queries(qv, P)           # full main_v4 (blend 0.9)
tg = {r["query_id"]: 1 - float(r["p_0"]) for r in ev["targets"]}
lg = lambda p: math.log(min(max(p, 1e-6), 1 - 1e-6) / (1 - min(max(p, 1e-6), 1 - 1e-6)))
ep = defaultdict(list)
for q in qv: ep[q["episode_id"]].append(q)
def feats(q):
    seq, c = q["seq"], q["cutoff"]; mates = ep[q["episode_id"]]
    f = {}
    for K in (8, 16, 32):
        f["ep_net_%d" % K] = sum(sum(m["seq"].get(c - k, 1) - 1 for k in range(1, K + 1)) for m in mates) / (5 * K)
        f["ep_wins_%d" % K] = sum(sum(int(m["seq"].get(c - k, 0) > 0) for k in range(1, K + 1)) for m in mates) / (8 * K)
        f["own_net_%d" % K] = sum(seq.get(c - k, 1) - 1 for k in range(1, K + 1)) / K
        f["oth_net_%d" % K] = sum(sum(m["seq"].get(c - k, 1) - 1 for k in range(1, K + 1)) for m in mates if m is not q) / (5 * K)
    for hl in (2, 4, 8, 16):
        f["ep_ew_%d" % hl] = sum(sum((m["seq"].get(c - k, 1) - 1) * 0.5 ** ((k - 1) / hl) for k in range(1, 65)) for m in mates) / (5 * hl)
        f["ep_ewwin_%d" % hl] = sum(sum((int(m["seq"].get(c - k, 0) > 0) - 0.5) * 0.5 ** ((k - 1) / hl) for k in range(1, 65)) for m in mates) / hl
    for K in (8, 16):
        f["ep_big5_%d" % K] = sum(sum(int(m["seq"].get(c - k, 0) >= 5) for k in range(1, K + 1)) for m in mates) / K
        f["ep_j20_%d" % K] = sum(sum(int(m["seq"].get(c - k, 0) == 20) for k in range(1, K + 1)) for m in mates) / K
        f["ep_pay2_%d" % K] = sum(sum(int(m["seq"].get(c - k, 0) == 2) for k in range(1, K + 1)) for m in mates) / K
    return f
F = {q["query_id"]: feats(q) for q in qv}
eps = sorted(ep); fold = {e: i % 2 for i, e in enumerate(eps)}
def apply(names, beta, qids):
    out = {}
    for k in qids:
        p = pred[k]; w = 1 - p[0]
        z = lg(w) + sum(b * (F[k][n] - mu[n]) for b, n in zip(beta, names))
        w2 = 1 / (1 + math.exp(-z)); s = [x / w for x in p[1:]] if w > 1e-12 else [0.25] * 4
        out[k] = [1 - w2] + [w2 * x for x in s]
    return out
def fit(names, qids):
    # least squares on win-prob error via a few Gauss-Newton steps
    beta = [0.0] * len(names)
    for _ in range(8):
        JtJ = [[0.0] * len(names) for _ in names]; Jtr = [0.0] * len(names)
        for k in qids:
            w = 1 - pred[k][0]; z = lg(w) + sum(b * (F[k][n] - mu[n]) for b, n in zip(beta, names))
            w2 = 1 / (1 + math.exp(-z)); d = w2 * (1 - w2)
            # error in 5-vector: only scale changes; approx with win component weight 2 (p0 and sum of rest)
            r = w2 - tg[k]; g = [d * (F[k][n] - mu[n]) for n in names]
            for i in range(len(names)):
                Jtr[i] += g[i] * r
                for j in range(len(names)): JtJ[i][j] += g[i] * g[j]
        # solve small system (Gaussian elimination) with ridge
        n = len(names); A = [JtJ[i][:] + [Jtr[i]] for i in range(n)]
        for i in range(n): A[i][i] += 1e-3
        for i in range(n):
            piv = A[i][i]
            for j in range(i, n + 1): A[i][j] /= piv
            for r2 in range(n):
                if r2 != i:
                    fct = A[r2][i]
                    for j in range(i, n + 1): A[r2][j] -= fct * A[i][j]
        beta = [b - A[i][n] for i, b in enumerate(beta)]
    return beta
mu = {n: sum(F[k][n] for k in F) / len(F) for n in next(iter(F.values()))}
base = official_score(pred, ev, qv)
print("main_v4 val %.3f" % base["points"], base["ratios"])
for names in (["ep_net_8"], ["ep_ew_2"], ["ep_ew_4"], ["ep_ew_8"], ["ep_ew_16"], ["ep_ewwin_4"], ["ep_big5_8"], ["ep_big5_16"], ["ep_j20_16"], ["ep_pay2_16"], ["ep_ewwin_4", "ep_j20_16"], ["ep_ew_4", "ep_ewwin_4"]):
    cvpred = {}
    betas = []
    for fo in (0, 1):
        tr = [q["query_id"] for q in qv if fold[q["episode_id"]] != fo]
        te = [q["query_id"] for q in qv if fold[q["episode_id"]] == fo]
        b = fit(names, tr); betas.append(b)
        cvpred.update(apply(names, b, te))
    s = official_score(cvpred, ev, qv)
    print("%-28s CV %.3f (%+.3f)  betas %s  %s" % ("+".join(names), s["points"], s["points"] - base["points"],
          [[round(x, 3) for x in b] for b in betas], s["ratios"]))
