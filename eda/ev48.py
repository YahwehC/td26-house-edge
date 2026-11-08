"""Residual screen on validation targets (main_v6): logit(true win) - logit(pred win) vs features; calibration slope per condition."""
import sys, math
from collections import defaultdict
sys.path.insert(0, '.')
from he.model import load_split
from he.api import predict_queries, load_params
P = load_params('runs/main_v6/params.json')
qs, ex = load_split('kit/public', 'validation')
cond = {r["query_id"]: r["condition"] for r in ex["conditions"]}
pred = predict_queries(qs, P)
T = {t["query_id"]: 1 - float(t["p_0"]) for t in ex["targets"]}
L = lambda p: math.log(max(p, 1e-6) / max(1 - p, 1e-6))
by = defaultdict(list)
for q in qs: by[q["episode_id"]].append(q)
def ols(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs); sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0; res = [y - my - b * (x - mx) for x, y in zip(xs, ys)]
    s2 = sum(r * r for r in res) / (n - 2); return b, b / math.sqrt(s2 / sxx) if sxx else 0
for cd in ("stable", "drift", "shared_change", "short_history"):
    ks = [q for q in qs if cond[q["query_id"]] == cd]
    lp = [L(1 - pred[q["query_id"]][0]) for q in ks]; lt = [L(T[q["query_id"]]) for q in ks]
    b, z = ols(lp, lt)
    print("%-14s n %d  calib slope %.3f (z vs 1: %+.1f)  resid sd %.3f" % (cd, len(ks), b, (b - 1) / (b / z) if z else 0,
          math.sqrt(sum((t - p) ** 2 for t, p in zip(lt, lp)) / len(lp))))
feats = defaultdict(dict)
for q in qs:
    k, c, s = q["query_id"], q["cutoff"], q["seq"]
    mates = [m for m in by[q["episode_id"]] if m is not q]
    for j in range(1, 9):
        if (c - j) in s: feats["own_win_%d" % j][k] = float(s[c - j] > 0); feats["own_big_%d" % j][k] = float(s[c - j] >= 2)
        feats["oth_win_%d" % j][k] = float(sum(m["seq"].get(c - j, 0) > 0 for m in mates))
        feats["oth_big_%d" % j][k] = float(sum(m["seq"].get(c - j, 0) >= 2 for m in mates))
    feats["cutoff"][k] = float(c); feats["signal"][k] = float(q["signal"])
    for p in (2, 3, 4, 5, 6, 7, 8): feats["c_mod%d" % p][k] = float(c % p == 0)
    feats["n_wins_20"][k] = float(sum(s.get(c - j, 0) > 0 for j in range(1, 21)))
    feats["lp"][k] = L(1 - pred[k][0])
res = {q["query_id"]: L(T[q["query_id"]]) - L(1 - pred[q["query_id"]][0]) for q in qs}
out = []
for nm, f in feats.items():
    ks = [k for k in f if k in res]
    b, z = ols([f[k] for k in ks], [res[k] for k in ks]); out.append((abs(z), nm, b, z, len(ks)))
for a, nm, b, z, n in sorted(out, reverse=True)[:14]: print("%-12s b %+.4f z %+.2f n %d" % (nm, b, z, n))
