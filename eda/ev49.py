"""Win-probability calibration in probability space and a logit-temperature sweep (validation, main_v6)."""
import sys, math, copy
sys.path.insert(0, '.')
from he.model import load_split, official_score
from he.api import predict_queries, load_params
P = load_params('runs/main_v6/params.json')
qs, ex = load_split('kit/public', 'validation')
cond = {r["query_id"]: r["condition"] for r in ex["conditions"]}
pred = predict_queries(qs, P)
T = {t["query_id"]: 1 - float(t["p_0"]) for t in ex["targets"]}
for cd in ("stable", "drift", "shared_change", "short_history"):
    ks = [k for k in pred if cond[k] == cd]
    x = [1 - pred[k][0] for k in ks]; y = [T[k] for k in ks]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sum((a - mx) ** 2 for a in x)
    print("%-14s prob-space slope %.3f  mean pred %.3f true %.3f" % (cd, b, mx, my))
L = lambda p: math.log(p / (1 - p))
for a in (0.95, 1.0, 1.03, 1.06, 1.1, 1.15):
    pr = {}
    for k, p in pred.items():
        w = 1 - p[0]; w2 = 1 / (1 + math.exp(-a * L(w)))
        pr[k] = [1 - w2] + [w2 * v / w for v in p[1:]]
    s = official_score(pr, ex, qs); print("temp %.2f  %.3f %s" % (a, s["points"], s["ratios"]))
