"""Stable validation: win-prob residual vs active-cell count; hi/lo coupling within a machine."""
import sys, math
from collections import defaultdict
sys.path.insert(0, '.')
from he.model import load_split
from he.api import load_params, predict_queries
qv, ev = load_split('kit/public', 'validation')
P = load_params('runs/main_v3/params.json'); P = dict(P); P.pop('blend', None); P.pop('fallback', None)
pred = predict_queries(qv, P)
tg = {r["query_id"]: 1 - float(r["p_0"]) for r in ev["targets"]}
cond = {r["query_id"]: r["condition"] for r in ev["conditions"]}
PAIRS = [(1, 3), (2, 5), (4, 7)]
lg = lambda p: math.log(min(max(p, 1e-4), 1 - 1e-4) / (1 - min(max(p, 1e-4), 1 - 1e-4)))
rows = []
for q in qv:
    if cond[q["query_id"]] != "stable":
        continue
    seq, c = q["seq"], q["cutoff"]; best = None
    for A, B in PAIRS:
        n = [[0, 0], [0, 0]]
        for t in seq:
            if (t - A) in seq and (t - B) in seq:
                n[int((seq[t - A] > 0) == (seq[t - B] > 0))][int(seq[t] > 0)] += 1
        r = [(n[z][1] + .5) / (sum(n[z]) + 1) for z in (0, 1)]
        if best is None or abs(r[1] - r[0]) > best[0]:
            best = (abs(r[1] - r[0]), n, (A, B))
    _, n, (A, B) = best
    if (c - A) not in seq or (c - B) not in seq:
        continue
    za = int((seq[c - A] > 0) == (seq[c - B] > 0)); zo = 1 - za
    na, no = sum(n[za]), sum(n[zo])
    ra = (n[za][1] + .5) / (na + 1); ro = (n[zo][1] + .5) / (no + 1)
    tp = tg[q["query_id"]]; ours = 1 - pred[q["query_id"]][0]
    rows.append(dict(type=q["machine_type"], sig=q["signal"], na=na, no=no, ra=ra, ro=ro, tp=tp, ours=ours, hi=tp > 0.5))
print("n", len(rows))
bins = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 400)]
for lo, hi in bins:
    v = [r for r in rows if lo <= r["na"] < hi]
    if not v: continue
    mse = sum((r["ours"] - r["tp"]) ** 2 for r in v) / len(v)
    binom = sum(r["tp"] * (1 - r["tp"]) / (r["na"] + 8) for r in v) / len(v)
    print("n_active [%d,%d): k=%d  MSE(ours-true)=%.5f  approx posterior var=%.5f  share of total SE=%.2f" % (
        lo, hi, len(v), mse, binom, sum((r["ours"] - r["tp"]) ** 2 for r in v) / sum((r["ours"] - r["tp"]) ** 2 for r in rows)))
# hi/lo coupling: residual of logit(true active) vs residual of logit(other cell empirical), by type x class
for ty in ("balanced", "jackpot", "steady"):
    for hi in (True, False):
        v = [r for r in rows if r["type"] == ty and r["hi"] == hi and r["no"] >= 30]
        if len(v) < 6: continue
        x = [lg(r["ro"]) - 0.3 * r["sig"] for r in v]; y = [lg(r["tp"]) - 0.3 * r["sig"] for r in v]
        mx, my = sum(x) / len(x), sum(y) / len(y)
        sxx = sum((a - mx) ** 2 for a in x); syy = sum((b - my) ** 2 for b in y); sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
        print(ty, "active=%s" % ("hi" if hi else "lo"), "k=%d" % len(v), "corr(logit other, logit true)=%.2f slope=%.2f  mean other=%.2f mean true=%.2f sd true=%.2f" % (
            sxy / math.sqrt(sxx * syy + 1e-12), sxy / (sxx + 1e-12), mx, my, math.sqrt(syy / len(v))))
