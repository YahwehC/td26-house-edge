"""Error decomposition on validation: oracle win prob / oracle size distribution; true-prob clustering."""
import sys, math, csv
from collections import defaultdict
sys.path.insert(0, '.')
from he.model import load_split, official_score
from he.api import load_params, predict_queries
qv, ev = load_split('kit/public', 'validation')
P = load_params('runs/main_v3/params.json'); P = dict(P); P.pop('blend', None); P.pop('fallback', None)
pred = predict_queries(qv, P)
tg = {r["query_id"]: [float(r[k]) for k in ("p_0", "p_1", "p_2", "p_5", "p_20")] for r in ev["targets"]}
def mix(wsrc, ssrc):
    out = {}
    for k in pred:
        w = 1 - wsrc[k][0]; sw = 1 - ssrc[k][0]
        s = [x / sw for x in ssrc[k][1:]] if sw > 1e-9 else [0.25] * 4
        out[k] = [1 - w] + [w * x for x in s]
    return out
for name, pr in [("ours", pred), ("oracle win", mix(tg, pred)), ("oracle size", mix(pred, tg))]:
    s = official_score(pr, ev, qv); print(name.ljust(12), "%.3f" % s["points"], s["ratios"], s["err"])
cond = {r["query_id"]: r["condition"] for r in ev["conditions"]}
# true win prob clustering by type x signal (stable only)
by = defaultdict(list)
for q in qv:
    if cond[q["query_id"]] == "stable":
        by[(q["machine_type"], q["signal"])].append(round(1 - tg[q["query_id"]][0], 3))
for k in sorted(by):
    v = sorted(by[k]); print(k, len(v), v)
