"""DeepSeek ideas: (a) slot effect on true win prob (validation targets), (b) transductive priors (fit on more histories)."""
import sys, math
from collections import defaultdict
sys.path.insert(0, '.')
from he.model import load_split, official_score
from he.api import fit_params, predict_queries
qv, ev = load_split('kit/public', 'validation')
tg = {r["query_id"]: 1 - float(r["p_0"]) for r in ev["targets"]}
cond = {r["query_id"]: r["condition"] for r in ev["conditions"]}
# residual of true logit vs type x hi/lo cluster mean, grouped by slot
cl = defaultdict(list)
for q in qv:
    p = min(max(tg[q["query_id"]], 1e-4), 1 - 1e-4); lg = math.log(p / (1 - p))
    cl[(q["machine_type"], lg > 0)].append((q, lg))
res = defaultdict(list)
for k, v in cl.items():
    m = sum(x for _, x in v) / len(v)
    for q, x in v:
        res[q["machine_id"][-4:]].append(x - m)
print("slot mean residual (sd of mean):", {s: "%.2f(%.2f)" % (sum(v) / len(v), (sum((x - sum(v) / len(v)) ** 2 for x in v) / len(v)) ** .5 / len(v) ** .5) for s, v in sorted(res.items())})
for fs in (("train",), ("train", "validation")):
    p = fit_params('kit/public', fit_splits=fs)
    s = official_score(predict_queries(qv, p), ev, qv)
    print(fs, "%.3f" % s["points"], s["ratios"])
