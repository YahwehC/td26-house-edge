"""Target-level: logit(true win) - logit(main_v5 pred) vs number of OTHER winners at cutoff-k (k=1..8)."""
import sys, math
from collections import defaultdict
sys.path.insert(0, '.')
from he.model import load_split
from he.api import load_params, predict_queries
qv, ev = load_split('kit/public', 'validation')
for tag, drop_corr in (("v4-style (no heat corr)", True), ("main_v5 (with heat corr)", False)):
    P = load_params('runs/main_v5/params.json')
    if drop_corr: P = {k: v for k, v in P.items() if k != "logit_corr"}
    pred = predict_queries(qv, P)
    tg = {r["query_id"]: 1 - float(r["p_0"]) for r in ev["targets"]}
    lg = lambda p: math.log(min(max(p, 1e-5), 1 - 1e-5) / (1 - min(max(p, 1e-5), 1 - 1e-5)))
    ep = defaultdict(list)
    for q in qv: ep[q["episode_id"]].append(q)
    rows = []
    for q in qv:
        tp = tg[q["query_id"]]; ours = 1 - pred[q["query_id"]][0]
        if not (0.01 < tp < 0.99 and 0.01 < ours < 0.99): continue
        c = q["cutoff"]
        f = {k: sum(int(m["seq"].get(c - k, 0) > 0) for m in ep[q["episode_id"]] if m is not q) for k in range(1, 9)}
        rows.append((lg(tp) - lg(ours), f))
    print("==", tag, "n", len(rows))
    for k in range(1, 9):
        x = [r[1][k] for r in rows]; y = [r[0] for r in rows]
        mx, my = sum(x) / len(x), sum(y) / len(y); sxx = sum((a - mx) ** 2 for a in x)
        b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sxx
        res = [c - my - b * (a - mx) for a, c in zip(x, y)]; se = math.sqrt(sum(e * e for e in res) / (len(x) - 2) / sxx)
        print("  others' wins at c-%d: slope %+.4f logit/winner  z %+.1f   (x mean %.2f sd %.2f)" % (k, b, b / se, mx, math.sqrt(sxx / len(x))))
