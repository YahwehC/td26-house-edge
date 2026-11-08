"""Does the size sign s flip in drift / shared_change episodes? Compare s evidence early (t < c-110) vs late (t >= c-30)."""
import sys, math, json
from collections import defaultdict
sys.path.insert(0, '.')
from he.model import load_split, DEFAULT_CFG
from he.joint import SizeEvidence
import he.coupling as hc
P = json.load(open('runs/main_v6/params.json')); cfg = dict(DEFAULT_CFG, **P["cfg"]); cfg["size_marg_missing"] = False
agree = defaultdict(lambda: [0, 0])
for split in ("train", "validation"):
    qs, ex = load_split('kit/public', split)
    cm = {r["query_id"]: r["condition"] for r in ex["conditions"]}
    for m in qs:
        cd = cm[m["query_id"]]
        if cd == "short_history": continue
        c = m["cutoff"]
        sep, (A, B), n = hc._best_pair(m["seq"], c - 110, hc.PAIRS)
        if sep < 0.3: continue
        def llr(lo, hi):
            q = dict(m); q["seq"] = {t: v for t, v in m["seq"].items() if lo <= t < hi}
            q["cutoff"] = hi
            se = SizeEvidence(q, P["size_vectors"], cfg); return se.lag[A][1][0] - se.lag[A][0][0]
        a, b = llr(0, c - 110), llr(c - 30, c)
        if abs(a) > 3 and abs(b) > 1.5:
            agree[cd][int((a > 0) == (b > 0))] += 1
for cd, v in agree.items(): print("%-14s early/late s agree %d/%d = %.2f" % (cd, v[1], sum(v), v[1] / sum(v)))
