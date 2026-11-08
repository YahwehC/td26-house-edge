"""Size sign s: is it shared within an episode or predictable from signal/type/pair/machine index? (train+val, non-short)"""
import sys, math, json
from collections import defaultdict, Counter
sys.path.insert(0, '.')
from he.model import load_split, DEFAULT_CFG
from he.joint import SizeEvidence
import he.coupling as hc
P = json.load(open('runs/main_v6/params.json')); cfg = dict(DEFAULT_CFG, **P["cfg"])
rows = []
for split in ("train", "validation"):
    qs, ex = load_split('kit/public', split)
    cm = {r["query_id"]: r["condition"] for r in ex["conditions"]}
    by = defaultdict(list)
    for q in qs: by[q["episode_id"]].append(q)
    for e, ms in by.items():
        if cm[ms[0]["query_id"]] == "short_history": continue
        for m in sorted(ms, key=lambda q: q["machine_id"]):
            sep, (A, B), n = hc._best_pair(m["seq"], m["cutoff"], hc.PAIRS)
            if sep < 0.3: continue
            se = SizeEvidence(m, P["size_vectors"], cfg)
            llr = se.lag[A][1][0] - se.lag[A][0][0]
            rows.append((e, m["machine_id"][-1], m["machine_type"], m["signal"], (A, B), llr, cm[m["query_id"]]))
conf = [r for r in rows if abs(r[5]) > 4]
print(len(rows), "machines,", len(conf), "with |LLR|>4; P(s=1) = %.3f" % (sum(r[5] > 0 for r in conf) / len(conf)))
for key, nm in ((2, "type"), (3, "signal"), (4, "pair"), (6, "cond"), (1, "machine idx")):
    c = defaultdict(lambda: [0, 0])
    for r in conf: c[r[key]][int(r[5] > 0)] += 1
    print(nm, {k: "%d/%d=%.2f" % (v[1], sum(v), v[1] / sum(v)) for k, v in sorted(c.items())})
E = defaultdict(list)
for r in conf: E[r[0]].append(int(r[5] > 0))
same = tot = 0
for v in E.values():
    for i in range(len(v)):
        for j in range(i): tot += 1; same += v[i] == v[j]
p = sum(r[5] > 0 for r in conf) / len(conf)
print("within-episode same-s rate %.3f vs chance %.3f (%d pairs)" % (same / tot, p * p + (1 - p) ** 2, tot))
print("episode s-counts sample:", [sum(v) * 1.0 / len(v) for v in list(E.values())[:20]])
