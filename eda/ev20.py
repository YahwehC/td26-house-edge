"""DeepSeek ideas: is the hi-cell orientation (XNOR=1 cell high?) fixed / correlated within episode / tied to slot?"""
import sys
from collections import defaultdict
sys.path.insert(0, '.')
from he.model import load_split
PAIRS = [(1, 3), (2, 5), (4, 7)]
rows = []
for split in ("train", "validation"):
    qs, ex = load_split('kit/public', split)
    cond = {r["query_id"]: r["condition"] for r in ex["conditions"]}
    for q in qs:
        if cond[q["query_id"]] != "stable":
            continue
        seq = q["seq"]; best = None
        for A, B in PAIRS:
            n = [[0, 0], [0, 0]]
            for t in seq:
                if (t - A) in seq and (t - B) in seq:
                    z = int((seq[t - A] > 0) == (seq[t - B] > 0)); n[z][int(seq[t] > 0)] += 1
            r = [(n[z][1] + .5) / (sum(n[z]) + 1) for z in (0, 1)]
            if best is None or abs(r[1] - r[0]) > best[0]:
                best = (abs(r[1] - r[0]), r)
        rows.append(dict(ep=q["episode_id"], slot=q["machine_id"][-4:], type=q["machine_type"], sig=q["signal"],
                         ori=int(best[1][1] > best[1][0]), sep=best[0]))
rows = [r for r in rows if r["sep"] > 0.3]
print("n", len(rows), "P(XNOR=1 cell is hi) overall %.3f" % (sum(r["ori"] for r in rows) / len(rows)))
for key in ("type", "slot", "sig"):
    g = defaultdict(list)
    for r in rows:
        g[r[key]].append(r["ori"])
    print(key, {k: "%.2f/%d" % (sum(v) / len(v), len(v)) for k, v in sorted(g.items())})
# within-episode agreement vs permutation null
by = defaultdict(list)
for r in rows:
    by[r["ep"]].append(r["ori"])
agree = tot = 0
for v in by.values():
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            agree += v[i] == v[j]; tot += 1
p = sum(r["ori"] for r in rows) / len(rows)
print("within-episode pair agreement %.3f vs independent %.3f (pairs %d)" % (agree / tot, p * p + (1 - p) ** 2, tot))
