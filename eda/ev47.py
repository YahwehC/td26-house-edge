"""Lag-3 coupling: win count vs payout-based features of the other machines at t-3 (train+val ticks)."""
import sys, math
from collections import defaultdict
sys.path.insert(0, '.')
from he.model import load_split
import he.coupling as hc
X, Y, O = [], [], []
for split in sys.argv[1:]:
    qs, ex = load_split('kit/public', split)
    cm = {r["query_id"]: r["condition"] for r in ex["conditions"]}
    by = defaultdict(list)
    for q in qs: by[q["episode_id"]].append(q)
    for e in sorted(by):
        ms = sorted(by[e], key=lambda q: q["machine_id"])
        c0 = cm[ms[0]["query_id"]]
        if c0 == "short_history" or any(len(m["seq"]) < 0.6 * m["cutoff"] for m in ms): continue
        T = min(m["cutoff"] for m in ms); tmax = T if c0 == "stable" else T - 110
        if tmax < 60: continue
        P = [[m["seq"].get(t, 0) for t in range(T)] for m in ms]
        for i, m in enumerate(ms):
            sep, (A, B), n = hc._best_pair(m["seq"], tmax, hc.PAIRS)
            if sep < 0.3: continue
            s = m["seq"]
            for t in range(20, tmax):
                if t not in s or (t - A) not in s or (t - B) not in s: continue
                z = int((s[t - A] > 0) == (s[t - B] > 0)); y = int(s[t] > 0)
                r = (n[z][1] - y + .5) / (sum(n[z]) - 1 + 1)
                O.append(math.log(r / (1 - r))); Y.append(y)
                o = [P[j][t - 3] for j in range(len(ms)) if j != i]
                f = [sum(v == 1 for v in o), sum(v == 2 for v in o), sum(v == 5 for v in o), sum(v == 20 for v in o),
                     sum(v > 0 for v in o), sum(o), sum(math.log(v) for v in o if v > 0),
                     sum(v > 0 for v in [P[j][t - 2] for j in range(len(ms)) if j != i]) * sum(v > 0 for v in o)]
                X.append([float(x) for x in f])
print(len(X), "ticks")
nm = ["n_pay1", "n_pay2", "n_pay5", "n_pay20", "n_win", "sum_pay", "sum_logpay", "x"]
for idx in ([0, 1, 2, 3], [4], [5], [4, 5], [4, 6]):
    b, se, mu = hc._logreg(X, Y, O, idx)
    print(" ".join("%s %+.4f(z%+.1f)" % (nm[i], b[j + 1], b[j + 1] / se[j + 1]) for j, i in enumerate(idx)))
