"""Is the lag-3 coupling specific? Logistic with offset on per-tick rows: others' win at t-3 split by
relative machine index d = (j - i) mod 8, and by the other machine's type / signal."""
import sys, math
from collections import defaultdict
sys.path.insert(0, '.')
from he.model import load_split
import he.coupling as hc
X, Y, O = [], [], []
TYPES = ["balanced", "jackpot", "steady"]
for split in sys.argv[1:]:
    qs, ex = load_split('kit/public', split)
    cm = {r["query_id"]: r["condition"] for r in ex["conditions"]}
    by = defaultdict(list)
    for q in qs: by[q["episode_id"]].append(q)
    for e in sorted(by):
        ms = sorted(by[e], key=lambda q: q["machine_id"])
        c0 = cm[ms[0]["query_id"]]
        if c0 == "short_history" or any(len(m["seq"]) < 0.6 * m["cutoff"] for m in ms) or len(ms) != 8: continue
        T = min(m["cutoff"] for m in ms); tmax = T if c0 == "stable" else T - 110
        if tmax < 60: continue
        W = [[int(m["seq"].get(t, 0) > 0) for t in range(T)] for m in ms]
        for i, m in enumerate(ms):
            sep, (A, B), n = hc._best_pair(m["seq"], tmax, hc.PAIRS)
            if sep < 0.3: continue
            s = m["seq"]
            for t in range(20, tmax):
                if t not in s or (t - A) not in s or (t - B) not in s: continue
                z = int((s[t - A] > 0) == (s[t - B] > 0)); y = int(s[t] > 0)
                r = (n[z][1] - y + .5) / (sum(n[z]) - 1 + 1)
                O.append(math.log(r / (1 - r))); Y.append(y)
                f = [float(W[(i + d) % 8][t - 3]) for d in range(1, 8)]
                f += [float(sum(W[j][t - 3] for j in range(8) if j != i and ms[j]["machine_type"] == ty)) for ty in TYPES]
                f += [float(sum(W[j][t - 3] for j in range(8) if j != i and ms[j]["signal"] == sg)) for sg in (0, 1)]
                f += [float(sum(W[j][t - 3] for j in range(8) if j != i and ms[j]["machine_type"] == m["machine_type"]))]
                f += [float(sum(W[j][t - 3] for j in range(8) if j != i))]
                X.append(f)
print(len(X), "ticks")
b, se, mu = hc._logreg(X, Y, O, list(range(7)))
print("by rel index d:", " ".join("d%d %+.3f(z%+.1f)" % (d + 1, b[d + 1], b[d + 1] / se[d + 1]) for d in range(7)))
b, se, mu = hc._logreg(X, Y, O, [7, 8, 9])
print("by other type:", " ".join("%s %+.3f(z%+.1f)" % (TYPES[j], b[j + 1], b[j + 1] / se[j + 1]) for j in range(3)))
b, se, mu = hc._logreg(X, Y, O, [10, 11])
print("by other signal:", " ".join("s%d %+.3f(z%+.1f)" % (j, b[j + 1], b[j + 1] / se[j + 1]) for j in range(2)))
b, se, mu = hc._logreg(X, Y, O, [12, 13])
print("same-type vs all:", "same %+.3f(z%+.1f) all %+.3f(z%+.1f)" % (b[1], b[1] / se[1], b[2], b[2] / se[2]))
