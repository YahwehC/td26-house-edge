"""Own-lag effects beyond the XNOR cell: logistic with offset, features own_win_k (k=1..8) + oth_win_3, by pair."""
import sys, math
from collections import defaultdict
sys.path.insert(0, '.')
from he.model import load_split
import he.coupling as hc
rows = defaultdict(lambda: ([], [], []))
for split in (sys.argv[1],):
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
        W = [[int(m["seq"].get(t, 0) > 0) for t in range(T)] for m in ms]
        for i, m in enumerate(ms):
            sep, (A, B), n = hc._best_pair(m["seq"], tmax, hc.PAIRS)
            if sep < 0.3: continue
            s = m["seq"]; R = rows[(A, B)]
            for t in range(20, tmax):
                if t not in s or (t - A) not in s or (t - B) not in s: continue
                z = int((s[t - A] > 0) == (s[t - B] > 0)); y = int(s[t] > 0)
                r = (n[z][1] - y + .5) / (sum(n[z]) - 1 + 1)
                R[2].append(math.log(r / (1 - r))); R[1].append(y)
                R[0].append([float(W[i][t - k]) for k in range(1, 9)] + [float(sum(W[j][t - 3] for j in range(len(ms)) if j != i))])
names = ["own_%d" % k for k in range(1, 9)] + ["oth_win_3"]
for pr, (X, Y, O) in sorted(rows.items()):
    idx = [k for k in range(9) if k + 1 not in pr]
    b, se, mu = hc._logreg(X, Y, O, idx)
    print(pr, len(X), " ".join("%s %+.3f(z%+.1f)" % (names[i], b[j + 1], b[j + 1] / se[j + 1]) for j, i in enumerate(idx)))
