"""Tick-level heat: residual (win - plug-in cell rate) of stable machines vs episode net payout in the K ticks before t."""
import sys, math
from collections import defaultdict
sys.path.insert(0, '.')
from he.model import load_split
PAIRS = [(1, 3), (2, 5), (4, 7)]
def plug(seq):
    best = None
    for A, B in PAIRS:
        n = [[0, 0], [0, 0]]
        for t in seq:
            if (t - A) in seq and (t - B) in seq:
                n[int((seq[t - A] > 0) == (seq[t - B] > 0))][int(seq[t] > 0)] += 1
        r = [(n[z][1] + .5) / (sum(n[z]) + 1) for z in (0, 1)]
        if best is None or abs(r[1] - r[0]) > best[0]:
            best = (abs(r[1] - r[0]), (A, B), r)
    return best
acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])  # sx, sy, sxx, sxy, n
lgacc = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])
for split in ("train", "validation"):
    qs, ex = load_split('kit/public', split)
    cond = {r["query_id"]: r["condition"] for r in ex["conditions"]}
    ep = defaultdict(list)
    for q in qs: ep[q["episode_id"]].append(q)
    for e, ms in ep.items():
        if cond[ms[0]["query_id"]] != "stable":
            continue
        # per-tick episode net and own net
        T = min(m["cutoff"] for m in ms)
        net = [sum(m["seq"].get(t, 1) - 1 for m in ms) for t in range(T)]
        cs = [0]
        for v in net: cs.append(cs[-1] + v)
        for m in ms:
            sep, (A, B), r = plug(m["seq"]); s = m["seq"]
            if sep < 0.3: continue
            own = [s.get(t, 1) - 1 for t in range(T)]; co = [0]
            for v in own: co.append(co[-1] + v)
            for t in range(40, T):
                if (t - A) not in s or (t - B) not in s: continue
                z = int((s[t - A] > 0) == (s[t - B] > 0)); y = int(s[t] > 0) - r[z]
                for K in (1, 2, 4, 8, 16, 32):
                    epn = (cs[t] - cs[t - K]) / K; on = (co[t] - co[t - K]) / K
                    for name, x in (("ep_%d" % K, epn), ("own_%d" % K, on), ("oth_%d" % K, epn - on)):
                        a = acc[name]; a[0] += x; a[1] += y; a[2] += x * x; a[3] += x * y; a[4] += 1
for name in sorted(acc, key=lambda k: (k.split("_")[0], int(k.split("_")[1]))):
    sx, sy, sxx, sxy, n = acc[name]
    vx = sxx / n - (sx / n) ** 2; b = (sxy / n - sx * sy / n / n) / vx
    se = math.sqrt(0.2 / n / vx)  # y var approx 0.2 (Bernoulli residual)
    print("%-8s slope %+.5f  z %+6.1f  x sd %.3f  effect(1sd) %+.4f  n %d" % (name, b, b / se, vx ** .5, b * vx ** .5, n))
