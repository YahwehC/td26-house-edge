"""Residual (win - own cell rate) vs candidate hidden features, stable machines, train+validation."""
import sys, math
from collections import defaultdict
sys.path.insert(0, '.')
from he.model import load_split
PAIRS = [(1, 3), (2, 5), (4, 7)]
feat = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))  # name -> value -> [sum r, sum r^2, n]
nm = 0
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
                    n[int((seq[t - A] > 0) == (seq[t - B] > 0))][int(seq[t] > 0)] += 1
            r = [(n[z][1] + .5) / (sum(n[z]) + 1) for z in (0, 1)]
            if best is None or abs(r[1] - r[0]) > best[0]:
                best = (abs(r[1] - r[0]), r, (A, B))
        sep, rate, (A, B) = best
        if sep < 0.3:
            continue
        nm += 1
        ticks = sorted(seq); net = 0; cum = {}
        for t in ticks:
            cum[t] = net; net += seq[t] - 1
        for t in ticks:
            if t < 24 or any((t - L) not in seq for L in range(1, 25)):
                continue
            z = int((seq[t - A] > 0) == (seq[t - B] > 0)); r = int(seq[t] > 0) - rate[z]
            f = {}
            for L in range(1, 9):
                f["big_lag%d" % L] = int(seq[t - L] >= 2)
            for L in (1, 2, 3):
                f["pay_lag%d" % L] = seq[t - L]
            f["big_lagA"] = int(seq[t - A] >= 2); f["big_lagB"] = int(seq[t - B] >= 2)
            f["jack_last8"] = int(any(seq[t - L] == 20 for L in range(1, 9)))
            f["nwin5"] = sum(int(seq[t - L] > 0) for L in range(1, 6))
            f["nwin20"] = min(sum(int(seq[t - L] > 0) for L in range(1, 21)) // 4, 5)
            f["net20"] = max(-3, min(3, sum(seq[t - L] - 1 for L in range(1, 21)) // 10))
            f["cum_net"] = max(-4, min(4, cum[t] // 40))
            f["t_mod2"] = t % 2; f["t_mod3"] = t % 3; f["t_bin32"] = t // 32
            f["cell_z"] = z
            for k, v in f.items():
                a = feat[k][v]; a[0] += r; a[1] += r * r; a[2] += 1
print("machines", nm)
for k in feat:
    vals = sorted(feat[k].items())
    means = [(v, a[0] / a[2], math.sqrt(max(a[1] / a[2] - (a[0] / a[2]) ** 2, 1e-9) / a[2]), a[2]) for v, a in vals if a[2] >= 200]
    if len(means) < 2: continue
    spread = max(m for _, m, _, _ in means) - min(m for _, m, _, _ in means)
    zmax = max(abs(m) / se for _, m, se, _ in means)
    print("%-11s spread=%.4f zmax=%5.1f  " % (k, spread, zmax) + " ".join("%s:%+.4f(n%d)" % (v, m, n) for v, m, _, n in means))
