"""Joint shape of (hi-cell, lo-cell) empirical logits per type, stable machines, train+validation (no targets)."""
import sys, math
from collections import defaultdict
sys.path.insert(0, '.')
from he.model import load_split
PAIRS = [(1, 3), (2, 5), (4, 7)]
lg = lambda p: math.log(p / (1 - p))
pts = defaultdict(list)
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
                best = (abs(r[1] - r[0]), n)
        sep, n = best
        if sep < 0.3 or min(sum(n[0]), sum(n[1])) < 40:
            continue
        r = [(n[z][1] + .5) / (sum(n[z]) + 1) for z in (0, 1)]
        hi, lo = (0, 1) if r[0] > r[1] else (1, 0)
        pts[q["machine_type"]].append((lg(r[hi]) - 0.3 * q["signal"], lg(r[lo]) - 0.3 * q["signal"], sum(n[hi]), sum(n[lo]), q["signal"]))
for ty, v in sorted(pts.items()):
    x = [a for a, _, _, _, _ in v]; y = [b for _, b, _, _, _ in v]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxx = sum((a - mx) ** 2 for a in x); syy = sum((b - my) ** 2 for b in y); sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    print("%s k=%d  hi: mean %.2f sd %.2f | lo: mean %.2f sd %.2f | corr %.2f slope(lo on hi) %.2f" % (
        ty, len(v), mx, math.sqrt(sxx / len(v)), my, math.sqrt(syy / len(v)), sxy / math.sqrt(sxx * syy), sxy / sxx))
    v2 = sorted(v)
    print("   hi,lo sorted by hi:", " ".join("%.1f/%.1f" % (a, b) for a, b, _, _, _ in v2[::max(1, len(v2) // 30)]))
    # sum and difference stats
    s = [a + b for a, b, _, _, _ in v]; d = [a - b for a, b, _, _, _ in v]
    print("   hi+lo: mean %.2f sd %.2f   hi-lo: mean %.2f sd %.2f" % (sum(s) / len(s), (sum((t - sum(s) / len(s)) ** 2 for t in s) / len(s)) ** .5, sum(d) / len(d), (sum((t - sum(d) / len(d)) ** 2 for t in d) / len(d)) ** .5))
