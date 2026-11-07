"""Heat mechanism: multiple regression of tick residual on counts of each payout type among OTHER machines, per lag."""
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
LAGS = list(range(1, 25))
PAY = (1, 2, 5, 20)
# features: for each lag k and payout v: number of other machines with payout v at t-k. Regress per (k, v) marginally,
# and fit a joint model: y = sum_v a_v * sum_{k in 3..16} cnt(v,k) to get payout weights.
stats = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])
XtX = defaultdict(float); Xty = defaultdict(float); N = 0
for split in ("train", "validation"):
    qs, ex = load_split('kit/public', split)
    cond = {r["query_id"]: r["condition"] for r in ex["conditions"]}
    ep = defaultdict(list)
    for q in qs: ep[q["episode_id"]].append(q)
    for e, ms in ep.items():
        if cond[ms[0]["query_id"]] != "stable":
            continue
        T = min(m["cutoff"] for m in ms)
        cnt = [[{v: 0 for v in PAY} for _ in range(T)] for _ in ms]  # cnt[i][t][v] = # machines != i with payout v at t
        tot = [{v: sum(int(m["seq"].get(t, 0) == v) for m in ms) for v in PAY} for t in range(T)]
        for i, m in enumerate(ms):
            sep, (A, B), r = plug(m["seq"]); s = m["seq"]
            if sep < 0.3: continue
            for t in range(40, T):
                if (t - A) not in s or (t - B) not in s: continue
                z = int((s[t - A] > 0) == (s[t - B] > 0)); y = int(s[t] > 0) - r[z]
                xs = {}
                for k in LAGS:
                    own = s.get(t - k, 0)
                    for v in PAY:
                        x = tot[t - k][v] - int(own == v)
                        a = stats[(k, v)]; a[0] += x; a[1] += y; a[2] += x * x; a[3] += x * y; a[4] += 1
                        if 3 <= k <= 16:
                            xs[v] = xs.get(v, 0) + x
                feats = [1.0] + [xs[v] for v in PAY]
                for a_ in range(5):
                    Xty[a_] += feats[a_] * y
                    for b_ in range(5): XtX[(a_, b_)] += feats[a_] * feats[b_]
                N += 1
print("per-lag marginal slopes (win-prob change per extra other machine with that payout at t-k), z in ()")
for k in LAGS:
    row = []
    for v in PAY:
        sx, sy, sxx, sxy, n = stats[(k, v)]
        vx = sxx / n - (sx / n) ** 2; b = (sxy / n - sx * sy / n / n) / vx; se = math.sqrt(0.2 / n / vx)
        row.append("%2dx:%+.4f(%+.1f)" % (v, b, b / se))
    print("k=%2d  " % k + "  ".join(row))
# solve joint
n = 5; A = [[XtX[(i, j)] for j in range(n)] + [Xty[i]] for i in range(n)]
for i in range(n):
    piv = A[i][i]
    for j in range(i, n + 1): A[i][j] /= piv
    for r2 in range(n):
        if r2 != i:
            f = A[r2][i]
            for j in range(i, n + 1): A[r2][j] -= f * A[i][j]
print("joint weights (lags 3..16 summed), per extra payout event:", {v: round(A[i + 1][n], 5) for i, v in enumerate(PAY)}, "N", N)
