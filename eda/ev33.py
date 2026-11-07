"""Cross-machine coupling: corr of residual_i(t) with residual_j(t-k) and win_j(t-k), split by own pair and mate pair."""
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
K = range(0, 7)
S = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0])  # sx sy sxx syy sxy n
for split in ("train", "validation"):
    qs, ex = load_split('kit/public', split)
    cond = {r["query_id"]: r["condition"] for r in ex["conditions"]}
    ep = defaultdict(list)
    for q in qs: ep[q["episode_id"]].append(q)
    for e, ms in ep.items():
        if cond[ms[0]["query_id"]] != "stable": continue
        T = min(m["cutoff"] for m in ms)
        info = []
        for m in ms:
            sep, (A, B), r = plug(m["seq"]); s = m["seq"]
            res = {}
            for t in range(T):
                if t in s and (t - A) in s and (t - B) in s:
                    z = int((s[t - A] > 0) == (s[t - B] > 0)); res[t] = int(s[t] > 0) - r[z]
            info.append((sep >= 0.3, (A, B), res, s))
        for i, (ok_i, pi, ri, si) in enumerate(info):
            if not ok_i: continue
            for j, (ok_j, pj, rj, sj) in enumerate(info):
                if j == i or not ok_j: continue
                for t in range(40, T):
                    if t not in ri: continue
                    for k in K:
                        if (t - k) not in rj: continue
                        for key, x in ((("res", k), rj[t - k]), (("win", k), int(sj[t - k] > 0)),
                                       (("res", k, "own" + str(pi)), rj[t - k]), (("res", k, "mate" + str(pj)), rj[t - k])):
                            a = S[key]; y = ri[t]
                            a[0] += x; a[1] += y; a[2] += x * x; a[3] += y * y; a[4] += x * y; a[5] += 1
def corr(a):
    sx, sy, sxx, syy, sxy, n = a
    cov = sxy / n - sx * sy / n / n; vx = sxx / n - (sx / n) ** 2; vy = syy / n - (sy / n) ** 2
    return cov / math.sqrt(vx * vy), cov / vx, n
for k in K:
    c, b, n = corr(S[("res", k)]); c2, b2, _ = corr(S[("win", k)])
    print("k=%d  corr(res_i(t), res_j(t-k)) %+.4f (z %+.1f) slope %+.4f | corr with win_j %+.4f  n %d" % (k, c, c * math.sqrt(n), b, c2, n))
print("lag-3 by own pair / mate pair:")
for key in sorted(k for k in S if len(k) == 3 and k[1] == 3):
    c, b, n = corr(S[key]); print("  ", key[2], "corr %+.4f z %+.1f slope %+.4f n %d" % (c, c * math.sqrt(n), b, n))
