"""Prob-space residual screen (validation, main_v6): (true win - pred win) vs features, episode-clustered SE."""
import sys, math
from collections import defaultdict
sys.path.insert(0, '.')
exec(open('eda/ev48.py').read().split("res = {")[0].split("for cd in")[0])
# rebuild feats (copy of ev48 feature block)
src = open('eda/ev48.py').read()
exec("feats = defaultdict(dict)\n" + src.split("feats = defaultdict(dict)\n")[1].split("res = {")[0])
for q in qs:
    k = q["query_id"]; feats["pw"][k] = 1 - pred[k][0]
    feats["cond_" + cond[k]][k] = 1.0
res = {k: T[k] - (1 - pred[k][0]) for k in pred}
ep = {q["query_id"]: q["episode_id"] for q in qs}
out = []
for nm, f in feats.items():
    ks = list(res); xs = [f.get(k, 0.0) for k in ks]; ys = [res[k] for k in ks]
    n = len(ks); mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0: continue
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    G = defaultdict(float)
    for k, x, y in zip(ks, xs, ys): G[ep[k]] += (x - mx) * (y - my - b * (x - mx))
    se = math.sqrt(sum(g * g for g in G.values())) / sxx
    out.append((abs(b / se), nm, b, b / se))
for a, nm, b, z in sorted(out, reverse=True)[:12]: print("%-14s b %+.5f z %+.2f" % (nm, b, z))
