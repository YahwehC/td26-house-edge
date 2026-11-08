"""Prob-space residual screen of main_v10 out-of-fold validation predictions (episode-clustered SE), per condition too."""
import sys, math, csv
from collections import defaultdict
sys.path.insert(0, '.')
from he.model import load_split
qs, ex = load_split('kit/public', 'validation')
cond = {r["query_id"]: r["condition"] for r in ex["conditions"]}
pred = {r['query_id']: [float(r[c]) for c in ('p_0', 'p_1', 'p_2', 'p_5', 'p_20')] for r in csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1 else 'eda/ev68_predictions.csv'))}
T = {t["query_id"]: [float(t[k]) for k in ("p_0", "p_1", "p_2", "p_5", "p_20")] for t in ex["targets"]}
by = defaultdict(list)
for q in qs: by[q["episode_id"]].append(q)
F = defaultdict(dict)
for q in qs:
    k, c, s = q["query_id"], q["cutoff"], q["seq"]
    mates = [m for m in by[q["episode_id"]] if m is not q]
    for j in range(1, 9):
        F["own_win_%d" % j][k] = float(s.get(c - j, 0) > 0); F["own_big_%d" % j][k] = float(s.get(c - j, 0) >= 2)
        F["oth_win_%d" % j][k] = float(sum(m["seq"].get(c - j, 0) > 0 for m in mates))
        F["oth_big_%d" % j][k] = float(sum(m["seq"].get(c - j, 0) >= 2 for m in mates))
    F["oth_net_8"][k] = sum(m["seq"].get(c - j, 1) - 1 for m in mates for j in range(1, 9)) / 8
    F["own_net_8"][k] = sum(s.get(c - j, 1) - 1 for j in range(1, 9)) / 8
    F["ep_wins_20"][k] = sum(m["seq"].get(c - j, 0) > 0 for m in by[q["episode_id"]] for j in range(1, 21)) / 20
    F["cutoff"][k] = float(c); F["signal"][k] = float(q["signal"]); F["pw"][k] = 1 - pred[k][0]
    F["n_obs"][k] = float(len(s))
    for ty in ("balanced", "jackpot", "steady"): F["type_" + ty][k] = float(q["machine_type"] == ty)
ep = {q["query_id"]: q["episode_id"] for q in qs}
def screen(keys, comp, label):
    res = {k: T[k][comp] - pred[k][comp] for k in keys} if comp is not None else {k: (1 - T[k][0]) - (1 - pred[k][0]) for k in keys}
    out = []
    for nm, f in F.items():
        ks = list(res); xs = [f[k] for k in ks]; ys = [res[k] for k in ks]; n = len(ks)
        mx, my = sum(xs) / n, sum(ys) / n; sxx = sum((x - mx) ** 2 for x in xs)
        if sxx == 0: continue
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx; G = defaultdict(float)
        for k, x, y in zip(ks, xs, ys): G[ep[k]] += (x - mx) * (y - my - b * (x - mx))
        se = math.sqrt(sum(g * g for g in G.values())) / sxx; out.append((abs(b / se), nm, b, b / se))
    print(label, " ".join("%s %+.4f(z%+.1f)" % (nm, b, z) for a, nm, b, z in sorted(out, reverse=True)[:6]))
allk = list(pred)
screen(allk, None, "WIN all:")
for cd in ("stable", "drift", "shared_change", "short_history"):
    screen([k for k in allk if cond[k] == cd], None, "WIN %s:" % cd)
for comp, nm in ((1, "p1"), (2, "p2"), (3, "p5"), (4, "p20")):
    screen(allk, comp, "SIZE %s all:" % nm)
