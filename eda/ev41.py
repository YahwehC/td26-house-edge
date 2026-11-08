"""Short-history anatomy on validation: per-episode history length, win-error, size-error (main_v6)."""
import sys
from collections import defaultdict
sys.path.insert(0, '.')
from he.model import load_split
from he.api import predict_queries, load_params
P = load_params('runs/main_v6/params.json')
qs, ex = load_split('kit/public', 'validation')
cond = {r["query_id"]: r["condition"] for r in ex["conditions"]}
ref = {r["query_id"]: [float(r[k]) for k in ("p_0", "p_1", "p_2", "p_5", "p_20")] for r in ex["reference"]}
pred = predict_queries(qs, P)
T = {t["query_id"]: [float(t[k]) for k in ("p_0", "p_1", "p_2", "p_5", "p_20")] for t in ex["targets"]}
E = defaultdict(lambda: [0, 0, 0, 0, 0, 0])
for q in qs:
    k = q["query_id"]
    if cond[k] != "short_history": continue
    p, t = pred[k], T[k]
    e = E[q["episode_id"]]
    e[0] += 1; e[1] += sum((a - b) ** 2 for a, b in zip(p, t)); e[2] += (p[0] - t[0]) ** 2
    e[3] += sum((a - b) ** 2 for a, b in zip(ref[k], t)); e[4] = len(q["seq"]); e[5] = q["cutoff"]
tot = sum(v[1] / v[0] for v in E.values())
print("eps", len(E), "mean err %.5f  ref %.5f" % (tot / len(E), sum(v[3] / v[0] for v in E.values()) / len(E)))
for e, v in sorted(E.items(), key=lambda kv: -kv[1][1] / kv[1][0]):
    print(e, "n_obs %3d cutoff %3d  err %.4f  p0err %.4f  ref %.4f" % (v[4], v[5], v[1] / v[0], v[2] / v[0], v[3] / v[0]))
