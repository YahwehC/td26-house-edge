"""House-edge feedback test: does logit(true win) - logit(our win) depend on payout history (own/episode net)?"""
import sys, math
from collections import defaultdict
sys.path.insert(0, '.')
from he.model import load_split
from he.api import load_params, predict_queries
qv, ev = load_split('kit/public', 'validation')
P = dict(load_params('runs/main_v4/params.json')); P.pop('blend', None); P.pop('fallback', None)
pred = predict_queries(qv, P)
tg = {r["query_id"]: 1 - float(r["p_0"]) for r in ev["targets"]}
cond = {r["query_id"]: r["condition"] for r in ev["conditions"]}
lg = lambda p: math.log(min(max(p, 1e-4), 1 - 1e-4) / (1 - min(max(p, 1e-4), 1 - 1e-4)))
ep = defaultdict(list)
for q in qv: ep[q["episode_id"]].append(q)
rows = []
for q in qv:
    seq, c = q["seq"], q["cutoff"]
    full = len(seq) >= c - 1  # consecutive history
    f = {}
    f["own_net_all"] = sum(v - 1 for v in seq.values()) / 100
    for K in (8, 16, 32, 64):
        f["own_net_%d" % K] = sum(seq.get(c - k, 1) - 1 for k in range(1, K + 1)) / 10
        f["own_wins_%d" % K] = sum(int(seq.get(c - k, 0) > 0) for k in range(1, K + 1)) / K
    mates = ep[q["episode_id"]]
    f["ep_net_all"] = sum(sum(v - 1 for v in m["seq"].values()) for m in mates) / 800
    f["ep_net_16"] = sum(sum(m["seq"].get(c - k, 1) - 1 for k in range(1, 17)) for m in mates) / 80
    f["ep_pay_1"] = sum(m["seq"].get(c - 1, 0) for m in mates) / 8
    f["others_net_all"] = (sum(sum(v - 1 for v in m["seq"].values()) for m in mates) - sum(v - 1 for v in seq.values())) / 700
    f["big_lag1"] = int(seq.get(c - 1, 0) >= 2); f["jack_lag1"] = int(seq.get(c - 1, 0) == 20)
    f["cutoff"] = c / 100; f["signal"] = q["signal"]
    ours = 1 - pred[q["query_id"]][0]; tp = tg[q["query_id"]]
    if not (0.02 < tp < 0.98 and 0.02 < ours < 0.98):
        continue
    rows.append((cond[q["query_id"]], q["machine_type"], lg(tp) - lg(ours), f))
print("n", len(rows))
def reg(sub, name):
    x = [r[3][name] for r in sub]; y = [r[2] for r in sub]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxx = sum((a - mx) ** 2 for a in x)
    if sxx < 1e-12: return None
    b = sum((a - mx) * (bb - my) for a, bb in zip(x, y)) / sxx
    res = [bb - my - b * (a - mx) for a, bb in zip(x, y)]
    se = math.sqrt(sum(e * e for e in res) / (len(x) - 2) / sxx)
    return b, b / se, math.sqrt(sxx / len(x))
for subname, sub in [("all", rows), ("stable", [r for r in rows if r[0] == "stable"]), ("non-short", [r for r in rows if r[0] != "short_history"])]:
    print("==", subname, len(sub), "resid mean %.3f sd %.3f" % (sum(r[2] for r in sub) / len(sub), (sum(r[2] ** 2 for r in sub) / len(sub)) ** .5))
    for name in rows[0][3]:
        o = reg(sub, name)
        if o: print("   %-15s slope %+.3f  z %+5.1f  (x sd %.2f)" % (name, o[0], o[1], o[2]))
