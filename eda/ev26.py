"""Short-history validation: size-regime (r1) errors - where do they come from?"""
import sys, math
from collections import defaultdict, Counter
sys.path.insert(0, '.')
from he.model import load_split, SIZE_IDX
from he.api import load_params
from he.joint import predict_joint, SizeEvidence, lag_prior_marginal, lag_prior_for_pair
qv, ev = load_split('kit/public', 'validation')
from he.model import DEFAULT_CFG
P = load_params("runs/main_v3/params.json"); cfg = dict(DEFAULT_CFG, **P["cfg"]); vecs = P["size_vectors"]
pred = predict_joint(qv, vecs, cfg)
tg = {r["query_id"]: [float(r[k]) for k in ("p_0", "p_1", "p_2", "p_5", "p_20")] for r in ev["targets"]}
cond = {r["query_id"]: r["condition"] for r in ev["conditions"]}
lagrule = cfg.get("size_lag_given_pair"); print("lag rule", lagrule, "size_const", cfg["size_const"], "size_eps", cfg["size_eps"])
print("vecs", {k: [[round(x, 3) for x in v] for v in vv] for k, vv in vecs.items()})
rows = []
for q in qv:
    if cond[q["query_id"]] != "short_history":
        continue
    t = tg[q["query_id"]]; w = 1 - t[0]
    if w < 0.02:
        continue
    s_true = [x / w for x in t[1:]]
    V = vecs[q["machine_type"]]
    d = [sum((a - b) ** 2 for a, b in zip(s_true, V[r])) for r in (0, 1)]
    true_r = 0 if d[0] < d[1] else 1
    p = pred[q["query_id"]]; wp = 1 - p[0]; s_ours = [x / wp for x in p[1:]]
    # our r1 from s_ours by projection
    denom = sum((V[1][k] - V[0][k]) ** 2 for k in range(4))
    r1 = sum((s_ours[k] - V[0][k]) * (V[1][k] - V[0][k]) for k in range(4)) / denom
    seq, c = q["seq"], q["cutoff"]
    nwin = sum(1 for v in seq.values() if v > 0)
    rows.append(dict(q=q, true_r=true_r, r1=r1, err=(r1 - true_r) ** 2, dmin=min(d), nwin=nwin, n=len(seq), w=w,
                     se=sum((a - b) ** 2 for a, b in zip(p, t))))
print("short machines", len(rows), "mean min dist of true size dist to nearest vector %.4f (max %.3f)" % (sum(r["dmin"] for r in rows) / len(rows), max(r["dmin"] for r in rows)))
print("mean (r1-true)^2 = %.3f ; share with |r1-true|>0.5: %.2f ; mean total SE %.4f" % (
    sum(r["err"] for r in rows) / len(rows), sum(r["err"] > 0.25 for r in rows) / len(rows), sum(r["se"] for r in rows) / len(rows)))
for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 100)]:
    v = [r for r in rows if lo <= r["nwin"] < hi]
    if v: print("nwin [%d,%d): k=%d mean (r1-true)^2=%.3f mean SE=%.4f" % (lo, hi, len(v), sum(r["err"] for r in v) / len(v), sum(r["se"] for r in v) / len(v)))
# worst cases: inspect evidence
rows.sort(key=lambda r: -r["err"])
for r in rows[:6]:
    q = r["q"]; seq, c = q["seq"], q["cutoff"]
    se = SizeEvidence(q, vecs, cfg)
    print("--", q["query_id"], q["machine_type"], "sig", q["signal"], "cutoff", c, "n", r["n"], "nwin", r["nwin"], "true_r", r["true_r"], "r1 %.2f" % r["r1"], "w_true %.2f" % r["w"])
    print("   const ll", [round(x, 1) for x in se.const], " lag(L): [(ll s=0, pr1), (ll s=1, pr1)]:", {L: [(round(a, 1), round(b, 2)) for a, b in se.lag[L]] for L in (1, 2, 4)})
    print("   last 12 payouts", [seq.get(c - k) for k in range(12, 0, -1)])
