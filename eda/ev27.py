"""Is the size sign s predictable? Calibration of r1 on short-history machines."""
import sys, math
from collections import defaultdict, Counter
sys.path.insert(0, '.')
from he.model import load_split, DEFAULT_CFG
from he.api import load_params
from he.joint import predict_joint, SizeEvidence
P = load_params('runs/main_v3/params.json'); cfg = dict(DEFAULT_CFG, **P["cfg"]); vecs = P["size_vectors"]
PAIRS = [(1, 3), (2, 5), (4, 7)]
def best_pair(seq):
    best = None
    for A, B in PAIRS:
        n = [[0, 0], [0, 0]]
        for t in seq:
            if (t - A) in seq and (t - B) in seq:
                n[int((seq[t - A] > 0) == (seq[t - B] > 0))][int(seq[t] > 0)] += 1
        r = [(n[z][1] + .5) / (sum(n[z]) + 1) for z in (0, 1)]
        if best is None or abs(r[1] - r[0]) > best[0]:
            best = (abs(r[1] - r[0]), (A, B), int(r[1] > r[0]))
    return best
tab = defaultdict(Counter)
for split in ("train", "validation"):
    qs, ex = load_split('kit/public', split)
    cond = {r["query_id"]: r["condition"] for r in ex["conditions"]}
    for q in qs:
        if cond[q["query_id"]] != "stable":
            continue
        sep, (A, B), ori = best_pair(q["seq"])
        if sep < 0.3: continue
        se = SizeEvidence(q, vecs, cfg)
        lls = {(L, s): se.lag[L][s][0] for L in (1, 2, 4) for s in (0, 1)}
        (L, s), top = max(lls.items(), key=lambda x: x[1])
        second = max(v for k, v in lls.items() if k != (L, s))
        if top - second < 3: continue
        tab["type"][(q["machine_type"], s)] += 1
        tab["signal"][(q["signal"], s)] += 1
        tab["orientation(hi cell is XNOR=1)"][(ori, s)] += 1
        tab["pair"][((A, B), s)] += 1
        tab["L==A"][(L == A, s)] += 1
        tab["slot"][(q["machine_id"][-4:], s)] += 1
for k, c in tab.items():
    keys = sorted({a for a, _ in c})
    print(k, {a: "s1=%.2f/n%d" % (c[(a, 1)] / (c[(a, 0)] + c[(a, 1)]), c[(a, 0)] + c[(a, 1)]) for a in keys})
# calibration on short validation
qv, ev = load_split('kit/public', 'validation')
cond = {r["query_id"]: r["condition"] for r in ev["conditions"]}
tg = {r["query_id"]: [float(r[k]) for k in ("p_0", "p_1", "p_2", "p_5", "p_20")] for r in ev["targets"]}
pred = predict_joint(qv, vecs, cfg)
cal = defaultdict(lambda: [0, 0.0, 0.0])
for q in qv:
    if cond[q["query_id"]] != "short_history": continue
    t = tg[q["query_id"]]; w = 1 - t[0]
    if w < 0.05: continue
    V = vecs[q["machine_type"]]; s_true = [x / w for x in t[1:]]
    true_r = 0 if sum((a - b) ** 2 for a, b in zip(s_true, V[0])) < sum((a - b) ** 2 for a, b in zip(s_true, V[1])) else 1
    p = pred[q["query_id"]]; wp = 1 - p[0]; s_ours = [x / wp for x in p[1:]]
    denom = sum((V[1][k] - V[0][k]) ** 2 for k in range(4))
    r1 = min(max(sum((s_ours[k] - V[0][k]) * (V[1][k] - V[0][k]) for k in range(4)) / denom, 0), 1)
    conf = abs(r1 - 0.5)
    b = cal[min(int(conf * 10), 4)]; b[0] += 1; b[1] += r1 * (1 - r1); b[2] += (r1 - true_r) ** 2
print("short calibration by |r1-0.5| bin: n, mean r1(1-r1) [model's own expected error], mean (r1-true)^2 [actual]")
for k in sorted(cal):
    n, e, a = cal[k]; print("  bin %d: n=%d expected %.3f actual %.3f" % (k, n, e / n, a / n))
