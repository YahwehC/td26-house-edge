"""Heat follow-ups: (a) residual-based heat feature, (b) does heat shift the size distribution?"""
import sys, math
from collections import defaultdict
sys.path.insert(0, '.')
from he.model import load_split, official_score
from he.api import load_params, predict_queries
from he.corr import episode_feature, fit_corr, apply_corr
qv, ev = load_split('kit/public', 'validation')
P = load_params('runs/main_v5/params.json'); base = {k: v for k, v in P.items() if k != "logit_corr"}
pred = predict_queries(qv, base)
tg = {r["query_id"]: [float(r[k]) for k in ("p_0", "p_1", "p_2", "p_5", "p_20")] for r in ev["targets"]}
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
ep = defaultdict(list)
for q in qv: ep[q["episode_id"]].append(q)
info = {q["query_id"]: plug(q["seq"]) for q in qv}
def resid_heat(q, K, own=True):
    c = q["cutoff"]; tot = 0.0; n = 0
    for m in ep[q["episode_id"]]:
        if not own and m is q: continue
        _, (A, B), r = info[m["query_id"]]; s = m["seq"]
        for k in range(1, K + 1):
            t = c - k
            if t in s and (t - A) in s and (t - B) in s:
                z = int((s[t - A] > 0) == (s[t - B] > 0)); tot += int(s[t] > 0) - r[z]; n += 1
    return tot / max(n, 1)
eps = sorted(ep); fold = {e: i % 2 for i, e in enumerate(eps)}
b0 = official_score(pred, ev, qv)["points"]
import he.corr as hc
for name, fn in [("resid_8", lambda q: resid_heat(q, 8)), ("resid_16", lambda q: resid_heat(q, 16)),
                 ("resid_8_others", lambda q: resid_heat(q, 8, False)), ("resid_32", lambda q: resid_heat(q, 32))]:
    feat = {q["query_id"]: fn(q) for q in qv}
    orig = hc.episode_feature
    hc.episode_feature = lambda qs, K=8, feat=feat: {q["query_id"]: feat[q["query_id"]] for q in qs}
    cv = {}
    for fo in (0, 1):
        tr = [q for q in qv if fold[q["episode_id"]] != fo]; te = [q for q in qv if fold[q["episode_id"]] == fo]
        c = hc.fit_corr(tr, pred, ev["targets"]); cv.update(hc.apply_corr(te, pred, c))
    hc.episode_feature = orig
    print("%-16s CV %.3f (%+.3f) beta %.2f" % (name, official_score(cv, ev, qv)["points"], official_score(cv, ev, qv)["points"] - b0, c["beta"]))
# (b) size shift: true P(size>=5 | win) residual vs heat
f8 = episode_feature(qv, 8)
xs, ys = [], []
for q in qv:
    t = tg[q["query_id"]]; w = 1 - t[0]; p = pred[q["query_id"]]; wp = 1 - p[0]
    if w < 0.05 or wp < 0.05: continue
    xs.append(f8[q["query_id"]]); ys.append((t[3] + t[4]) / w - (p[3] + p[4]) / wp)
mx, my = sum(xs) / len(xs), sum(ys) / len(ys); sxx = sum((a - mx) ** 2 for a in xs)
b = sum((a - mx) * (c - my) for a, c in zip(xs, ys)) / sxx
res = [c - my - b * (a - mx) for a, c in zip(xs, ys)]; se = math.sqrt(sum(e * e for e in res) / (len(xs) - 2) / sxx)
print("size: P(>=5|win) residual vs ep_net_8: slope %.4f z %.1f (n %d, resid sd %.3f)" % (b, b / se, len(xs), (sum(e * e for e in res) / len(res)) ** .5))
