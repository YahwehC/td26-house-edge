"""Sweep variants on pseudo-short episodes (eda/_pshort.pkl): paired Brier diff vs baseline (main_v6 params)."""
import sys, pickle, copy, math, time, json
from collections import defaultdict
sys.path.insert(0, '.')
from he.api import predict_queries, load_params
qs, ys = pickle.load(open('eda/_pshort.pkl', 'rb'))
P0 = load_params('runs/main_v6/params.json')

def brier(P):
    pred = predict_queries(qs, P)
    E = defaultdict(list)
    for q in qs:
        k = q["query_id"]; p = pred[k]; y = ys[k]
        E[q["episode_id"]].append(math.fsum((x - (1.0 if i == y else 0.0)) ** 2 for i, x in enumerate(p)))
    return {e: sum(v) / len(v) for e, v in E.items()}

def variant(name, f):
    P = copy.deepcopy(P0); f(P); t = time.time(); b = brier(P)
    d = [b[e] - base[e] for e in base]; m = sum(d) / len(d)
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d))
    print("%-28s brier %.5f  diff %+.6f (z %+.1f)  %.0fs" % (name, sum(b.values()) / len(b), m, m / se if se else 0, time.time() - t), flush=True)

t = time.time(); base = brier(P0)
print("base brier %.5f  %.0fs" % (sum(base.values()) / len(base), time.time() - t), flush=True)
V = sys.argv[1:] or ["k"]
def setcfg(**kw): return lambda P: P["cfg"].update(kw)
if "k" in V:
    for ks in (0.5, 0.7, 1.4, 2.0): variant("kappa_scale %.1f" % ks, setcfg(kappa_scale=ks))
    variant("w_main 1.0", lambda P: P["blend"].update(w_main=1.0))
    variant("w_main 0.8", lambda P: P["blend"].update(w_main=0.8))
    variant("no corr", lambda P: P.pop("logit_corr"))
    variant("corr x1.5", lambda P: P["logit_corr"].update(beta=[b * 1.5 for b in P["logit_corr"]["beta"]]))
