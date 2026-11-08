"""Per-condition gap of main_v6 on validation: actual vs oracle win prob (size kept) vs oracle size (win kept)."""
import sys, json
sys.path.insert(0, '.')
from he.model import load_split, official_score
from he.api import predict_queries, load_params
P = load_params('runs/main_v6/params.json')
qs, ex = load_split('kit/public', 'validation')
pred = predict_queries(qs, P)
T = {t["query_id"]: [float(t[k]) for k in ("p_0", "p_1", "p_2", "p_5", "p_20")] for t in ex["targets"]}
ow, osz = {}, {}
for k, p in pred.items():
    t = T[k]; w, tw = 1 - p[0], 1 - t[0]
    ow[k] = [t[0]] + [tw * x / w for x in p[1:]]
    osz[k] = [p[0]] + [w * x / tw for x in t[1:]]
for name, pr in (("main_v6", pred), ("oracle_win", ow), ("oracle_size", osz)):
    s = official_score(pr, ex, qs)
    print(name, round(s["points"], 3), s["ratios"])
