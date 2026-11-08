"""Coupling scale check: validation score with logit_corr beta scaled (params from a run dir, data from a kit)."""
import sys, copy
sys.path.insert(0, '.')
from he.model import load_split, official_score
from he.api import predict_queries, load_params
P = load_params(sys.argv[1]); data = sys.argv[2]; mp = float(sys.argv[3]) if len(sys.argv) > 3 else 60
qs, ex = load_split(data, 'validation')
from he.api import _predict_blend
from he.corr import apply_corr
base = _predict_blend(qs, P)
for sc in (0.0, 0.7, 1.0, 1.2, 1.4, 1.7):
    c = copy.deepcopy(P["logit_corr"]); c["beta"] = [b * sc for b in c["beta"]]
    s = official_score(apply_corr(qs, base, c), ex, qs, mp); print("scale %.1f  %.3f %s" % (sc, s["points"], s["ratios"]))
