"""fit_coupling on train ticks -> apply at cutoff on validation (out-of-sample); then train+val fit for reference."""
import sys, time
sys.path.insert(0, '.')
from he.model import load_split, official_score
from he.api import load_params, predict_queries
from he.corr import apply_corr
from he.coupling import fit_coupling
t0 = time.time()
tr, etr = load_split('kit/public', 'train'); qv, ev = load_split('kit/public', 'validation')
cm = lambda ex: {r["query_id"]: r["condition"] for r in ex.get("conditions", [])}
c = fit_coupling([tr], [cm(etr)]); print("fit %.1fs" % (time.time() - t0))
P = {k: v for k, v in load_params('runs/main_v5/params.json').items() if k != "logit_corr"}
pred = predict_queries(qv, P)
s0 = official_score(pred, ev, qv); s1 = official_score(apply_corr(qv, pred, c), ev, qv)
print("val no corr %.3f -> train-fitted coupling %.3f (%+.3f)" % (s0["points"], s1["points"], s1["points"] - s0["points"]), s1["ratios"])
c2 = fit_coupling([tr, qv], [cm(etr), cm(ev)]); s2 = official_score(apply_corr(qv, pred, c2), ev, qv)
print("train+val fitted (in-sample ticks, not targets) %.3f" % s2["points"])
