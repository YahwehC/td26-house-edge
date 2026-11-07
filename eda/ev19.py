"""Codex idea: compare hypotheses on a common usable-tick support (win pairs and size lags)."""
import sys, json
sys.path.insert(0, '.')
from hx.api import fit_params
from hx.model import load_split, official_score
from hx.joint import predict_joint
p = fit_params('kit/public', fit_splits=('train',))
qv, ev = load_split('kit/public', 'validation')
for g in [{"shared_sd": 0.3, "offset_kappa_mult": 1.0}, {"shared_sd": 0.3, "offset_kappa_mult": 1.5}, {"shared_sd": 0.45, "offset_kappa_mult": 2.0}, {"shared_sd": 0.2, "offset_kappa_mult": 1.2}]:
    cfg = dict(p['cfg'], **g)
    s = official_score(predict_joint(qv, p['size_vectors'], cfg), ev, qv)
    print(json.dumps(g).ljust(30), '%.3f' % s['points'], s['ratios'])
