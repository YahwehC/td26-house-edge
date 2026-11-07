"""Out-of-sample on validation targets: fixed-beta coupling corrections (betas from train ticks, ev36)."""
import sys
sys.path.insert(0, '.')
from he.model import load_split, official_score
from he.api import load_params, predict_queries
from he.corr import apply_corr, feature
qv, ev = load_split('kit/public', 'validation')
P = {k: v for k, v in load_params('runs/main_v5/params.json').items() if k != "logit_corr"}
pred = predict_queries(qv, P)
s0 = official_score(pred, ev, qv); print("no corr   %.3f" % s0["points"], s0["ratios"])
s5 = official_score(apply_corr(qv, pred, load_params('runs/main_v5/params.json')["logit_corr"]), ev, qv)
print("main_v5 heat (in-sample fit) %.3f" % s5["points"], s5["ratios"])
for names, beta in ((["oth_wdev_3"], [0.09]), (["oth_wdev_3"], [0.07]), (["oth_wdev_3"], [0.11]), (["oth_win_3"], [0.09]),
                    (["oth_wdev_3", "oth_wdev_6"], [0.09, 0.02]), (["oth_wdev_3", "ep_net_8"], [0.09, 0.05])):
    mu = [0.0 if n.startswith("oth_wdev") else sum(feature(qv, n).values()) / len(qv) for n in names]
    if names == ["oth_win_3"]: mu = [2.88]
    s = official_score(apply_corr(qv, pred, {"names": names, "beta": beta, "mu": mu}), ev, qv)
    print("%-24s beta %-12s %.3f (%+.3f)" % ("+".join(names), beta, s["points"], s["points"] - s0["points"]), s["ratios"])
