"""2-fold CV (by episode) of multi-feature logit corrections on top of main_v4-style predictions."""
import sys
sys.path.insert(0, '.')
from he.model import load_split, official_score
from he.api import load_params, predict_queries
from he.corr import fit_corr, apply_corr
qv, ev = load_split('kit/public', 'validation')
P = {k: v for k, v in load_params('runs/main_v5/params.json').items() if k != "logit_corr"}
pred = predict_queries(qv, P)
b0 = official_score(pred, ev, qv)["points"]
eps = sorted({q["episode_id"] for q in qv})
for seed in (0, 1):
    fold = {e: (i + seed) % 2 if seed == 0 else (hash(e) % 2) for i, e in enumerate(eps)}
    fold = {e: (i // (1 + seed)) % 2 for i, e in enumerate(eps)}
    print("== split", seed, "base %.3f" % b0)
    for names in (["ep_net_8"], ["oth_win_3"], ["ep_net_8", "oth_win_3"], ["ep_net_16", "oth_win_3"],
                  ["ep_net_8", "oth_win_3", "oth_win_6"], ["ep_net_8", "oth_win_3", "oth_win_4"],
                  ["ep_net_8", "oth_win_3", "oth_win_8"]):
        cv, bs = {}, []
        for fo in (0, 1):
            tr = [q for q in qv if fold[q["episode_id"]] != fo]; te = [q for q in qv if fold[q["episode_id"]] == fo]
            c = fit_corr(tr, pred, ev["targets"], names=names); bs.append([round(b, 3) for b in c["beta"]])
            cv.update(apply_corr(te, pred, c))
        s = official_score(cv, ev, qv)
        print("  %-36s CV %.3f (%+.3f) betas %s %s" % ("+".join(names), s["points"], s["points"] - b0, bs, s["ratios"]))
