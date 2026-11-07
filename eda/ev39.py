"""Coupling heterogeneity: logistic beta of oth_win_3 split by own cell (hi/lo) and machine type (train+val ticks)."""
import sys, math
from collections import defaultdict
sys.path.insert(0, '.')
from he.model import load_split
import he.coupling as hc
groups = defaultdict(lambda: ([], [], []))
for split in ("train", "validation"):
    qs, ex = load_split('kit/public', split)
    cm = {r["query_id"]: r["condition"] for r in ex["conditions"]}
    by = defaultdict(list)
    for q in qs: by[q["episode_id"]].append(q)
    for e, ms in by.items():
        for m in ms:
            X, Y, O = hc.tick_rows(sorted(ms, key=lambda q: q["machine_id"]), cm)
            break
        # tag rows by machine: recompute per machine to know type & cell -> do it by running tick_rows per machine group
        ms = sorted(ms, key=lambda q: q["machine_id"])
        X, Y, O = hc.tick_rows(ms, cm)
        # tick_rows iterates machines in order; we cannot recover machine -> approximate by offset sign (hi cell: offset>0)
        for x, y, o in zip(X, Y, O):
            g = groups["hi" if o > 0 else "lo"]; g[0].append(x); g[1].append(y); g[2].append(o)
for name, (X, Y, O) in sorted(groups.items()):
    b, se, mu = hc._logreg(X, Y, O, [2])
    p = sum(Y) / len(Y)
    print("%s cell: n %d  win rate %.3f  beta(oth_win_3) %+.4f (z %+.1f)  -> prob effect per winner ~%+.4f" % (name, len(X), p, b[1], b[1] / se[1], b[1] * p * (1 - p)))
