"""LEGACY benchmark: shared old-tick masks and cutoffs starting at 120.
Use eda/ev107.py for independent-mask and matched-cutoff model selection.
Retained to reproduce historical experiments, not as a faithful official sampler.

Pseudo-short episodes from train histories: keep 40 recent + 24 sampled ticks before a random cutoff c',
target = one-hot of the realised payout at c'. Brier on realised outcomes is an unbiased proxy (up to a constant)
of squared error vs true probs. Saves the pseudo queries to eda/_pshort.pkl for variant sweeps."""
import sys, random, pickle, time
from collections import defaultdict
sys.path.insert(0, '.')
from he.model import load_split
rng = random.Random(42)
print('LEGACY shared-mask benchmark; use eda/ev107.py for faithful independent masks.', flush=True)
qs, ex = load_split('kit/public', 'train')
cm = {r["query_id"]: r["condition"] for r in ex["conditions"]}
by = defaultdict(list)
for q in qs: by[q["episode_id"]].append(q)
out, ys = [], {}
PAY = {0: 0, 1: 1, 2: 2, 5: 3, 20: 4}
for e in sorted(by):
    ms = by[e]
    c0 = cm[ms[0]["query_id"]]
    if c0 == "short_history" or any(len(m["seq"]) < 0.6 * m["cutoff"] for m in ms):
        continue
    T = min(m["cutoff"] for m in ms) - (0 if c0 == "stable" else 110)
    if T < 140: continue
    for r in range(25):
        c = rng.randint(120, T - 1)
        samp = rng.sample(range(0, c - 40), 24)
        keep = set(range(c - 40, c)) | set(samp)
        for m in ms:
            if c not in m["seq"]: break
        else:
            for m in ms:
                q = dict(m)
                q["episode_id"] = "%s-r%d" % (e, r)
                q["query_id"] = "%s-r%d-%s" % (e, r, m["machine_id"])
                q["cutoff"] = c
                q["seq"] = {t: v for t, v in m["seq"].items() if t in keep}
                out.append(q)
                ys[q["query_id"]] = PAY[m["seq"][c]]
print(len(out), "pseudo queries", len({q["episode_id"] for q in out}), "episodes")
pickle.dump((out, ys), open('eda/_pshort.pkl', 'wb'))
