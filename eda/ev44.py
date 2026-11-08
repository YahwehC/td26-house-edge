"""Within-episode structure on stable train+val machines: is the lag pair (or size lag) shared across machines?"""
import sys
from collections import defaultdict, Counter
sys.path.insert(0, '.')
from he.model import load_split
import he.coupling as hc
same = tot = 0; cnt = Counter(); ep_pairs = []
for split in ("train", "validation"):
    qs, ex = load_split('kit/public', split)
    cm = {r["query_id"]: r["condition"] for r in ex["conditions"]}
    by = defaultdict(list)
    for q in qs: by[q["episode_id"]].append(q)
    for e, ms in by.items():
        if cm[ms[0]["query_id"]] != "stable" or len(ms[0]["seq"]) < 0.6 * ms[0]["cutoff"]: continue
        bp = []
        for m in ms:
            sep, pr, n = hc._best_pair(m["seq"], m["cutoff"], hc.PAIRS)
            if sep > 0.35: bp.append((pr, m["machine_type"]))
        for i in range(len(bp)):
            cnt[bp[i][0]] += 1
            for j in range(i):
                tot += 1; same += bp[i][0] == bp[j][0]
        ep_pairs.append([x[0] for x in bp])
n = sum(cnt.values())
print("pair freq", dict(cnt), "same-pair rate %.3f vs chance %.3f (%d machine pairs)" % (same / tot, sum((v / n) ** 2 for v in cnt.values()), tot))
print("examples", ep_pairs[:6])
