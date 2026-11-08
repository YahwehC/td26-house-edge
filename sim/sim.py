"""Synthetic kit in the main-round format, from the recovered generator, with configurable lag pairs and coupling lag.
Used only to stress-test the round-2 pipeline (r2.py) on changed temporal relationships. Pure stdlib, seeded.

python sim/sim.py --out sim/kit_a/public --pairs 2,3 1,6 5,8 --couple-lag 2 --seed 1
"""
import argparse, csv, json, math, random
from pathlib import Path

TYPES = ["balanced", "jackpot", "steady"]
PAY = [1, 2, 5, 20]
P0 = json.load(open(Path(__file__).resolve().parents[1] / "runs/main_v6/params.json"))
VEC = P0["size_vectors"]
CP = P0["cfg"]["cell_prior"]
sig = lambda x: 1 / (1 + math.exp(-x))


def episode(rng, cond, pairs, couple_lag, beta_c, eid, size_lag="first"):
    c = rng.randint(192, 256)
    ms = []
    for i in range(8):
        ty = rng.choice(TYPES); sg = rng.choice([-1, 0, 1]); A, B = rng.choice(pairs)
        hi = rng.gauss(CP[ty]["hi"][0] + CP[ty]["hi"][1] * sg, 0.6)
        lo = rng.gauss(CP[ty]["lo"][0] + CP[ty]["lo"][1] * sg, 0.6)
        ms.append(dict(ty=ty, sg=sg, A=A, B=B, cell={1: hi, 0: lo} if rng.random() < 0.5 else {1: lo, 0: hi},
                       s=int(rng.random() < 0.45), flip=None, mid="%s-m%03d" % (eid, i)))
    if cond == "drift":
        for m in ms:
            if rng.random() < 0.9: m["flip"] = rng.randint(c - 104, c - 32)
    elif cond == "shared_change":
        tf = rng.randint(c - 104, c - 32)
        for m in ms:
            if rng.random() < 0.9: m["flip"] = tf
    seq = [[0] * (c + 1) for _ in ms]
    truth = []
    for t in range(c + 1):
        for i, m in enumerate(ms):
            if t < 8:
                pw = 0.4
            else:
                z = int((seq[i][t - m["A"]] > 0) == (seq[i][t - m["B"]] > 0))
                if m["flip"] is not None and t >= m["flip"]: z = 1 - z
                oth = sum(seq[j][t - couple_lag] > 0 for j in range(8) if j != i)
                pw = sig(m["cell"][z] + rng.gauss(0, 0.5) + beta_c * (oth - 2.9))
            L = m["A"] if size_lag == "first" else m["B"]
            r = (int(t >= L and seq[i][t - L] >= 2) ^ m["s"])
            v = VEC[m["ty"]][r]
            if t == c:
                truth.append([1 - pw] + [pw * x for x in v])
            else:
                seq[i][t] = 0 if rng.random() >= pw else rng.choices(PAY, weights=v)[0]
    if cond == "short_history":
        keep = set(range(c - 40, c)) | set(rng.sample(range(0, c - 40), 24))
    else:
        keep = set(range(c))
    return c, ms, seq, truth, keep


def write(rows, path, header):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pairs", nargs="+", default=["1,3", "2,5", "4,7"])
    ap.add_argument("--size-lag", default="first", choices=["first", "second"])
    ap.add_argument("--couple-lag", type=int, default=3); ap.add_argument("--beta-c", type=float, default=0.09)
    a = ap.parse_args()
    pairs = [tuple(int(x) for x in p.split(",")) for p in a.pairs]
    rng = random.Random(a.seed)
    out = Path(a.out)
    conds = ["stable", "short_history", "drift", "shared_change"]
    pooled = {}
    for split, per in (("train", 24), ("validation", 24), ("test", 48)):
        d = out / split; d.mkdir(parents=True, exist_ok=True)
        Q, H, T, C, Lb, R = [], [], [], [], [], []
        k = 0
        for cond in conds:
            for _ in range(per):
                eid = "sim-%s-e%05d" % (split, k); k += 1
                c, ms, seq, truth, keep = episode(rng, cond, pairs, a.couple_lag, a.beta_c, eid, a.size_lag)
                for i, m in enumerate(ms):
                    qid = m["mid"] + "-next"
                    Q.append([qid, eid, m["mid"], m["ty"], m["sg"], c])
                    H += [[eid, m["mid"], t, seq[i][t]] for t in sorted(keep)]
                    tv = truth[i]
                    T.append([qid, eid] + tv + [0]); C.append([qid, cond]); Lb.append([qid, 0])
                    if split == "train":
                        for t in keep: pooled.setdefault(m["ty"], [1] * 5)[[0, 1, 2, 5, 20].index(seq[i][t])] += 1
                    R.append([qid, m["ty"]])
        write(Q, d / "queries.csv", ["query_id", "episode_id", "machine_id", "machine_type", "signal", "cutoff"])
        write(H, d / "history.csv", ["episode_id", "machine_id", "tick", "payout"])
        if split != "test":
            write(C, d / "conditions.csv", ["query_id", "condition"]); write(Lb, d / "labels.csv", ["query_id", "outcome"])
        if split == "validation":
            write(T, d / "targets.csv", ["query_id", "episode_id", "p_0", "p_1", "p_2", "p_5", "p_20", "outcome"])
            ref = {ty: [x / sum(v) for x in v] for ty, v in pooled.items()}
            write([[q, *ref[ty]] for q, ty in R], d / "reference.csv", ["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"])
        write([[r[0], .2, .2, .2, .2, .2] for r in Q], d / "sample_submission.csv", ["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"])
    (out / "metadata.json").write_text(json.dumps({"round": "adaptation", "max_points": 40, "pairs": a.pairs,
                                                   "couple_lag": a.couple_lag}), encoding="utf-8")


if __name__ == "__main__":
    main()
