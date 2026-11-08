"""Round-2 stage 2: refit the global-state model (main_v9 mechanisms) on a new kit and adopt it if validation gains.

Mechanisms (see HANDOVER.md, main_v9): the payout-size regime shifts the win logit per machine type, the number of
other machines that won at t - peer_lag shifts it too, and a shared persistent binary episode state (two-state
HMM, amplitude `scale`, switch rate `hazard`) moves all machines of an episode together.

Fitting uses only TRAIN histories (logistic GLM, then EM over the episode state); validation targets are used once,
to compare against the base entry. Offline fitting needs NumPy/SciPy; the packaged predictor stays stdlib-only.

    .venv\\Scripts\\python.exe r2g.py --data kit2\\public --base runs\\r2_v0\\params.json --name r2_v1
"""
import argparse
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from he.api import load_params, predict_queries, save_params, write_submission  # noqa: E402
from he.model import load_split, official_score  # noqa: E402
from he.global_state import state_rows  # noqa: E402
from he.weather import filter_episode  # noqa: E402


def peer_lag_from(params):
    """Lag of the strongest 'oth_win_k' term of the base entry's coupling (default 3)."""
    c = params.get("logit_corr") or {}
    best = None
    for n, z in zip(c.get("names", []), c.get("z", [])):
        if n.startswith("oth_win_") and (best is None or abs(z) > best[0]):
            best = (abs(z), int(n.rsplit("_", 1)[1]))
    return best[1] if best else 3


def design(rows, types):
    T = len(types)
    X = np.zeros((len(rows), 3 * T + 2))
    Y = np.zeros(len(rows))
    for k, r in enumerate(rows):
        i = types.index(r["type"])
        X[k, 2 * i + r["high"]] = 1
        X[k, 2 * T] = r["signal"]
        X[k, 2 * T + 1 + i] = r["regime"] - .5
        X[k, 3 * T + 1] = r["peer"] - 3.5
        Y[k] = r["y"]
    return X, Y


def fit_glm(X, Y, init):
    def loss(v):
        eta = X @ v
        prob = 1 / (1 + np.exp(-eta))
        return float(np.sum(np.logaddexp(0, eta) - Y * eta) + .001 * (v @ v)), X.T @ (prob - Y) + .002 * v
    return minimize(loss, init, jac=True, method="BFGS", options={"gtol": 1e-5, "maxiter": 300}).x


def fit_em(rows, X, Y, coef, log):
    by = defaultdict(lambda: defaultdict(list))
    for k, r in enumerate(rows):
        by[r["episode"]][r["tick"]].append(k)
    eta = X @ coef
    best = None
    for scale in (.2, .4, .6, .8):
        for hazard in (.005, .02, .08, .25):
            v = 0.0
            for ep, times in by.items():
                trace = (max(times) + 1, {t: [(float(eta[k]), int(Y[k])) for k in ids] for t, ids in times.items()})
                v += filter_episode(trace, scale, hazard)[0]
            if best is None or v > best[0]:
                best = (v, scale, hazard)
    _, scale, hazard = best
    log(f"[em] grid start scale {scale} hazard {hazard} (loglik {best[0]:.1f})")

    def smooth(eta, scale):
        expected = np.zeros(len(Y))
        ll = 0.0
        l0 = Y * (eta - scale) - np.logaddexp(0, eta - scale)
        l1 = Y * (eta + scale) - np.logaddexp(0, eta + scale)
        for ep, times in by.items():
            ticks = sorted(times)
            em = np.array([[l0[times[t]].sum(), l1[times[t]].sum()] for t in ticks])
            shift = em.max(1)
            em = np.exp(em - shift[:, None])
            f, norms, prob = [], [], np.array([.5, .5])
            for j, t in enumerate(ticks):
                if j:
                    stay = (1 + (1 - 2 * hazard) ** (t - ticks[j - 1])) / 2
                    prob = np.array([stay * prob[0] + (1 - stay) * prob[1], (1 - stay) * prob[0] + stay * prob[1]])
                prob = prob * em[j]
                norm = prob.sum()
                prob /= norm
                f.append(prob.copy())
                norms.append(norm)
                ll += math.log(norm) + shift[j]
            back = np.ones(2)
            for j in range(len(ticks) - 1, -1, -1):
                post = f[j] * back
                post /= post.sum()
                expected[times[ticks[j]]] = post[1]
                if j:
                    stay = (1 + (1 - 2 * hazard) ** (ticks[j] - ticks[j - 1])) / 2
                    v = em[j] * back
                    back = np.array([stay * v[0] + (1 - stay) * v[1], (1 - stay) * v[0] + stay * v[1]]) / norms[j]
        return expected, ll

    d = X.shape[1]
    for it in range(6):
        hot, ll = smooth(X @ coef, scale)

        def objective(v):
            e = X @ v[:d]
            s = v[d]
            lo, hi = e - s, e + s
            p0, p1 = 1 / (1 + np.exp(-lo)), 1 / (1 + np.exp(-hi))
            val = np.sum((1 - hot) * (np.logaddexp(0, lo) - Y * lo) + hot * (np.logaddexp(0, hi) - Y * hi)) \
                + .001 * np.dot(v[:d], v[:d])
            res = (1 - hot) * (p0 - Y) + hot * (p1 - Y)
            grad = np.r_[X.T @ res + .002 * v[:d], np.sum(-(1 - hot) * (p0 - Y) + hot * (p1 - Y))]
            return val, grad
        opt = minimize(objective, np.r_[coef, scale], jac=True, method="L-BFGS-B",
                       bounds=[(None, None)] * d + [(.01, 1.5)], options={"maxiter": 150, "ftol": 1e-10})
        coef, scale = opt.x[:d], float(opt.x[d])
        log(f"[em] iter {it} loglik {ll:.1f} scale {scale:.4f}")
    return coef, scale, hazard


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", required=True, help="params.json of the r2.py entry (fitted on train+validation)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-points", type=float, default=40.0)
    ap.add_argument("--peer-lag", type=int, default=None)
    ap.add_argument("--min-gain", type=float, default=0.02)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    log = lambda *x: print(*x, flush=True)
    data = Path(a.data)
    base = load_params(a.base)
    p = json.loads(json.dumps(base))
    p["cfg"].pop("sparse_joint", None)
    p.pop("logit_corr", None)  # the global-state model carries its own peer term
    p["cfg"]["peer_lag"] = a.peer_lag or peer_lag_from(base)
    log(f"[r2g] peer_lag {p['cfg']['peer_lag']}")
    train_q, _ = load_split(data, "train")
    rows, _ = state_rows(train_q, p)
    types = sorted({r["type"] for r in rows})
    log(f"[r2g] {len(rows)} train tick rows, types {types} ({time.perf_counter() - t0:.0f}s)")
    if len(types) != 3:
        log("[r2g] he/structured_state.py assumes 3 machine types -> not applicable, stop")
        return 2
    X, Y = design(rows, types)
    cp = p["cfg"]["cell_prior"]
    init = np.array([cp[ty][nm][0] for ty in types for nm in ("lo", "hi")] + [.3, -.8, -1, -.5, .09])
    coef = fit_glm(X, Y, init)
    log("[r2g] glm coef " + " ".join(f"{c:+.3f}" for c in coef))
    coef, scale, hazard = fit_em(rows, X, Y, coef, log)
    p["global_state"] = {"types": types, "coef": coef.tolist(), "scale": scale, "hazard": hazard}
    p["global_state_fit"] = {"method": "r2g.py: GLM + two-state episode HMM by EM", "source": "train histories only"}
    log(f"[r2g] fitted global state ({time.perf_counter() - t0:.0f}s): {p['global_state']}")

    val_q, val_ex = load_split(data, "validation")
    s0 = official_score(predict_queries(val_q, base), val_ex, val_q, a.max_points)
    pv = predict_queries(val_q, p)
    s1 = official_score(pv, val_ex, val_q, a.max_points)
    gain = s1["points"] - s0["points"]
    log(f"[r2g] validation base {s0['points']:.3f} {s0['ratios']}")
    log(f"[r2g] validation glob {s1['points']:.3f} {s1['ratios']}  gain {gain:+.3f}")
    out = HERE / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    rep = {"base": s0, "global": s1, "gain": gain, "global_state": p["global_state"], "peer_lag": p["cfg"]["peer_lag"]}
    (out / "r2g_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    if gain < a.min_gain:
        log(f"[r2g] gain < {a.min_gain} -> keep the base entry; nothing built")
        return 1
    save_params(out / "params.json", p)
    write_submission(out / "validation_predictions.csv", pv)
    test_q, _ = load_split(data, "test")
    write_submission(out / "submission.csv", predict_queries(test_q, p))
    py = sys.executable
    chk = subprocess.run([py, str(HERE / "check_csv.py"), str(out / "submission.csv"), str(data / "test" / "queries.csv")],
                         capture_output=True, text=True)
    log("[check]", chk.stdout.strip().splitlines()[-1] if chk.stdout.strip() else chk.stderr[-300:])
    zp = HERE / "dist" / f"{a.name}.zip"
    b = subprocess.run([py, str(HERE / "build_entry.py"), "--params", str(out / "params.json"), "--csv",
                        str(out / "submission.csv"), "--data", str(data), "--out", str(zp)], capture_output=True, text=True)
    tail = (b.stdout + b.stderr).strip().splitlines()[-3:]
    log("[build]", " | ".join(tail))
    log(f"[done] {a.name}: validation {s1['points']:.3f} (base {s0['points']:.3f}), total {time.perf_counter() - t0:.0f}s")
    return 0 if chk.returncode == 0 and b.returncode == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
