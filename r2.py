"""Round-2 one-shot runner: fit main + fallback, score blends, refit, submit, zip.

Usage:
    python r2.py --data kit/public --name r2_dryrun --max-points 60

Never leaves the run without a CSV: if the main model cannot be fitted or predicted,
the runner continues with the fallback model alone (w_main = 0).
"""
import argparse
import json
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from check_csv import check
from he.api import PARAMS_VERSION, fit_params, predict_queries, predict_split, save_params, write_submission
from he.corr import apply_corr
from he.coupling import fit_coupling, pairs_from_cfg
from he.fallback import fit_fallback, predict_fallback
from he.model import load_split, official_score

COND_ORDER = ("stable", "short_history", "drift", "shared_change")


def elapsed(t0):
    return time.perf_counter() - t0


def parse_weights(text):
    """Parse a comma-separated list of blend weights in [0, 1]."""
    try:
        weights = [float(part) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad --weights value: {exc}")
    if not weights:
        raise argparse.ArgumentTypeError("--weights must contain at least one value")
    for w in weights:
        if not 0.0 <= w <= 1.0:
            raise argparse.ArgumentTypeError(f"weight {w} outside [0, 1]")
    return sorted(set(weights))  # ascending, so a tie keeps the larger weight


def blend(main_pred, fb_pred, w):
    """Row-wise blend of two prediction dicts; w == 1/0 return the source dict."""
    if w >= 1.0:
        return main_pred
    if w <= 0.0:
        return fb_pred
    return {qid: [w * a + (1.0 - w) * b for a, b in zip(main_pred[qid], fb_pred[qid])]
            for qid in main_pred}


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"[metadata] cannot read {path}: {exc}")
        return None


def split_counts(queries, extra, split):
    """Compact data diagnostics for one split."""
    episodes = len({q["episode_id"] for q in queries})
    machines = len({(q["episode_id"], q["machine_id"]) for q in queries})
    types = Counter(q["machine_type"] for q in queries)
    cutoffs = [q["cutoff"] for q in queries]
    counts = {
        "queries": len(queries),
        "episodes": episodes,
        "machines": machines,
        "machine_types": dict(sorted(types.items())),
        "cutoff_min": min(cutoffs) if cutoffs else None,
        "cutoff_max": max(cutoffs) if cutoffs else None,
    }
    print(f"[data] {split}: queries={len(queries)} episodes={episodes} machines={machines} "
          f"cutoff={counts['cutoff_min']}-{counts['cutoff_max']}")
    print(f"[data] {split} machine types: {counts['machine_types']}")
    if "conditions" in extra:
        cond = Counter(r["condition"] for r in extra["conditions"])
        ordered = {c: cond[c] for c in COND_ORDER if c in cond}
        ordered.update({c: cond[c] for c in sorted(cond) if c not in ordered})
        counts["conditions"] = ordered
        print(f"[data] {split} conditions: {ordered}")
    return counts


def compact_votes(params):
    """Top-6 pair votes plus a compact view of the whole votes structure."""
    votes = params.get("votes") or {}
    pairs = votes.get("pairs") or {}
    top = sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))[:6]
    joint = votes.get("joint") or {}
    joint_top = dict(sorted(joint.items(), key=lambda kv: (-kv[1], kv[0]))[:6])
    compact = {
        "pairs_top6": dict(top),
        "lags": votes.get("lags"),
        "n_pair": votes.get("n_pair"),
        "n_lag": votes.get("n_lag"),
        "joint_top6": joint_top,
    }
    return dict(top), compact


def choose_logit_corr(val_q, val_extra, params, max_points, train_q=None, train_extra=None, min_gain=0.02, cfg=None):
    """Cross-machine coupling (he/coupling.py): fit on TRAIN ticks, check out-of-sample on validation targets;
    if it gains >= min_gain, refit on train+validation ticks (histories only) and return it, else None."""
    if not val_extra.get("targets") or train_q is None:
        print("[corr] no validation targets / train - skipped")
        return None
    cm = lambda ex: {r["query_id"]: r["condition"] for r in (ex or {}).get("conditions", [])}
    base_params = {k: v for k, v in params.items() if k != "logit_corr"}
    pred = predict_queries(val_q, base_params)
    base = official_score(pred, val_extra, val_q, max_points)
    pairs = pairs_from_cfg(cfg or params.get("cfg"))  # the kit's fitted lag pairs (main model)
    c_tr = fit_coupling([train_q], [cm(train_extra)], pairs=pairs)
    if not c_tr:
        print("[corr] nothing significant on train ticks - not adopted")
        return None
    s = official_score(apply_corr(val_q, pred, c_tr), val_extra, val_q, max_points)
    gain = s["points"] - base["points"]
    print(f"[corr] train-fitted coupling on validation: {base['points']:.3f} -> {s['points']:.3f} ({gain:+.3f}) "
          f"ratios {s['ratios']}")
    if gain < min_gain:
        print("[corr] not adopted")
        return None
    c = fit_coupling([train_q, val_q], [cm(train_extra), cm(val_extra)], pairs=pairs)
    print(f"[corr] adopted (refit on train+validation ticks): {c and c['names']}")
    return c


def main(argv=None):
    parser = argparse.ArgumentParser(description="House Edge round-2 one-shot runner")
    parser.add_argument("--data", required=True, type=Path, help="public folder (train/validation/test)")
    parser.add_argument("--name", required=True, help="run name (runs/<name>, dist/<name>.zip)")
    parser.add_argument("--max-points", type=float, default=40.0,
                        help="max points for the official metric (round-2 default 40)")
    parser.add_argument("--weights", type=parse_weights, default=parse_weights("0,0.5,0.7,0.8,0.9,0.95,1"),
                        help="comma-separated blend weights")
    args = parser.parse_args(argv)

    data = args.data.resolve()
    if not (data / "train" / "queries.csv").is_file():
        raise SystemExit(f"{data}: not a public data folder (missing train/queries.csv)")

    run_dir = ROOT / "runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    params_path = run_dir / "params.json"
    val_csv_path = run_dir / "validation_predictions.csv"
    sub_csv_path = run_dir / "submission.csv"
    report_path = run_dir / "r2_report.json"
    zip_path = ROOT / "dist" / f"{args.name}.zip"

    total_t0 = time.perf_counter()
    timings = {}

    # ------------------------------------------------------------------ step 1
    t0 = time.perf_counter()
    print(f"[step 1] metadata (max_points used for scoring: {args.max_points:g})")
    metadata = {}
    meta_paths = (
        ("data", data / "metadata.json"),
        ("validation", data / "validation" / "metadata.json"),
        ("test", data / "test" / "metadata.json"),
    )
    for label, path in meta_paths:
        if not path.is_file():
            continue
        meta = read_json(path)
        if meta is None:
            continue
        metadata[label] = meta
        print(f"[metadata] {label}: round={meta.get('round')!r} max_points={meta.get('max_points')!r}")
    timings["1_metadata"] = elapsed(t0)
    print(f"[step 1] done in {timings['1_metadata']:.1f}s")

    # ------------------------------------------------------------------ step 2
    t0 = time.perf_counter()
    print("[step 2] loading splits ...")
    splits = {}
    for split in ("train", "validation", "test"):
        splits[split] = load_split(data, split)
    counts = {}
    for split in ("train", "validation", "test"):
        queries, extra = splits[split]
        counts[split] = split_counts(queries, extra, split)
    train_q = splits["train"][0]
    val_q, val_extra = splits["validation"]
    timings["2_data"] = elapsed(t0)
    print(f"[step 2] done in {timings['2_data']:.1f}s")

    # ------------------------------------------------------------------ step 3
    t0 = time.perf_counter()
    print("[step 3] fitting main model on train ...")
    main_params = None
    top_pairs = {}
    try:
        main_params = fit_params(data, fit_splits=("train",))
        top_pairs, votes_compact = compact_votes(main_params)
        print(f"[main] top-6 pair votes: {top_pairs}")
        print(f"[main] votes: {repr(votes_compact)[:600]}")
        print(f"[main] lag_rule: {main_params.get('lag_rule')}")
        print(f"[main] fitted on train ({len(main_params)} keys)")
    except Exception:
        print("[main] fit on train FAILED; continuing with the fallback model only")
        traceback.print_exc()
        main_params = None
    timings["3_fit_main"] = elapsed(t0)
    print(f"[step 3] done in {timings['3_fit_main']:.1f}s (main_available={main_params is not None})")

    # ------------------------------------------------------------------ step 4
    t0 = time.perf_counter()
    print("[step 4] fitting fallback on train ...")
    fb_params = fit_fallback([train_q])
    print(f"[fallback] fitted on train ({len(train_q)} queries)")
    timings["4_fit_fallback"] = elapsed(t0)
    print(f"[step 4] done in {timings['4_fit_fallback']:.1f}s")

    # ------------------------------------------------------------------ step 5
    t0 = time.perf_counter()
    print("[step 5] predicting validation (main / fallback, once each) ...")
    main_pred = None
    if main_params is not None:
        try:
            _, _, main_pred = predict_split(data, "validation", main_params)
            print(f"[main] validation predictions: {len(main_pred)} rows")
        except Exception:
            print("[main] validation prediction FAILED; continuing with the fallback model only")
            traceback.print_exc()
            main_pred = None
    fb_pred = predict_fallback(val_q, fb_params)
    print(f"[fallback] validation predictions: {len(fb_pred)} rows")

    weights = list(args.weights) if main_pred is not None else [0.0]
    if main_pred is None:
        print("[blend] main unavailable -> evaluating w=0 only")
    table = []
    for w in weights:
        pred = blend(main_pred, fb_pred, w)
        score = official_score(pred, val_extra, val_q, args.max_points)
        table.append({"w": w, "points": score["points"], "R": score["R"], "ratios": score["ratios"]})

    ratio_keys = [c for c in COND_ORDER if c in table[0]["ratios"]]
    ratio_keys += [c for c in sorted(table[0]["ratios"]) if c not in ratio_keys]
    header = f"{'weight':>7} {'points':>8} {'R':>8} " + " ".join(f"{c:>12}" for c in ratio_keys)
    print("[blend] validation scores")
    print(header)
    for row in table:
        ratios = " ".join(f"{row['ratios'].get(c, float('nan')):>12.4f}" for c in ratio_keys)
        print(f"{row['w']:>7.3f} {row['points']:>8.2f} {row['R']:>8.4f} {ratios}")
    timings["5_validate"] = elapsed(t0)
    print(f"[step 5] done in {timings['5_validate']:.1f}s")

    # ------------------------------------------------------------------ step 6
    t0 = time.perf_counter()
    best = table[0]
    for row in table[1:]:
        if row["points"] >= best["points"]:  # ascending weights -> larger weight wins ties
            best = row
    w_star = best["w"]
    print(f"[choose] w* = {w_star:g} (validation points {best['points']:.2f}, R {best['R']:.4f})")
    timings["6_choose"] = elapsed(t0)
    print(f"[step 6] done in {timings['6_choose']:.1f}s")

    # ------------------------------------------------------------------ step 7
    t0 = time.perf_counter()
    print("[step 7] refitting on train+validation ...")
    main_full = None
    main_refit_ok = None  # None = skipped (w* == 0)
    if w_star == 0.0:
        print("[refit] w*=0 -> skipping main refit")
    else:
        try:
            main_full = fit_params(data, fit_splits=("train", "validation"))
            main_refit_ok = True
            print("[refit] main fitted on train+validation")
        except Exception:
            print("!!! main refit on train+validation FAILED; falling back to w*=0 (fallback only)")
            traceback.print_exc()
            w_star = 0.0
            main_full = None
            main_refit_ok = False
    fallback_refit_ok = True
    try:
        fb_full = fit_fallback([train_q, val_q])
        print("[refit] fallback fitted on train+validation")
    except Exception:
        print("!!! fallback refit FAILED; reusing the train-only fallback params")
        traceback.print_exc()
        fb_full = fb_params
        fallback_refit_ok = False
    if w_star == 0.0:
        params = {
            "params_version": PARAMS_VERSION,
            "blend": {"w_main": 0.0},
            "fallback": fb_full,
        }
    else:
        params = dict(main_full)
        params["blend"] = {"w_main": w_star}
        params["fallback"] = fb_full
    # step 7b: episode-heat logit correction, adopted only if 2-fold CV (by episode) on validation gains
    try:
        corr = choose_logit_corr(val_q, val_extra, params, args.max_points, train_q, splits["train"][1],
                                 cfg=(main_full or {}).get("cfg") if isinstance(main_full, dict) else None)
        if corr:
            params["logit_corr"] = corr
    except Exception:
        print("!!! logit correction FAILED; continuing without it")
        traceback.print_exc()
        params.pop("logit_corr", None)
    save_params(params_path, params)
    print(f"[refit] wrote {params_path}")
    timings["7_refit"] = elapsed(t0)
    print(f"[step 7] done in {timings['7_refit']:.1f}s (w*={w_star:g})")

    # ------------------------------------------------------------------ step 8
    t0 = time.perf_counter()
    print("[step 8] writing validation predictions and test submission ...")
    val_blend = blend(main_pred, fb_pred, w_star)
    if params.get("logit_corr"):
        val_blend = apply_corr(val_q, val_blend, params["logit_corr"])
    write_submission(val_csv_path, val_blend)
    print(f"[write] {val_csv_path} ({len(val_blend)} rows)")
    _, _, test_pred = predict_split(data, "test", params)
    write_submission(sub_csv_path, test_pred)
    print(f"[write] {sub_csv_path} ({len(test_pred)} rows)")
    errors = check(sub_csv_path, data / "test" / "queries.csv")
    if errors:
        print(f"[check] FAILED with {len(errors)} error(s)")
        for err in errors:
            print(f"[check] {err}")
    else:
        print(f"[check] OK {len(test_pred)} rows")
    timings["8_predict_write_check"] = elapsed(t0)
    print(f"[step 8] done in {timings['8_predict_write_check']:.1f}s")

    # ------------------------------------------------------------------ step 9
    t0 = time.perf_counter()
    print("[step 9] building + verifying the code ZIP ...")
    build_rc = None
    build_verified = False
    try:
        proc = subprocess.run(
            [sys.executable, "build_entry.py",
             "--params", str(params_path),
             "--csv", str(sub_csv_path),
             "--data", str(data),
             "--out", str(zip_path)],
            cwd=str(ROOT), capture_output=True, text=True)
        build_rc = proc.returncode
        lines = (proc.stdout or "").splitlines() + (proc.stderr or "").splitlines()
        for line in lines[-5:]:
            print(f"[build] {line}")
        build_verified = proc.returncode == 0 and "VERIFIED" in (proc.stdout or "")
        print(f"[build] exit code {build_rc}, verified={build_verified}, zip={zip_path}")
    except Exception:
        print("!!! build_entry could not be run")
        traceback.print_exc()
        build_rc = -1
    timings["9_build"] = elapsed(t0)
    print(f"[step 9] done in {timings['9_build']:.1f}s")

    # ------------------------------------------------------------------ step 10
    t0 = time.perf_counter()
    report = {
        "name": args.name,
        "data": str(data),
        "max_points": args.max_points,
        "metadata": metadata,
        "counts": counts,
        "main": {
            "available": main_params is not None,
            "top6_pairs": top_pairs,
            "lag_rule": main_params.get("lag_rule") if main_params else None,
            "refit_ok": main_refit_ok,
            "fallback_refit_ok": fallback_refit_ok,
        },
        "weights": table,
        "w_main": w_star,
        "check": {"ok": not errors, "errors": errors},
        "build": {"exit_code": build_rc, "verified": build_verified, "zip": str(zip_path)},
        "elapsed_seconds": {k: round(v, 3) for k, v in timings.items()},
        "total_seconds": round(time.perf_counter() - total_t0, 3),
        "exit_code": 0 if (not errors and build_verified) else 1,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[report] wrote {report_path}")
    timings["10_report"] = elapsed(t0)
    print(f"[step 10] done in {timings['10_report']:.1f}s")

    ok = not errors and build_verified
    print(f"[done] name={args.name} w*={w_star:g} check={'OK' if not errors else 'FAIL'} "
          f"build={'VERIFIED' if build_verified else 'FAILED'} total={time.perf_counter() - total_t0:.1f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
