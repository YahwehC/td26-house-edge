"""One-command House Edge pipeline: fit, validate, predict test, check the CSV."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from he.api import fit_params, predict_split, save_params, score_validation, write_submission
from check_csv import check


def _secs(t):
    return f"{time.perf_counter() - t:.1f}s"


def _ratios(r):
    return "{" + ", ".join(f"{k}: {v}" for k, v in r.items()) + "}"


def main(argv=None):
    parser = argparse.ArgumentParser(description="House Edge end-to-end pipeline")
    parser.add_argument("--data", required=True, type=Path, help="public folder (train/validation/test)")
    parser.add_argument("--out", required=True, type=Path, help="test submission CSV")
    parser.add_argument("--workdir", type=Path, default=None, help="default: folder of --out")
    args = parser.parse_args(argv)

    data = args.data.resolve()
    out = args.out.resolve()
    workdir = args.workdir.resolve() if args.workdir else out.parent
    workdir.mkdir(parents=True, exist_ok=True)

    t = time.perf_counter()
    print("[1/4] fitting on train ...", flush=True)
    pv = fit_params(data, fit_splits=("train",))
    print(f"[1/4] fitted train params ({_secs(t)})", flush=True)

    t = time.perf_counter()
    if (data / "validation" / "targets.csv").exists():
        s = score_validation(data, pv)
        print(f"[1/4] validation points={s['points']:.2f} R={s['R']:.4f} ratios={_ratios(s['ratios'])} ({_secs(t)})",
              flush=True)
        (workdir / "validation_score.json").write_text(
            json.dumps(s, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        s = None
        print("[1/4] validation has no targets.csv - scoring skipped", flush=True)

    t = time.perf_counter()
    _, _, val_pred = predict_split(data, "validation", pv)
    write_submission(workdir / "validation_predictions.csv", val_pred)
    print(f"[1/4] wrote validation_predictions.csv ({len(val_pred)} rows, {_secs(t)})", flush=True)
    if s is not None:
        print(f"[1/4] wrote validation_score.json ({_secs(t)})", flush=True)

    t = time.perf_counter()
    print("[2/4] fitting on train+validation ...", flush=True)
    p = fit_params(data, fit_splits=("train", "validation"))
    save_params(workdir / "params.json", p)
    print(f"[2/4] wrote params.json ({_secs(t)})", flush=True)

    t = time.perf_counter()
    _, _, pred = predict_split(data, "test", p)
    write_submission(out, pred)
    print(f"[3/4] wrote {out} ({len(pred)} rows, {_secs(t)})", flush=True)

    t = time.perf_counter()
    errors = check(out, data / "test" / "queries.csv")
    if errors:
        print(f"[4/4] checker FAILED with {len(errors)} error(s) ({_secs(t)})", flush=True)
        for err in errors:
            print(err)
        return 1
    print(f"[4/4] checker OK {len(pred)} rows ({_secs(t)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
