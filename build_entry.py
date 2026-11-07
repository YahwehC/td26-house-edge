"""Package entry_template + params.json + he/*.py into a zip and verify it reproduces the CSV."""
import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "entry_template"
HEADER = ["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"]


def _read_submission(path):
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != HEADER:
            raise ValueError(f"{path}: bad header {header!r}")
        for lineno, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != 6:
                raise ValueError(f"{path}: line {lineno} has {len(row)} fields")
            qid = row[0]
            if qid in rows:
                raise ValueError(f"{path}: duplicate query_id {qid!r}")
            rows[qid] = [float(x) for x in row[1:]]
    return rows


def _compare(expected_path, actual_path):
    expected = _read_submission(expected_path)
    actual = _read_submission(actual_path)
    if set(expected) != set(actual):
        missing = len(set(expected) - set(actual))
        extra = len(set(actual) - set(expected))
        raise SystemExit(f"id mismatch: {missing} missing, {extra} extra")
    max_diff = 0.0
    for qid, exp in expected.items():
        act = actual[qid]
        for a, b in zip(exp, act):
            max_diff = max(max_diff, abs(a - b))
    if max_diff > 1e-9:
        raise SystemExit(f"max abs diff {max_diff:.3e} exceeds 1e-9")
    return max_diff


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build and verify the House Edge code zip")
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    params = args.params.resolve()
    expected_csv = args.csv.resolve()
    data = args.data.resolve()
    out = args.out.resolve()
    src = Path(str(out.with_suffix("")) + "_src")

    if src.exists():
        shutil.rmtree(src)
    shutil.copytree(TEMPLATE, src, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(params, src / "params.json")
    he_src = ROOT / "he"
    he_dst = src / "he"
    he_dst.mkdir()
    he_files = sorted(he_src.glob("*.py"))
    for f in he_files:
        shutil.copyfile(f, he_dst / f.name)

    files = sorted(p for p in src.rglob("*") if p.is_file())
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, p.relative_to(src).as_posix())
    print(f"zip: {out} ({out.stat().st_size} bytes)")

    tmp = Path(tempfile.mkdtemp(prefix=".verify-", dir=str(out.parent)))
    try:
        with zipfile.ZipFile(out) as zf:
            zf.extractall(tmp)
        sub = tmp / "sub.csv"
        result = subprocess.run(
            [sys.executable, "predict.py", "--data", str(data), "--out", str(sub)],
            cwd=str(tmp), capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout, end="")
            print(result.stderr, end="", file=sys.stderr)
            print(f"predict.py failed with exit code {result.returncode}")
            return 1
        if not sub.exists():
            print("predict.py did not write the requested CSV")
            return 1
        max_diff = _compare(expected_csv, sub)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"files: {len(files)}")
    print(f"max abs diff: {max_diff:.3e}")
    print("VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
