"""Standard-library format checker for House Edge submission CSVs.

Usage: python check_csv.py <csv> <queries.csv>
Import: from check_csv import check; errors = check(csv_path, queries_path)
"""
import csv
import math
import sys

HEADER = "query_id,p_0,p_1,p_2,p_5,p_20"
PCOLS = ("p_0", "p_1", "p_2", "p_5", "p_20")


def _read_expected(queries_path):
    with open(queries_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or "query_id" not in header:
            raise ValueError(f"{queries_path}: missing query_id column")
        i = header.index("query_id")
        ids = []
        for row in reader:
            if not row:
                continue
            if i >= len(row):
                raise ValueError(f"{queries_path}: short row {row!r}")
            ids.append(row[i])
    return ids


def _check(csv_path, queries_path):
    errors = []
    expected = _read_expected(queries_path)
    expected_set = set(expected)
    if len(expected_set) != len(expected):
        errors.append(f"queries.csv lists {len(expected) - len(expected_set)} duplicate query_id(s)")

    seen = set()
    nrows = 0
    try:
        f = open(csv_path, newline="", encoding="utf-8")
    except OSError as exc:
        return [f"cannot open {csv_path}: {exc}"], 0
    with f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return [f"{csv_path}: empty file"], 0
        if header != HEADER.split(","):
            errors.append(f"{csv_path}: bad header {','.join(header)!r}; expected {HEADER!r}")
        for lineno, row in enumerate(reader, start=2):
            if not row:
                errors.append(f"line {lineno}: blank row")
                continue
            nrows += 1
            if len(row) != 6:
                errors.append(f"line {lineno}: expected 6 fields, got {len(row)}")
                continue
            qid = row[0]
            if not qid:
                errors.append(f"line {lineno}: empty query_id")
            if qid in seen:
                errors.append(f"line {lineno}: duplicate query_id {qid!r}")
            seen.add(qid)
            if qid not in expected_set:
                errors.append(f"line {lineno}: extra query_id {qid!r}")
            vals = []
            for name, text in zip(PCOLS, row[1:]):
                try:
                    v = float(text)
                except ValueError:
                    errors.append(f"line {lineno}: {name} is not a float: {text!r}")
                    continue
                if not math.isfinite(v):
                    errors.append(f"line {lineno}: {name} is not finite: {text!r}")
                    continue
                if not (0.0 <= v <= 1.0):
                    errors.append(f"line {lineno}: {name} outside [0,1]: {text!r}")
                    continue
                vals.append(v)
            if len(vals) == 5:
                total = math.fsum(vals)
                if abs(total - 1.0) > 1e-6:
                    errors.append(f"line {lineno}: row sums to {total!r}, not 1 within 1e-6")

    for qid in sorted(expected_set - seen):
        errors.append(f"missing query_id {qid!r}")
    return errors, nrows


def check(csv_path, queries_path):
    """Return a list of human-readable errors (empty list means valid)."""
    return _check(csv_path, queries_path)[0]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: python check_csv.py <csv> <queries.csv>")
        return 2
    try:
        errors, nrows = _check(argv[0], argv[1])
    except (OSError, ValueError) as exc:
        print(f"ERROR {exc}")
        return 2
    if errors:
        print(f"FAIL {len(errors)} error(s)")
        for err in errors:
            print(err)
        return 1
    print(f"OK {nrows} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
