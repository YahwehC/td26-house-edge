"""Re-fit params.json from a public/ data folder (train + validation). Pure stdlib, deterministic.

    python train.py --data /path/to/public --out params.json
"""
import sys
sys.dont_write_bytecode = True

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from he.api import fit_params, save_params


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fit House Edge parameters")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    save_params(Path(a.out), fit_params(Path(a.data), fit_splits=("train", "validation")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
