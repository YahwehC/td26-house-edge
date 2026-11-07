import sys
sys.dont_write_bytecode = True

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from he.api import load_params, predict_split, write_submission


def main(argv=None):
    parser = argparse.ArgumentParser(description="House Edge submission predictor")
    parser.add_argument("--data", required=True, help="folder containing test/")
    parser.add_argument("--out", required=True, help="submission CSV path")
    args = parser.parse_args(argv)

    params = load_params(HERE / "params.json")
    _, _, pred = predict_split(Path(args.data), "test", params)
    write_submission(Path(args.out), pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
