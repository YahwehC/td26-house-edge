# House Edge - team pipeline

Latest main-round candidate: **main_v13**, 59.123069 on episode-held-out calibration,
versus main_v12 59.117106. Use `runs/main_v13/submission.csv` with `dist/main_v13.zip`.
The ZIP reproduces all 1,536 test predictions exactly under Python 3.13 using only
the standard library. The small public-score gain does not establish a hidden-test
gain; retain v12. See `HANDOVER.md` for the experiment audit and current findings.

## Run

```sh
.venv\Scripts\python.exe run.py --data kit\public --out runs\main_v1\submission.csv
```

This fits on train, scores validation (`runs/main_v1/validation_score.json`), writes
`runs/main_v1/validation_predictions.csv`, refits on train+validation (`runs/main_v1/params.json`),
predicts test to `--out`, then runs `check_csv.py`. Exit code is non-zero if the checker fails.
`--workdir` overrides the default (the folder of `--out`).

```sh
.venv\Scripts\python.exe check_csv.py <csv> <queries.csv>
.venv\Scripts\python.exe build_entry.py --params runs\main_v1\params.json --csv runs\main_v1\submission.csv --data kit\public --out dist\main_v1.zip
```

`build_entry.py` stages `entry_template/` + `params.json` + `he/*.py` in `dist/main_v1_src/`,
zips them (README.md and reproduce.json at the root), then extracts the zip to a fresh folder,
reruns `predict.py` there and compares against `--csv` (max abs diff <= 1e-9 prints `VERIFIED`).

## Layout

```
run.py              one-command pipeline
check_csv.py        stdlib submission format checker (importable: check(csv, queries))
build_entry.py      package + verify the code ZIP
he/                 model code (pure stdlib), not edited by the pipeline
entry_template/     predict.py, reproduce.json, README.md copied into the ZIP
kit/public/         current round data
runs/               outputs (params.json, validation predictions/score, submission CSV)
dist/               code ZIPs and staging folders
```

## Round 2 (adaptation)

Extract the new kit to `kit2/`, then:

```sh
.venv\Scripts\python.exe run.py --data kit2\public --out runs\r2\submission.csv
.venv\Scripts\python.exe build_entry.py --params runs\r2\params.json --csv runs\r2\submission.csv --data kit2\public --out dist\r2.zip
```

Upload `runs/r2/submission.csv` and `dist/r2.zip` separately.
