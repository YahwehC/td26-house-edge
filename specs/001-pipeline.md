# 001 — Reproducible pipeline + submission packaging (House Edge)

Work dir: `F:\CYH\projects\datathon\challenges\house-edge` (call it ROOT). Python: `ROOT\.venv\Scripts\python.exe`.
**Do not modify anything under `ROOT\he\` or `ROOT\kit\`.** Do not run any script inside `ROOT\kit\`.
All new code must use **only the Python standard library** (the official runner is Python 3.13 stdlib-only).

## Existing API you must use (in `ROOT\he\api.py`, read it)
- `fit_params(public, fit_splits=("train","validation"), size_split="validation") -> dict`
- `predict_split(public, split, params) -> (queries, extra, pred)`; `pred` = dict query_id -> 5 probs
- `write_submission(path, pred)`, `save_params(path, params)`, `load_params(path)`
- `score_validation(public, params) -> dict` (keys: points, R, ratios, err)
`public` = folder containing `train/`, `validation/`, `test/` (each with queries.csv, history.csv, ...).
Kit data for testing: `ROOT\kit\public`.

## Files to create

### 1. `ROOT\run.py` — one-command pipeline
`python run.py --data <public_dir> --out <csv> [--workdir <dir>]` (workdir default: folder of --out).
Steps, printing one short line per step with elapsed seconds:
1. `pv = fit_params(data, fit_splits=("train",))`; `s = score_validation(data, pv)`; print points/R/ratios;
   write `<workdir>/validation_predictions.csv` (via `predict_split(data,"validation",pv)` + `write_submission`)
   and `<workdir>/validation_score.json`. If validation has no targets.csv, skip scoring with a message.
2. `p = fit_params(data, fit_splits=("train","validation"))`; `save_params(<workdir>/params.json, p)`.
3. `predict_split(data, "test", p)` → `write_submission(--out, pred)`.
4. Run the checker (below) on --out against `<data>/test/queries.csv`; exit non-zero if it fails.
Use `sys.path.insert(0, <folder of run.py>)` so `import he.api` works from any cwd.

### 2. `ROOT\check_csv.py` — stdlib format checker
`python check_csv.py <csv> <queries.csv>`; also importable `check(csv_path, queries_path) -> list[str]` (errors).
Rules: header exactly `query_id,p_0,p_1,p_2,p_5,p_20`; every query_id in queries.csv exactly once, no extras,
no duplicates; all 5 values parse as finite floats in [0,1]; each row sums to 1 within 1e-6. Print `OK <n> rows` or the errors.

### 3. `ROOT\entry_template\` — files copied into the code ZIP
- `predict.py`: `python predict.py --data /data --out /output/submission.csv`. Set `sys.dont_write_bytecode = True`
  first. Adds its own folder to sys.path, `load_params(<its folder>/params.json)`, `predict_split(data,"test",params)`,
  `write_submission(out, pred)`. Writes nothing else. Must work when its folder is read-only.
- `reproduce.json`: exactly `{"image": "python", "command": ["python", "predict.py", "--data", "/data", "--out", "/output/submission.csv"]}`
- `README.md`: short: what the model is (structured Bayesian mixture; see below), Python 3.13 stdlib only, no seeds
  needed (fully deterministic), params.json fitted by `run.py` on the round's train+validation, command to recreate.

### 4. `ROOT\build_entry.py` — package and verify
`python build_entry.py --params <params.json> --csv <submission.csv> --data <public_dir> --out <zip>`
1. Make a fresh folder `<zip without .zip>_src/` (delete if exists): copy `entry_template/*`, `params.json`,
   and `ROOT\he\*.py` into `he/` (only .py files, no __pycache__).
2. Zip it so that `README.md` and `reproduce.json` are at the zip root (no enclosing folder), deflate.
3. Verify: extract the zip into a new temp folder, run `<sys.executable> predict.py --data <public_dir> --out <tmp>/sub.csv`
   with cwd = extracted folder, then compare to `--csv`: same ids, max abs diff per probability <= 1e-9. Print
   max diff and `VERIFIED` or fail with non-zero exit. Also print zip size and file count.

### 5. `ROOT\README.md` — team-facing, short
How to run (`run.py`, `build_entry.py`), folder layout, how to do round 2 (extract new kit to `kit2/`, run
`python run.py --data kit2/public --out runs/r2/submission.csv`, then build_entry with that params.json and data).

## Acceptance (run all, all must pass; paste outputs in your report)
```
cd F:\CYH\projects\datathon\challenges\house-edge
.venv\Scripts\python.exe run.py --data kit\public --out runs\main_v1\submission.csv
.venv\Scripts\python.exe check_csv.py runs\main_v1\submission.csv kit\public\test\queries.csv
.venv\Scripts\python.exe check_csv.py runs\main_v1\validation_predictions.csv kit\public\validation\queries.csv
.venv\Scripts\python.exe build_entry.py --params runs\main_v1\params.json --csv runs\main_v1\submission.csv --data kit\public --out dist\main_v1.zip
```
Expected: validation points ≈ 57.37 (printed by run.py), checker OK 1536 rows / 768 rows, build_entry prints VERIFIED.
Report: the printed lines only (no code dump).
