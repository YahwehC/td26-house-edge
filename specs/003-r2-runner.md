# 003 - Round-2 one-shot runner `r2.py`

## Goal
One command for round 2 (new kit drops; little time): fit main + fallback on train, score main / fallback / blends on
validation with the official metric, pick the best blend weight, refit on train+validation, write the test CSV,
build + verify the code ZIP, and print a compact diagnostics report. Must never leave us without a valid CSV:
if the main model crashes, fall back to the fallback model alone (weight 0).

## Files
Create only `r2.py` (project root). Do not modify other files. (Claude adds blend support to `he/api.py`; the
interface you rely on is below.)

## Interfaces you may use
- `he/api.py`: `fit_params(public, fit_splits=(...))` -> main params dict; `predict_split(public, split, params)`
  -> (queries, extra, pred); `score_validation(public, params)` -> dict with `points, R, ratios`
  (ratios: condition -> float); `write_submission(path, pred)`; `save_params(path, params)`.
  **Blend support (added by Claude):** if `params` has key `"blend": {"w_main": w}` and key `"fallback"`
  (output of `fit_fallback`), `predict_split` returns `w*main + (1-w)*fallback` per query; with `w_main == 0`
  the main model is not run and `params` may contain only `{"params_version": 1, "blend": {"w_main": 0.0},
  "fallback": ...}`.
- `he/fallback.py`: `fit_fallback(query_lists)` -> dict, `predict_fallback(queries, fparams)` -> {qid: row}.
- `he/model.py`: `load_split(public, split)`, `official_score(pred, extra, queries, max_points=60.0)`.
- `check_csv.py`: `check(csv_path, queries_csv_path)` -> list of error strings.
- `build_entry.py` CLI: `python build_entry.py --params P --csv C --data D --out Z` (prints VERIFIED on success,
  non-zero exit otherwise). Run it as a subprocess with `sys.executable`.

## CLI
`python r2.py --data <public dir> --name <run name> [--max-points 40] [--weights 0,0.5,0.7,0.8,0.9,0.95,1]`
Outputs: `runs/<name>/{submission.csv, params.json, validation_predictions.csv, r2_report.json}`, `dist/<name>.zip`.

## Steps
1. Print `metadata.json` of `<data>/validation` and `<data>/test` if present (round name, max points).
2. Load train/validation/test queries via `load_split`. Print counts: episodes, machines, machine types and their
   counts, cutoff min/max, condition counts in validation (from `extra["conditions"]`, column `condition`).
3. Fit main on train (`fit_params(data, fit_splits=("train",))`) inside try/except; on exception print the
   traceback and continue with main unavailable. Print from main params: top-6 pair votes (`params["votes"]`
   if present, print its structure compactly, truncated to 600 chars) and `params["lag_rule"]`.
4. Fit fallback on train only (`fit_fallback([train_queries])`).
5. Predictions on validation: main (if available) and fallback, computed once each; blends computed from these two
   prediction dicts (row-wise `w*a + (1-w)*b`), no re-prediction per weight. Score each weight with
   `official_score(pred, extra, val_queries, max_points)` (max_points from CLI; default 40 = round-2 max per kit
   RULES.md). Print a table: weight, points, R, each condition ratio (4 decimals). If main unavailable, only w=0.
6. Choose w* = weight with highest points; ties -> larger weight. Print it.
7. Refit on train+validation: main `fit_params(data, fit_splits=("train","validation"))` (skip if w* == 0) and
   fallback `fit_fallback([train, validation])`. Assemble params: main params dict (or the minimal dict when
   w* == 0) plus `"blend": {"w_main": w*}` and `"fallback": fp`. Save `runs/<name>/params.json`.
8. Write `validation_predictions.csv` (blend at w* from step 5 predictions) and test `submission.csv` via
   `predict_split(data, "test", params)` + `write_submission`. Run `check`; print OK or the errors.
9. Run build_entry as subprocess -> `dist/<name>.zip`; print its last 5 output lines and exit code.
10. Write `r2_report.json` (counts, weight table, w*, check result, build exit code). Print elapsed time per step.

## Constraints
- Stdlib only. English comments. Wrap steps 3, 7 (main refit) and 9 in try/except so one failure does not stop the
  rest; if the main refit fails, fall back to w* = 0 and say so loudly.
- Exit code 0 only if the CSV passes `check` and build_entry printed VERIFIED.

## Acceptance (Claude runs; from `challenges/house-edge`)
`.venv\Scripts\python.exe r2.py --data kit\public --name r2_dryrun --max-points 60`
Must: print the weight table (w=1 row ~57.9 points), write all outputs, CSV check OK, VERIFIED, exit code 0.
