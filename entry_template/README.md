# House Edge submission

## Recreate the forecast
```sh
python predict.py --data /data --out /output/submission.csv
```
Reads `/data/test/{queries,history}.csv` and the bundled `params.json`; writes only `--out`.

## Environment
- Python 3.13 (official `python` image), **standard library only** - no third-party dependencies.
- Fully deterministic: no random numbers, no seeds, no network. Runtime ~10 s on one core.

## Model
Structured Bayesian forecaster (`he/`):
- Win chance of a machine depends on whether its wins at two lags (A, B) agree (XNOR cell), with
  Beta-Binomial cell probabilities and type/signal-dependent hi/lo cell priors.
- Episode-level mixture: stable / drift (each machine's cell mapping flips at its own tick) /
  shared change (all machines flip at a common tick); lag pair and flip tick are marginalised exactly.
- Payout size given a win: two size vectors per machine type; the active one is
  (payout at lag L >= 2) XOR sign, with L coupled to the win pair; posterior-mixed.
- Insurance blend (`he/fallback.py`): a generic Bayesian context model (any boolean function of up to two
  win lags, optional single change point; size given win conditioned on one big-payout lag), averaged over
  contexts with empirical-Bayes priors. Final forecast = w * main + (1 - w) * fallback, with w
  (`params.json` -> `blend.w_main`) chosen on validation.

## Training / params.json
`params.json` holds everything fitted from the round's public data: size vectors (from validation
targets), lag-pair prior, size-lag rule and cell priors (from stable train+validation machines).
Re-create it with:
```sh
python train.py --data /path/to/public --out params.json
```
(`--w-main` overrides the blend weight; default = the bundled value. Re-fit verified bit-identical.)
Validation (fitted on train only, official metric): see the team README.
