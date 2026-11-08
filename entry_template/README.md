# House Edge submission

## Recreate the forecast
```sh
python predict.py --data /data --out /output/submission.csv
```
Reads `/data/test/{queries,history}.csv` and the bundled `params.json`; writes only `--out`.

## Environment
- Python 3.13 (official `python` image), **standard library only** - no third-party dependencies.
- Fully deterministic: no random numbers, no seeds, no network. Sparse joint inference takes
  a few minutes on one core, within the official 30-minute runtime limit.

## Model
Structured Bayesian forecaster (`he/`):
- Win chance of a machine depends on whether its wins at two lags (A, B) agree (XNOR cell), with
  Beta-Binomial cell probabilities and type/signal-dependent hi/lo cell priors.
- Episode-level mixture: stable / drift (each machine's cell mapping flips at its own tick) /
  shared change (all machines flip at a common tick); lag pair and flip tick are marginalised exactly.
- Payout size given a win: two size vectors per machine type; the active one is
  (payout at lag L >= 2) XOR sign, with L coupled to the win pair; posterior-mixed.
- If `global_state` is present, joint hypotheses also let the payout-size regime
  shift the win logit. A shared binary Markov state pools history evidence across
  machines; its one-step predictive distribution is integrated at the cutoff.
  Peer effects are included inside this model, so the old post-hoc correction
  is not applied a second time. The frozen checkpoint is fitted offline.
- When `cfg.sparse_joint` is enabled, short histories use a joint hidden-state forward
  algorithm over win bits and big-payout bits. It uses the dominant learned size lag per
  pair, posterior-mean cell rates, and averages orientation/sign and constant-size hypotheses.
  This retains observations whose required lag values are missing. Dense histories use
  the original episode model. `cfg.sparse_win` enables the simpler win-only variant.
- Insurance blend (`he/fallback.py`): a generic Bayesian context model (any boolean function of up to two
  win lags, optional single change point; size given win conditioned on one big-payout lag), averaged over
  contexts with empirical-Bayes priors. Final forecast = w * main + (1 - w) * fallback, with w
  (`params.json` -> `blend.w_main`) chosen on validation.
- Cross-machine coupling (`he/coupling.py`, applied by `he/corr.py`): the win-probability logit is shifted by
  beta . (x - mu), x = (number of OTHER machines of the episode that won 3 ticks before the cutoff, recent net
  payout of the other machines). beta, mu come from a logistic regression on the per-tick outcomes of train +
  validation histories (offset = leave-one-out cell rate; candidates: other machines' wins at lags 1..8 and net
  payout; kept only |z| >= 3). Adopted after an out-of-sample check on validation targets (train-only fit: +0.095).

## Training / params.json
`params.json` holds everything fitted from the round's public data: size vectors (from validation
targets), lag-pair prior, size-lag rule and cell priors (from stable train+validation machines).
Re-create it with:
```sh
python train.py --data /path/to/public --out params.json
```
(`--w-main` overrides the blend weight; default = the bundled value. Re-fit verified bit-identical.)
For global-state entries this helper refits the original priors and preserves
the bundled offline global-state checkpoint; it does not rerun the EM fit.
Full checkpoint fitting scripts and validation details are in the team's repository.
