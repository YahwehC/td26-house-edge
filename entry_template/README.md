# House Edge submission

For adaptation entry `r2_v2`, the lag pairs and size lags are fitted to the adaptation
kit. Global type/signal/size/peer logits pool released main and adaptation targets;
the reported adaptation score uses three folds that exclude the predicted adaptation
episode's targets. The final test checkpoint refits on all released targets.
Persistent offset variance 0.04 and main-model weight 1.0 were selected on adaptation
training histories. Exact sparse inference is not enabled in this entry.

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
- `global_state.machine_variance`, when present, enables a regularized persistent
  machine intercept estimated from its history. The shared-weather covariate used
  to estimate that intercept excludes the machine itself and uses earlier ticks.
- `cfg.exact_sparse_glm` integrates missing payout categories through a finite-state
  forward recursion under the global logits, including the machine intercept. This
  is enabled in main_v13. `cfg.marginalize_missing_peers` replaces unobserved peer
  wins by smoothed empirical rates; an observed zero payout is still a known loss.
  Forecast-time peer histories are observed. Weather during sparse recursion uses
  its prior mixture; the final prediction uses the history-filtered weather posterior.
- When `cfg.sparse_joint` is enabled, short histories use a joint hidden-state forward
  algorithm over win bits and big-payout bits. It uses the dominant learned size lag per
  pair, posterior-mean cell rates, and averages orientation/sign and constant-size hypotheses.
  This retains observations whose required lag values are missing. Dense histories use
  the original episode model. `cfg.sparse_win` enables the simpler win-only variant.
- Insurance blend (`he/fallback.py`): a generic Bayesian context model (any boolean function of up to two
  win lags, optional single change point; size given win conditioned on one big-payout lag), averaged over
  contexts with empirical-Bayes priors. Final forecast = w * main + (1 - w) * fallback, with w
  (`params.json` -> `blend.w_main`) stored in the checkpoint. If no blend is bundled,
  the main model is used alone. For main_v12, original-episode folds of training
  pseudo-short histories select weights 1.0, 0.9992, 1.0; the full training fit selects
  1.0. These are corrected episode-isolated estimates; validation is a later check.
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
