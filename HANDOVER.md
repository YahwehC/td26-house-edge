# House Edge — handover for another optimizer (Sat Nov 7, ~19:00 CT)

Goal: squeeze more points out of the TAMU Datathon 2026 "House Edge" forecasting entry. Read this first, then
`CLAUDE.md` (full lab notebook: every experiment with numbers) and `README.md` (how to run). Everything here is
reproducible from the repo plus the official kit.

## 1. The task in one paragraph
Slot-machine logs: episodes of 8 machines, each with a payout history (0/1/2/5/20 per tick) up to a `cutoff` tick.
Predict the probability vector of the machine's payout **at the cutoff tick** (`p_0, p_1, p_2, p_5, p_20`).
Score = squared error vs the simulator's **true** probabilities, mean per episode → mean per condition
(stable / short_history / drift / shared_change) → ratio to a pooled reference per condition → R = mean ratio →
points = max_points / (1 + R). Main round max 60, round two ("adaptation") max 40.
Deadlines (Central): main closes **Sat Nov 7, 9 PM**; round-two data drops then, closes **Sun Nov 8, 11 AM**.

## 2. Hard constraints
- Submission = test CSV + code ZIP (README.md + reproduce.json at zip root) that recreates the CSV. Our entry is
  **pure Python stdlib**, deterministic, runs on the official `python:3.13` image in ~10 s. Keep it that way.
- Kit data is not in git (`kit/` is gitignored). Get it from the kit download link in `GUIDE.txt` and extract it to
  `kit/` so that `kit/public/{train,validation,test}` exists. Do not run the kit's own scripts on untrusted data
  (our policy); our own `check_csv.py` and `build_entry.py` replace them.
- Validation is only 96 episodes (24 per condition) → **any knob tuned on validation targets overfits**. Lesson learned
  twice (main_v5): fit on per-tick outcomes of train histories, use validation targets only to check out-of-sample.
- Never submit on the user's behalf; the user uploads via Google Form.

## 3. Current best: main_v6 — validation 58.20 (out-of-sample), 58.23 with final params
| condition | ratio (main_v6) | ratio with oracle win prob | ratio with oracle size |
|---|---|---|---|
| stable | .0242 | .0009 | .0233 |
| drift | .0235 | .0012 | .0222 |
| shared_change | .0236 | .0029 | .0207 |
| short_history | .0504 | .0170 | .0342 |
Oracle win prob → 59.67 points; oracle size → 58.53. Reproduce: `.venv\Scripts\python.exe eda\ev40.py`.

## 4. What we know about the generator (recovered from data, see CLAUDE.md "Data facts")
- Win: P(win_t) = cell prob of XNOR(win_{t-A}, win_{t-B}); pair per machine ∈ {(1,3),(2,5),(4,7)} (uniform, not shared
  within an episode). Two cells per machine (hi/lo, extreme); type-level logit means, signal shifts logit ≈ +0.3/unit.
- Per-tick iid noise on the cell logit, sd ≈ 0.5 → this is the irreducible floor stable/drift/shared already sit at.
- **Cross-machine coupling**: +0.085 logit per OTHER machine of the episode that won at t−3 (z≈12), plus weak
  "heat" (others' net payout over 8 ticks, β≈0.015). Homogeneous across machine index, signal, hi/lo cells.
- drift: each machine's hi/lo mapping flips at its own tick (prob ≈0.9); shared_change: common flip tick. Flips
  32–104 ticks before cutoff. Size sign does NOT flip.
- Size given win: exactly 2 vectors per machine type; active vector = (payout_{t−A} ≥ 2) XOR s, s per machine,
  P(s=1) ≈ 0.45, random (not shared within episode, not predictable from anything).
- short_history: only 64 ticks observed (40 most recent + 24 random older ones).
- `sim/sim.py` re-implements this generator (configurable pairs / size lag / coupling lag) for stress tests.

## 5. Model (package `he/`, ~1.5k lines)
- `model.py` + `joint.py`: exact Bayesian episode mixture. Per machine: Beta-Binomial cell counts for every lag pair
  and every flip tick; episode-level mixture stable/drift/shared with flip tick marginalised exactly; size evidence
  for (lag L, sign s) coupled to the pair via P(L | pair); missing-lag wins marginalised.
- `fit.py`: all priors learned from data — pair prior (votes over all 28 pairs ≤ 8), cell priors (type × hi/lo:
  logit mean, signal slope, κ), size-lag rule (first/second/none), size vectors from validation targets.
- `fallback.py`: generic insurance model (BMA over any boolean function of ≤2 lags × optional change point).
  53.56 alone. Final forecast = 0.9·main + 0.1·fallback.
- `coupling.py` + `corr.py`: logistic regression on per-tick outcomes with offset = leave-one-out cell logit,
  candidates others' wins at lags 1..8 + others' net payout; keeps |z| ≥ 3; applied as a logit shift at the cutoff.
- `api.py`: `fit_params`, `predict_queries`, `write_submission`, `official_score`.

## 6. Already tried — do NOT repeat (all ≤ +0.02 or negative; details in CLAUDE.md)
Recency half-life; learned flip-time prior; HMM hot/cold state (residual autocorr ≈ 0); episode random effect on cell
levels (+0.01); common tick support (−0.3); joint hi/lo covariance / shared offset (+0.01); size-sign prior; soft
assignment of missing-lag ticks (−0.07); validation-fitted heat/multi-feature logit corrections (unstable across CV
splits); temperature scaling (+0.015, model is calibrated: prob-space slope 1.01); kappa / blend weight / coupling
scale tuned on 6000 pseudo-short episodes (all |Δ| ≤ 0.0003 Brier); residual screen of 45 features on validation in
prob space with episode-clustered SE (max |z| 3.3); coupling split by machine index / signal / payout size
(homogeneous); shared lag pair or size sign within episode (no); blending in the pooled reference (hurts).

## 7. Ideas not yet tried (ranked by my expected gain; all small)
1. **short_history size error** (ratio .050 vs .034 with oracle size; biggest excess). The sign s posterior from
   ~16 wins is weak. Any principled extra evidence for s or L in short episodes is the largest pot (~+0.2 max).
2. short_history win error (excess ≈ .011): better cell priors for few-tick machines, e.g. hierarchical prior with
   the hi/lo logit correlation (≈0.4) used jointly with the pair posterior (tested a simple version: +0.01).
3. Put the coupling inside the cell estimation (per-tick logit offset in the Beta-Binomial counts) instead of a
   post-hoc shift. Expected tiny (coupling sd ≈ 0.1 logit vs noise 0.5).
4. Own lag 2A effect: pair (2,5) shows own win at t−4 (+0.12 logit, replicated on train and validation, z 2.7/3.2);
   other pairs unclear (`eda/ev45.py`). Needs the pair posterior at the cutoff.
How to test cheaply without overfitting: per-tick outcome likelihood on train histories (`he/coupling.py`
pattern), pseudo-short episodes with Brier vs realised outcome (`eda/ev42.py` builds them, `eda/ev43.py` sweeps
variants), then a single out-of-sample check on validation targets.

## 8. Round two (probably worth more than further main gains)
"Adaptation changes which temporal relationships recur" + fresh train/validation/test. Run as soon as the kit drops:
```
.venv\Scripts\python.exe r2.py --data kit2\public --name r2_v0
```
~2 min: fits main + fallback on train, prints a main-vs-fallback weight table per condition on validation, picks the
blend weight, fits the coupling on train ticks and adopts it only if it gains out-of-sample, refits on train+val,
writes `runs\r2_v0\submission.csv`, checks it, builds and verifies `dist\r2_v0.zip`. Stress-tested on two synthetic
kits with changed pairs / size lag / coupling lag (`sim/sim.py`): adapts correctly. If the fallback beats main by a
lot, the rule family changed (e.g. 3-lag rules, non-XNOR functions, lags > 8 — `MAXLAG = 8` in `he/model.py`):
inspect `runs\r2_v0\params.json` votes and extend the model family; the fallback is the safety net meanwhile.

## 9. File map
`run.py` main pipeline · `r2.py` round-two one-shot · `build_entry.py` zip + verify · `check_csv.py` format check ·
`entry_template/` files that go into the zip (predict.py, train.py, reproduce.json, README.md) · `runs/main_vN/`
params + CSVs · `dist/` zips · `eda/` experiment scripts (evNN.py, numbered chronologically) · `hx/` experiment copy
of the model with feature flags · `sim/` synthetic generator · `research/` web-research briefs.
Env: `.venv` (Python 3.12, stdlib only needed); results are identical on 3.13.

## Follow-up optimization (Codex, Nov 7)
- Production speedup: skip change-point construction when the episode prior is stable-only;
  cache the consecutive identical hi/lo assignment and precompute Beta normalization constants.
- `eda/check_fast.py` compares against handover commit `d783b27` on all 768 validation queries:
  6.50 s -> 4.73 s (one local timing, ~27% less), bit-identical probabilities; score remains
  58.2275115 with existing final params. This is a runtime gain, not an accuracy gain.
- `dist/main_v6_fast.zip` reproduces `runs/main_v6/submission.csv` with max difference 0.
  Existing v6 params and CSV are unchanged. Packaging verification passed.
- Experimental `size_chain_missing` (off by default): forward marginalization along size-lag
  residue chains, using empirical win rate only for missing wins. Exact for that auxiliary
  model, approximate for the full XNOR process. Standard forward recursion reference:
  https://mc-stan.org/docs/2_28/stan-users-guide/hmms.html . Independently checked against
  exhaustive hidden-bit enumeration for lags 1, 2, 3 and both signs (`eda/test_size_chain.py`).
- Predeclared comparison on 6,000 pseudo-short queries: Brier 0.349617826 -> 0.349583772;
  paired delta -0.00003405, SE 0.00006279 clustered by the 30 ORIGINAL episodes (not the
  750 overlapping resamples); 19/30 improve. Insufficient evidence: NOT enabled and no
  validation sweep performed. Baseline priors are existing main_v6 final priors, not an
  independent cross-fit, so these are exploratory training diagnostics, not OOS gains.
- Reproduce: run `eda/ev42.py`, `eda/ev54.py`, `eda/test_size_chain.py`, `eda/check_fast.py`.
  Summary results are `eda/ev54_train.json` and `eda/check_fast_result.json`.

## Score-focused follow-up: main_v7 (Nov 7)

**Current recommendation: `runs/main_v7/submission.csv` + `dist/main_v7.zip`.**
Same fitted parameters and 0.9/0.1 blend as main_v6, plus `cfg.sparse_win=true`.
Validation score: 58.227511527 -> **58.245920527** (+0.018409).
Short-history ratio: 0.0504 -> 0.0491; short-history error 0.00895 -> 0.00872
(about 2.6% lower). Dense-history predictions are unchanged.

The improvement is structural, not a validation parameter sweep. Old WinMachine discarded
observed wins/losses when either required lag was missing. `he/sparse.py` integrates hidden
win bits with a finite-state forward pass, retaining all old observations. Each pair and
orientation keeps its Beta-binomial likelihood for fully observed contexts; a likelihood
ratio adds missing-context evidence without double-counting. Cell rates are posterior-mean
plug-ins, so this is approximate Bayesian inference despite an exact fixed-rate forward pass.
Initial hidden bits use a stationary distribution (256 iterations, tolerance 1e-11).

Frozen training diagnostic (`eda/ev56.py --full`): 6,000 pseudo-short queries, 30 original
source episodes: Brier 0.349617826 -> 0.348937847; delta -0.000679979, episode-clustered
SE 0.000407665, 18/30 improve. Then one validation comparison (`eda/ev56_validation.py`).
No parameters were selected using validation targets. As with ev54, the frozen baseline
priors include train+validation histories; this diagnostic is not fully cross-fitted.
The independent validation check supports the direction, but the gain is small and is not
proof of an improvement on private test data.

Checks: `eda/test_sparse.py` matches independent exhaustive enumeration of hidden paths
and the IID special case. `build_entry.py` reports zero prediction differences on Python
3.12 and 3.13. Packaged `train.py` reproduces params exactly, including the new flag.
No external dependencies in the submission. `eda/build_v7.py` rebuilds the candidate.

Rejected experiments (EDA only; SciPy is not packaged):
- `eda/ev55.py`: marginal-likelihood cell priors, three folds by source episode. Training
  delta -0.0000902 +/- SE 0.0001423. Single validation comparison worsens 58.23397 ->
  58.22699 when both cell-prior fits use train histories. Not adopted.
- `eda/ev57.py`: learn size noise/constant-regime/lag-leakage weights from training marginal
  likelihood. Three-fold training Brier worsens by 0.0001154 (SE 0.0001331). Not adopted;
  validation not inspected for this candidate.
- New forward-model methodology uses the standard HMM recursion documented at
  https://mc-stan.org/docs/2_28/stan-users-guide/hmms.html . No third-party code copied.

## Stronger score candidate: main_v8

**Supersedes v7: `runs/main_v8/submission.csv` + `dist/main_v8.zip`.**
The joint sparse-history model reaches **58.270735797** on the same validation set
(v6 58.227511527, v7 58.245920527). Short-history error is 0.00840 vs 0.00895 for v6,
about 6.1% lower; short ratio 0.0474 vs 0.0504. Other conditions are unchanged.

`he/sparse_joint.py` tracks binary wins for the win-pair lags plus big-payout bits for
the dominant learned size lag. Both missing wins and missing payout regimes are jointly
integrated. All four (hi/lo orientation, size sign) hypotheses are averaged, retaining
constant-size insurance. Posterior-mean cell-rate approximation remains; the tiny
non-dominant size-lag prior mass is dropped in this explicit candidate. There are no new
validation-tuned numeric knobs. Opt-in flag: `cfg.sparse_joint=true`.

Training comparison (`eda/ev58.py --full`): frozen candidate on 6,000 pseudo-short queries,
Brier 0.349617826 -> 0.348727375, delta -0.000890452, cluster SE 0.000428541, 20/30 source
episodes improve. Then a single validation check (`eda/ev58_validation.py`). Same caveat
about frozen baseline priors as ev56. No guarantee of private-test gain is implied.

The NumPy experiment is `hx/joint_sparse.py`; production was ported to pure stdlib.
`eda/test_sparse_joint.py` compares both on eight real queries (max difference about
2.1e-15), checks normalization, and verifies an independently calculated IID categorical
case. `eda/build_v8.py` rebuilds params/predictions from the original main_v6 params.

Final extra comparison: `eda/ev59.py --full` uses 3x3 Gauss-Jacobi quadrature to integrate
cell-rate uncertainty as well as hidden win bits. Training delta -0.00060359 (SE
0.00045692) is weaker than plug-in sparse wins and joint sparse inference, so it was
not promoted to a validation check. Best tested candidate remains main_v8.
The joint production recursion also passes independent exhaustive coupled hidden-path
enumeration for both size signs (not merely equivalence to the NumPy implementation).
Final v8 delivery checks: all 1,536 test rows pass our CSV checker; bundled `train.py`
refits parameters exactly. Production validation matches the NumPy pilot score exactly.
The package README documents the higher runtime of joint inference; the official limit
is 30 minutes / two CPU cores / 16 GiB RAM (`kit/RULES.md`).
Python 3.13 packaged prediction: VERIFIED, maximum difference 0.000e+00 versus the
Python 3.12 generated CSV. The final ZIP's README was refreshed afterward; every executable
byte was checked unchanged, all packaged source files match the workspace, and ZIP CRCs pass.
