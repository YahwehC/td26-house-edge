# House Edge — handover for another optimizer (Sat Nov 7, ~19:00 CT)

> UPDATE: current best is **main_v12 (59.117106 episode-held-out calibration)**.
> See the final section. Earlier claims of an irreducible 58.2-58.3 ceiling are retracted.
> New submission pair: `runs/main_v12/submission.csv` and `dist/main_v12.zip`.

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

## IMPORTANT: earlier noise-floor conclusions were wrong (Nov 7, Codex)

**Current best: main_v10, 59.070745 / 60 in three-fold episode-held-out calibration.**
Use `runs/main_v10/submission.csv` with `dist/main_v10.zip` after verification below.
The previous 58.2-58.3 "information limit" statements in this document and CLAUDE.md
are retracted. In particular, a lack of marginal residual autocorrelation does NOT
establish iid hidden noise when payout regime and episode state are omitted.

Discovery source (public research, no code/checkpoint copied):
https://github.com/Apickle47/house-edge-datathon-2026/tree/main/submissions/main_peer_main
The public repo reports 59.0356 with held-episode calibration. It has no license, so
we independently implemented the modeling ideas and fitted our own coefficients.
The user's reported 59.88 has not been independently located or verified.

Missing mechanisms:
1. The payout-size regime also changes WIN probability. Train-history leave-one-out
   offset regressions (`eda/ev60.py`) give logit coefficients steady -0.545 (z -10.8),
   balanced -0.792 (z -16.4), jackpot -1.021 (z -14.8). Effects are not copied from
   another model. The old win/size conditional-independence assumption lost this signal.
2. There is a shared binary, persistent episode effect. A forward-backward EM fit on
   TRAIN history outcomes (`eda/ev63.py`, `eda/ev64.py`) estimates amplitude 0.3562 and
   switching probability 0.02. Previous tests failed to isolate this after size effects.

Progression (same validation set; not independent repeated test sets):
- v8: 58.270736.
- Size/win post-hoc correction: 58.704185 (`ev61`).
- Add train-fitted shared-state filter: 58.811322 (`ev62`).
- Joint global-state structural likelihood: **v9 59.052226** (`ev66`).
- A fixed interpretable calibration (11 coefficients + episode shift amplitude),
  fit on two thirds of released validation target episodes and predicting the remaining
  third: **v10 59.070745** (`ev68`). All three folds improve over v9: +0.0290, +0.0068,
  +0.0241 points when each fold is scored separately. Final test checkpoint refits this
  same calibration on all released validation episodes; reported validation predictions
  remain the out-of-fold predictions, not predictions from the final checkpoint.

v10 ratios: stable .0086, short .0322, drift .0136, shared .0086.
Gain over v8 +0.800009; descriptive episode bootstrap 95% interval [+0.5685,+1.0397].
This public set has been used repeatedly during development; this is not an independent
private-test guarantee. Existing baseline priors still use released train+validation
histories and the size vectors come from released targets, as before. New supervised
logit calibration excludes the predicted episode's targets.

Implementation: `he/structured_state.py` enumerates pair/orientation/size-sign/flip
hypotheses with joint payout likelihoods. The common weather posterior comes from
history-only emissions; marginalize its one-step predictive distribution at cutoff.
`he/api.py` selects this model when `params.global_state` exists; no second post-hoc
coupling shift is applied. Generic fallback weight remains the original 0.1.
`he/weather.py` contains the standard-library binary filter; `he/global_state.py`
extracts history states. No third-party dependencies enter the prediction package.

Training reproduction: `eda/ev63.py` -> `eda/ev64.py` -> `eda/ev68.py` ->
`eda/build_v10.py`. Offline fitting uses NumPy/SciPy. Packaged `train.py` refits the old
priors but PRESERVES the bundled global-state checkpoint and its provenance; it is not
an end-to-end reimplementation of the offline EM/supervised calibration. `predict.py`
only needs test histories and the frozen checkpoint to reproduce the submission.

Rejected new variant: exact sparse latent-category recursion under the global model
(`ev67`) scored 59.02947, below v9; not enabled. Its missing-peer marginalization remains
approximate. Do not silently enable `exact_sparse_glm` for main or adaptation.

Checks: `eda/test_weather.py` independently enumerates hidden-state paths, including
irregular observation gaps and prediction horizon. `eda/test_global_state.py` checks
normalization and verifies that adding future payouts to input files leaves predictions
bit-identical. v9 ZIP reproduced on Python 3.13 with max absolute difference 0.

Final main_v10 delivery verification: 1,536 test rows pass the format checker. The ZIP
was extracted and run under Python 3.13; all probabilities reproduce the Python 3.12
CSV with maximum difference **0.000e+00**. No competition submission was made.
Packaged `train.py` verification also passed: refitted base priors plus preserved
checkpoint/provenance match the full v10 parameter JSON exactly.

## Further mechanism search: persistent machine offsets (main_v11)

New candidate **main_v11: 59.097438551**, versus v10 59.070745008 (+0.026694).
Matched files: `runs/main_v11/submission.csv` and `dist/main_v11.zip`.
It retains the same three-fold episode-held-out calibration predictions and adds a
zero-centered, history-estimated machine intercept with prior variance 0.04.
`he/machine_offset.py` estimates each offset using that machine's outcomes; the weather
covariate uses OTHER machines' strictly earlier outcomes, excluding current/future ticks.
The offset is applied consistently in joint hypothesis likelihoods, weather emissions,
and forecast probabilities. No new validation-fitted scalar is used.

Training selection (`eda/ev77.py`): stable train machines, first 60% of ticks for offset
estimation and remaining 40% for outcome scoring. Variance 0/.01/.04/.09/.16 yielded
Brier .08411245/.08398851/.08384424/.08383403/.08389788. Select .04 as the smaller
variance within one SE of the minimum (.09); paired episode delta -.0002682, SE .0000903.
Caveat: the latent structural state labels/global coefficient initialization were
inferred from full training histories; this is a chronological check of the intercept
component, not a wholly future-blind end-to-end cross-fit. No validation targets enter
this variance selection. The subsequent held-episode calibration check is ev78.

Condition tradeoffs: stable ratio .0086 -> .0070; drift .0136 -> .0125;
short .0322 -> .0329; shared .0086 -> .0087. Keep v10 as fallback; the public-set gain
is small and no private-test gain is guaranteed.

Other newly tested rules:
- **Drift reversals are gradual; shared changes abrupt** (ev71). A linear transition
  with half-width 8 ticks improves TRAIN marginal log-evidence by +173.4 across 192
  drift machines, but worsens shared-change evidence by -146.2. Abrupt/ramp use equal
  onset-time support. The pilot uses inferred pair/sign states, not ground-truth labels.
  Enabling a ramp only for drift in the episode mixture scores 59.06902, below v10;
  `global_state.drift_ramp` therefore remains absent in shipped params.
- A released-validation residual screen of own/peer lags, payouts and interactions (ev70)
  has max episode-clustered score |z| 2.44, own net over 8 ticks. No feature adopted.
- Minimum-duration weather regimes (ev73) improve likelihood by only ~1.05 versus the
  existing memoryless filter; insufficient evidence to add complexity, not adopted.
- Integrating orientation/size uncertainty in weather emissions (ev74) scores 59.07115
  (+0.00040), effectively neutral. Joint-likelihood MAP weather states (ev75) score
  59.06466; not adopted. Both remain explicit opt-in experimental flags only.
- Oracle state diagnosis (ev76), NOT a deployable result: knowing high/low, size, and
  weather states yields 59.7729 under the current global coefficients. Residual logit
  RMS ~0.197 motivates persistent machine variation. This is NOT an upper bound.

Rebuild v11: `eda/build_v11.py` from the fixed main_v10 checkpoint and ev78 OOF report.
Checks: `eda/test_machine_offset.py` verifies the regularized intercept's score equation,
zero-variance behavior, and weather probability/eta emission equivalence;
`eda/test_weather.py` independently enumerates HMM paths. `eda/test_global_state.py
runs/main_v11/params.json` checks normalization and exact future-tick invariance.

main_v11 final verification: all 1,536 test rows pass the checker. Extracted ZIP rerun
under Python 3.13 reproduces the Python 3.12 CSV with maximum absolute difference 0.
Independent drift-path test confirms the optional ramp does not affect shared-change
likelihoods. Neither ramp, soft-weather, nor joint-weather experimental flags are enabled
in main_v11. The only new enabled feature is machine_variance=0.04.

## Continuing search after v11 (ev79-81)
- Episode-specific weather amplitude/hazard mixtures (ev79), fitted on other original
  episode folds, evaluated on last 40% of held train histories: amplitude-only mean
  log-score gain +.0344, cluster SE .0372; amplitude+hazard -.00048, SE .0452. Neither
  supports a reliable improvement; no validation sweep or production change.
- Separate machine offsets for low/high states (ev80) worsen chronological held-out
  training Brier vs a common offset: independent +.0000487 (SE .0000261), correlated
  rho=.5 +.0000171 (SE .0000131). Keep the shared machine intercept.
- Train-fitted global high-minus-low logit gaps are 4.7626/4.8289/4.7346 across the three
  types. Testing a common gap as a structural constraint, removing two free coefficients,
  with the same episode-held-out calibration protocol (`eda/ev81.py`).
These chronological diagnostics use latent structure inferred from full training
histories, as in ev77; they are component checks, not fully causal end-to-end forecasts.
Common-gap result: 59.09533271 versus v11 59.09743855, so the constraint is not adopted.
It remains a generator-structure clue, not a demonstrated score improvement.

## Continuing mechanism search and main_v12 (ev82-88)

Current best candidate: **main_v12, 59.117106224**, up from v11 59.097438551.
Matched artifacts: `runs/main_v12/submission.csv` and `dist/main_v12.zip`.
The new global-state model no longer benefits from the legacy 10% generic fallback.
`eda/ev85.py` fits one convex blend weight on two thirds of ORIGINAL training source
episodes and evaluates the remaining third. All three folds choose main weight 1.0.
On 6,000 pseudo-short queries / 30 source episodes, Brier delta versus weight .9 is
-.00051450 (episode-cluster SE .00022610), with 22/30 episodes improving.
This is an out-of-fold check of the BLEND choice; the frozen base priors/checkpoint
still use previously documented public-data fitting. It is not a fully independent
end-to-end training cross-fit.

One subsequent held-episode calibration evaluation (`eda/ev87.py`) gives 59.117106.
Ratios: stable .0062, drift .0112, shared .0073, short .0350. Short history worsens
versus v11 (.0329), despite the training pseudo-short improvement. Retain v11 as a
fallback artifact; the small public-score improvement does not establish private-test
superiority. v12 changes only the final blend, with no new validation-fitted scalar.
The default absence of `blend` and `fallback` means pure main prediction.

Verification: all 1,536 test rows pass format checking. ZIP extraction and inference
under Python 3.13 exactly reproduce the CSV (max difference 0). Packaged `train.py`
refits the old priors, preserves the offline global-state checkpoint and metadata,
and produces exactly the same v12 parameter JSON. Future-tick exclusion remains
bit-identical (`eda/test_global_state.py`). Rebuild with `eda/build_v12.py` and
`build_entry.py` using the usual paths.

Other tested mechanisms, all DISABLED in v12:
- ev82: asymmetric hot/cold switch hazards and a previous-tick win-rate trigger do
  not improve episode-held-out train tail likelihood. Mean gains -.01396 and -.01395
  respectively (SE about .01297). Re-estimating one symmetric hazard also loses.
- ev83: missing peer observations formerly contribute zero wins. Replacing missing
  values by smoothed empirical peer win rates yields pseudo-short Brier delta
  +.00005778 (SE .00006435); no validation test, not enabled.
- ev84/86: other-machine-only weather smoothing in historical hypothesis likelihoods
  improves pseudo-short Brier -.00021097 (SE .00015842), but the single held-episode
  validation check is 59.092612 versus v11 59.097439. Not enabled. The forward-backward
  calculation matches exhaustive weather-path enumeration (`test_weather_history.py`).
- ev88: jointly integrate each short-history pair/sign hypothesis, its persistent
  machine intercept (five-point normal quadrature, fixed variance .04), and binary
  weather path, conditional on fixed other-machine emissions. Missing own lags retain
  empirical marginalization. Pseudo-short Brier worsens +.00076024 (SE .00041083),
  only 9/30 source episodes improve. No validation test, not enabled. Independent
  path enumeration and normal moments through degree eight pass.

Correction to the earlier residual-screen description: `eda/ev70.py` uses released
validation target residuals, not train-history outcomes. Its maximum cluster z=2.44
was a diagnostic of that public set and supplied no new production coefficient.

### Further train-only screens (ev89-93)
- ev89 uses 129,916 train-history rows with episode-fold coefficient fitting and
  final-40%-of-history scoring. Additional signal nonlinearity/type/state interactions
  worsen Brier by +.00002761 (SE .00001320); size-state interactions are neutral
  -.00000212 (SE .00000744); nonlinear/type/state-specific peer coupling is neutral
  +.00000547 (SE .00001383). No validation sweep and none enabled.
- ev90 compares binary weather against a continuous AR(1) latent process, using
  21 discretization states and episode-held-out tail likelihood. AR persistence fits
  rho .957-.964 with scale .308-.334, but loses mean log-score .0490 (SE .0590)
  versus the fixed binary model. Even refitting binary amplitude/hazard loses .0190
  (SE .0173); retain the current weather model. No validation test.
- ev91 adds own win lags 1-8 or extra peer win lags 1,2,4-8 plus 8-tick peer net
  payout after controlling for the current state/size/weather model. Held-episode
  tail Brier worsens +.00001594 (SE .00001403) and +.00003202 (SE .00001356).
  No extra lag coefficient enters production.
- ev92 uses smoothed OTHER-machine weather within the observed prefix to estimate
  machine intercepts. Stable-train prefix/tail Brier improves -.00002685 (SE .00001699).
  The single held-episode calibration check ev93 gives 59.115211, below v12 59.117106;
  `global_state.offset_smoothing` remains disabled. Independent exhaustive weather-path
  enumeration checks the resulting regularized intercept's score equation.
All these component checks freeze structural labels inferred from full training
histories; this limitation remains. No new best submission after main_v12.
