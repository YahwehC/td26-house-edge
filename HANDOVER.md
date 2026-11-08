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
