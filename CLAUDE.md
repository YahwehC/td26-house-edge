# House Edge (TAMU Datathon 2026, challenge 4) — session notes

Answer the user in Chinese. Code/specs in English. Own venv: `.venv` (uv-installed numpy/pandas/scipy/sklearn — EDA only).
Final predictor is **pure stdlib** (official runner: Python 3.13 stdlib image, no network). Verified identical under
uv's Python 3.13.14.

## Rules (from GUIDE.txt + kit/RULES.md)
- Predict next-spin probs [0,1,2,5,20] per query; CSV `query_id,p_0,p_1,p_2,p_5,p_20`.
- Score: squared error vs simulator TRUE probs; mean within episode → mean per condition (stable, short_history,
  drift, shared_change) → ratio to frozen pooled baseline per condition → R = mean ratio; points = max/(1+R).
  Kit RULES.md: main 60 / adaptation 40 (GUIDE.txt says 80/20; metadata.json max_points=60 → trust kit).
- Main closes **Sat Nov 7 9 PM CT**; round 2 ("adaptation") Sat 9 PM → **Sun 11 AM CT**, new kit, separate CSV+ZIP.
- Submit CSV + code ZIP (README.md + reproduce.json at zip root). Main submission form: GUIDE.txt line 11
  (Google Form; team registration form line 9 is mandatory). **User submits** (needs their account/team).
- Kit's own scripts can't be run by Claude (auto-mode blocks downloaded code) → metric reimplemented in
  `he/model.py::official_score`; our `check_csv.py` replaces `check_submission.py`.

## Data facts (main round) — recovered generator
- train/validation: 96 episodes × 8 machines; test 192. Cutoffs 192–256. Short history: 64 ticks (40 recent + 24 old).
- Win: P(win_t) = cell prob of XNOR(win_{t-A}, win_{t-B}); pairs ∈ {(1,3),(2,5),(4,7)}. Cells extreme (hi/lo);
  type-level hi/lo logit means, signal shifts logit ≈ +0.3/unit. Some machines absorbed (all-loss / all-win).
- drift: each machine's hi/lo mapping flips at its own tick; shared_change: common flip tick. Flips 32–104 ticks before cutoff.
- Size given win: exactly 2 vectors per type; regime = (payout_{t-L} ≥ 2) XOR s with **L = first lag A of the win pair**.
- Residual win-logit noise sd ≈0.5: others' wins, extra lags, recency weighting all tested → no gain (hidden state).
- Test episodes classify 48/48/48/48 (stable/drift/shared/short) with >95% posterior → same generator as validation.

## Model (`he/`)
- `model.py`: io, `WinMachine` (per pair × flip-tick Beta-Binomial sufficient stats, hi/lo mixture cell prior),
  `episode_posterior` (stable / drift / shared mixture; short-history episodes → stable-only prior), scoring.
- `joint.py`: `predict_joint` — pair prior × size-evidence factor via P(L|pair); size regime mixed by pair posterior.
- `fit.py`: data-driven priors: pair prior (votes of stable machines over all 28 pairs), size-lag votes,
  cell prior (type × hi/lo: logit mean, signal slope, κ by moments), lag rule (first/second/none by likelihood).
- `api.py`: `fit_params(public, fit_splits)`, `predict_split`, `write_submission`, `save/load_params`, `score_validation`.
- Pipeline: `run.py` (fit train → score val → refit train+val → predict test → check), `build_entry.py` (zip + rerun verify),
  `entry_template/` (predict.py, train.py, reproduce.json, README.md). `hx/` = scratch copy for experiments.
- `fallback.py`: generic insurance model - BMA over win contexts (any boolean fn of <=2 lags in 1..8) x optional single
  change point, size given win conditioned on one big-payout lag; EB priors; recovers the 3 pairs from scratch.
  `api.predict_queries` blends `w*main + (1-w)*fallback` via `params["blend"]["w_main"]` (absent -> main only).
- `r2.py --data <public> --name <n> [--max-points 40]`: one shot (~90 s): fit main+fallback on train, weight table on
  validation, pick w*, refit train+val, test CSV + check, build+verify `dist/<n>.zip`, `runs/<n>/r2_report.json`.
  If main crashes it continues with fallback only (w=0). Entry `train.py` re-fits blended params bit-identically.

## Validation log (validation split, official metric, main max 60; priors fitted on train only)
| version | points | R | stable | short | drift | shared |
|---|---|---|---|---|---|---|
| pooled reference | 30.0 | 1 | 1 | 1 | 1 | 1 |
| v1 per-machine flip prior (bug: hedging) | 52.10 | 0.152 | 0.418 | 0.114 | 0.041 | 0.034 |
| v2 episode-level mixture | 56.80 | 0.056 | 0.039 | 0.112 | 0.041 | 0.034 |
| + hi/lo type cell priors | 57.20 | 0.049 | 0.036 | 0.093 | 0.035 | 0.033 |
| + learned structure priors, short→stable | 57.37 | 0.046 | 0.035 | 0.082 | 0.034 | 0.033 |
| + joint pair↔size-lag (rule learned, δ=0.01) = main_v2 | 57.90 | 0.036 | 0.029 | 0.064 | 0.028 | 0.025 |
| generic fallback alone (`he/fallback.py`, no generator knowledge) | 53.56 | 0.120 | 0.064 | 0.193 | 0.121 | 0.103 |
| **0.9·main_v2 + 0.1·fallback = main_v3** (w by validation; 0.8–0.95 all ≥57.89) | **57.96** | 0.035 | 0.027 | 0.061 | 0.029 | 0.025 |
Tried, no gain: others'-wins logit adjustment, additive lag effects, recency half-life, learned flip-time prior, EM P(L|pair).

## Prior-art research (Sat ~13:00, read-only; WebSearch backend was down → GitHub API/curl)
- No public simulator/generator anywhere; tamu-datathon-org has no House Edge repo. https://github.com/longv1-code/TD27-House-Edge
  = re-upload of the same kit (no generator). Team repos https://github.com/alexnstevens06/house_edge_datathon_2026 and
  https://github.com/sreenivasulureddy-tamu/House-edge-tamu-datathon were empty → maybe re-check before round 2.
- BOCPD (Adams & MacKay 2007, https://arxiv.org/abs/0710.3742; impl. https://github.com/gwgundersen/bocd) — same math as
  our exact flip-tick mixture; our flip-time prior is already tested (learned histogram = neutral).
- Empirical-Bayes Dirichlet pooling (Minka; https://github.com/ericsuh/dirichlet) — we already fit type-level hi/lo Beta
  cell priors by moments; size mix is 2 discrete vectors, not Dirichlet → no gain expected.
- Context tree weighting / variable-order Markov (https://github.com/gabeschamberg/context-tree-weighting) — weaker
  generic prior in-family; **keep as round-2 insurance** if new rules fall outside the lag-pair XNOR family
  (low-weight fallback component, chosen on new validation).
- Squared error vs true probs: posterior predictive mean is optimal → no tempering/sharpening; gains only from priors.
- Conclusion: nothing found that replaces our model; no reusable library worth adopting for main.
- 2nd sweep (Sat ~13:30, DuckDuckGo html via browser + GitHub pages; WebSearch tool broken): Reddit/Instagram/TikTok/X
  only have event promo posts; no House Edge discussion. GitHub repos created since Nov 6 with house-edge/datathon names:
  the 2 House Edge team repos still empty, others unrelated. Re-check before round 2 is optional.
- Insurance test: blending ε·pooled reference into our forecast: ε=0.01 → 57.897, 0.03 → 57.85, 0.10 → 57.37
  (hurts in-family). Use only in round 2 if new validation shows the rule family no longer fits.
- 3rd sweep: analogous projects (not datathon-specific). Kaggle Rock-Paper-Scissors (top bots = ensembles of lag-pattern
  predictors, e.g. https://github.com/abdelghanibelgaid/RPS; solution index https://github.com/faridrashidi/kaggle-solutions)
  ≈ our exact BMA over lag rules. Kaggle Santa 2020 bandit (exact posterior tracking of decaying win rates,
  https://github.com/toshi-k/kaggle-santa-2020, https://www.kaggle.com/code/somnambwl/posteriordistribution)
  ≈ our change-point mixture. BOCPD libs https://github.com/BayesOnBikes/bocd. HMM regime switching → tested below.
  - Hidden-Markov "hot/cold" state: residual autocorrelation within stable machines ≈0 at gaps 1..20 (n≈80k) → no HMM gain
    (`eda/hmm_check.py`).
  - Episode random effect on cell levels: real (corr own vs mates' residual 0.31 hi / 0.22 lo, `eda/ep_effect.py`), but
    two-pass pooling (`hx/pool.py`, `eda/pool_test.py`) gives ≤ +0.01 points → not adopted.

- 4th sweep (Sat ~16:15; Claude via DuckDuckGo + Codex + DeepSeek, briefs/answers in `research/`): new team repo
  https://github.com/phillip-122/datathon-2026 = kit copy + starter outputs; kit README names the organizers' reference
  repo https://github.com/tamu-datathon-org/slots-rl → **404 (private)**. Literature pointers: BASIC shared change
  points across sequences https://arxiv.org/abs/1508.01280, Stan pooling case study
  https://mc-stan.org/users/documentation/case-studies/pool-binary-trials.html, Bayesian context trees
  https://arxiv.org/abs/2007.14900. Ideas tested on validation (all no gain, `eda/ev19-21.py`, flags in `hx/`):
  common tick support for pair/lag likelihoods 57.73 / size-only 57.85 (drops data); plug-in fill of unequal supports
  57.88; joint E[w·s] over pairs 57.903 (= baseline); hi-cell orientation: P(XNOR=1 is hi)=0.53, within-episode
  agreement 0.508 vs 0.502 null, no type/slot/signal effect → random; slot offset on true logits not significant
  (χ²≈12/7 dof); transductive priors (fit on train+val histories) 57.906 (+0.003). ⇒ main model at its ceiling;
  remaining error ≈ irreducible cell-level uncertainty. Round-2 value: rule-family zoo (3-lag, pair switch at τ,
  other size predicates) only if the new kit's validation shows main < fallback.

## Status
- [x] **Main entry ready**: `runs/main_v2/submission.csv` + `dist/main_v2.zip` (VERIFIED rerun, identical on Py3.13).
- [x] main_v3 (blend, +0.06 val): `runs/main_v3/submission.csv` + `dist/main_v3.zip` (VERIFIED, identical on Py3.13).
- [x] Round-2 runner `r2.py` + fallback ready; dry-run on main kit: exit 0, 88 s.
- [ ] User: register team (form) and upload main CSV + ZIP before Sat 9 PM CT.
- [ ] Round 2 at 9 PM (plan below).

## Round-2 plan (Sat 9 PM → Sun 11 AM)
1. Download new kit → `kit2/` (separate folder; check `public/metadata.json` round = adaptation).
2. `.venv\Scripts\python.exe r2.py --data kit2\public --name r2_v0` (max points default 40) → read the weight table
   (main vs fallback per condition) → user uploads `runs\r2_v0\submission.csv` + `dist\r2_v0.zip` as safe first entry.
   If fallback >> main, the rule family changed → spend the time on step 3.
3. Check what changed ("which temporal relationships recur"): `params.json` votes (pairs, lags, lag_rule), episode
   classification counts, size vectors (2 per type still?), cell priors. Re-run EDA scripts (eda/eda3-4, eda9-10, ev15)
   on kit2 if votes look diffuse — the rule family itself may change (e.g. new pairs/triples, category-based cells).
4. Adapt model only where validation says so; rebuild + upload; keep receipt.
