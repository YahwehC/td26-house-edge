# 002 - Generic fallback model `he/fallback.py` (round-2 insurance)

## Goal
A generic, assumption-light Bayesian predictor of the next-spin distribution that still works if round 2 changes the
hidden rule family (other lags, other boolean cell functions, different change mechanism). It will be blended with
the main model (`he/joint.py`) using a weight chosen on validation. **Pure Python standard library only**
(no numpy/pandas), deterministic, JSON-serializable params.

## File
Create only `he/fallback.py`. Do not modify any other file.

## Inputs (existing code, read but do not change)
- `he/model.py::load_split(public, split) -> (queries, extra)`. Each query dict has
  `query_id, episode_id, machine_id, machine_type (str), signal (int), cutoff (int)`,
  `seq`: dict tick(int) -> payout(int in {0,1,2,5,20}), only ticks < cutoff. Ticks may have gaps
  (short-history machines: 40 recent consecutive ticks + 24 sampled older ticks).
- Reuse `he/model.py` helpers: `logsumexp`, `betaln`, `SIZE_IDX = {1:0, 2:1, 5:2, 20:3}`, `MAXLAG = 8`.

## Public API
```python
FALLBACK_DEFAULTS = {"max_lag": 8, "win_kappa_grid": [0.5, 1.0, 2.0, 4.0], "size_alpha_grid": [1.0, 4.0, 16.0],
                     "change_prior": 0.5, "tau_min_back": 16, "tau_max_back": 128, "tau_step": 8,
                     "ctx_smooth": 0.5, "em_iters": 3}

def fit_fallback(query_lists, opts=None) -> dict
    """query_lists: list of query lists (e.g. [train_queries, val_queries]). Uses ONLY seq/machine_type
    (no targets). Returns JSON-serializable params (see below)."""

def predict_fallback(queries, fparams) -> dict
    """Returns {query_id: [p_0, p_1, p_2, p_5, p_20]}, each row sums to 1, every entry in [1e-9, 1]."""
```

## Model
### Win part (w_t = 1 if seq[t] > 0)
- Context sets C: `()`, `(L,)` for L=1..max_lag, `(A,B)` for 1<=A<B<=max_lag (37 sets for max_lag=8).
- For a tick t to be usable under C, t and all t-L (L in C) must be in seq. Cell key = tuple of w_{t-L}.
  Each cell has a full Beta prior Beta(kappa*m, kappa*(1-m)), m = type-level win rate (fit), so any boolean
  function of the context bits is representable (XNOR, XOR, AND, single lag, ...).
- Marginal likelihood of a table = sum over cells of `betaln(a+n1, b+n0) - betaln(a, b)`.
- Change point: hypothesis H0 (one table for all ticks) vs H_tau for tau in
  `range(cutoff - tau_max_back, cutoff - tau_min_back + 1, tau_step)` (skip tau where fewer than 8 usable ticks are
  on either side): independent tables for ticks < tau and >= tau. Prior: P(H0) = 1 - change_prior,
  each H_tau = change_prior / n_tau. If no tau valid, only H0.
  Predictive for the query = P(w=1 | cell at cutoff) from the active (post-tau, or whole for H0) table,
  posterior-weighted over hypotheses. Compute counts per tau incrementally (sort usable ticks; cumulative counts),
  not by recounting per tau.
- Cell at cutoff: bits w_{cutoff-L}. If cutoff-L is missing from seq, marginalize that bit with the machine's
  empirical win rate (Laplace-smoothed).
- Context prior pi(C): fit by empirical Bayes. Start uniform; for em_iters iterations, compute every fit machine's
  posterior over C (marginal over change hypotheses) under pi, set pi(C) proportional to
  `ctx_smooth + sum of posteriors`, renormalize.
- kappa: choose from win_kappa_grid the value maximizing the total log marginal likelihood (summed over fit machines,
  mixture over C and change hypotheses under the uniform C prior). Do this before the EM on pi.
- m per machine_type = pooled win rate over all ticks of fit machines of that type (unknown type at predict time
  -> global rate).

### Size part (given a win; size index k = SIZE_IDX[payout])
- Context D: `None` (no context) or L=1..max_lag with bit b = 1 if seq[t-L] >= 2 (t-L must be in seq; otherwise that
  tick is skipped for this D). For D=None every win tick is used.
- Each D cell (b=0, b=1; one cell for None) is Dirichlet-multinomial with prior alpha * mu_type
  (mu_type = pooled size distribution of wins of that type, Laplace +1). Log marginal likelihood via lgamma.
- alpha from size_alpha_grid by total log marginal likelihood; prior over D by the same EM as the win part.
- No change point in the size part. Predictive at cutoff: posterior over D; bit at cutoff from seq[cutoff-L]
  (missing -> marginalize with machine's empirical P(payout>=2), Laplace).
- Output row = [1 - pw, pw * s_1, pw * s_2, pw * s_5, pw * s_20], clip to [1e-9, 1], renormalize.

### Params (JSON)
`{"fallback_version": 1, "opts": {...}, "win_kappa": float, "size_alpha": float, "win_rate": {type: m, "_global": m},
"size_mu": {type: [4 floats], "_global": [...]}, "win_ctx_prior": {"": p, "3": p, "1,3": p, ...},
"size_ctx_prior": {"none": p, "1": p, ...}}` - context keys are comma-joined lags.

## Constraints
- Pure stdlib (`math`, `collections`, `json` ok). Deterministic (sorted iteration everywhere).
- Speed: `fit_fallback` on train+validation (1536 machines, ~250 ticks each) < 120 s; `predict_fallback` on test
  (1536 queries) < 60 s, single core. Use cumulative counts; avoid per-tau recounting.
- Docstrings/comments in English.

## Acceptance (run from `challenges/house-edge`, venv python `.venv\Scripts\python.exe`)
```
.venv\Scripts\python.exe -c "import time,sys; sys.path.insert(0,'.'); from he.model import load_split, official_score; from he.fallback import fit_fallback, predict_fallback; t=time.time(); tr,_=load_split('kit/public','train'); va,ex=load_split('kit/public','validation'); fp=fit_fallback([tr]); print('fit',round(time.time()-t,1)); t=time.time(); p=predict_fallback(va,fp); print('pred',round(time.time()-t,1)); print(official_score(p,ex,va))"
```
Must print a score dict with `points` > 45 (pooled reference = 30; main model = 57.9), every row summing to 1,
fit < 120 s, predict < 60 s. Also `json.dumps(fp)` must succeed.
