# Research brief - House Edge (TAMU Datathon 2026): find ideas to lower our error

## Problem
Slot-machine logs. Each episode = 8 machines (types e.g. "balanced"; each machine has an integer `signal`),
history of payouts per tick in {0,1,2,5,20}. For each machine predict the probability vector of the NEXT spin
(at its cutoff tick, 192-256). Score = squared error vs the simulator's TRUE probability vector, averaged per
episode, then per condition (stable / short_history (only 64 observed ticks: 40 recent + 24 sampled older) /
drift (each machine's rule flips at its own tick) / shared_change (all machines flip at one common tick)),
divided by a frozen pooled baseline's error per condition; R = mean ratio; points = 60/(1+R).
Round 2 (14 h overnight) gives a NEW data kit where "which temporal relationships recur" may change.

## What we recovered (generator)
- P(win_t) = cell probability of XNOR(win_{t-A}, win_{t-B}), lag pair (A,B) in {(1,3),(2,5),(4,7)} per machine.
  The two cell probs are extreme (one high, one low); type-level means on the logit scale; signal shifts logit
  by ~+0.3 per unit. Residual logit noise sd ~0.5 between machines (hidden), correlated within episode (r~0.3).
- drift / shared_change: hi/lo cells swap at a flip tick 32-104 ticks before cutoff.
- Size given win: two fixed size vectors per type; active vector = (payout_{t-L} >= 2) XOR s, L = first lag A.

## Our model (validation 57.96 / 60, R = 0.035; per-condition ratios stable 0.027, short 0.061, drift 0.029,
shared 0.025)
Exact Bayesian mixture per episode: {stable, drift, shared} x per-machine lag pair x flip tick, Beta-Binomial
cells with a 2-component (hi/lo) empirical-Bayes Beta prior, size regime coupled to the pair; blended 0.9/0.1
with a generic context-tree-like fallback (BMA over contexts of <=2 lags x one change point).
Tried with no gain: others'-wins logit term, extra lags, recency half-life, learned flip-time prior,
EM P(L|pair), HMM hot/cold state (residual autocorrelation ~0), two-pass episode pooling of cell levels
(<= +0.01), blending with the pooled baseline (hurts).

## What we want from you
1. Similar public problems / competitions / papers / repos: forecasting with a known simulator family and
   squared error vs true probabilities (proper scoring, Brier), Bayesian change-point (BOCPD) with shared change
   points across series, hierarchical / multi-task Beta-Binomial, context-tree weighting, sequence prediction
   in Kaggle simulation competitions (RPS, Santa), slot-machine / RNG pattern detection, "structure discovery"
   for unknown Markov rules, test-time adaptation when the generator changes (round 2).
2. Concrete, testable ideas that could lower error further, especially for short_history (worst ratio) and for
   adapting quickly if round 2 changes the rule family (e.g. 3-lag rules, other boolean functions, other change
   mechanisms). Rank by expected gain / effort. For each: 2-4 lines, what to change, why it should help.
3. Links (URLs) to sources. Say clearly what is from a real source vs your own reasoning.
Write the answer in English, <= 60 lines.
