# House Edge — research answer (TAMU Datathon 2026)

Scope: [S] = real source (linked); [R] = my reasoning from he/model.py, he/joint.py, he/fallback.py. I ran no new experiments: this session's shell sandbox denied workspace writes, so every idea includes its validation test.

## 1. Closest public work (real)
- BOCPD (run-length posterior, one change point, predictive): Adams & MacKay 2007, https://arxiv.org/abs/0710.3742 — the same object as our tau sweep.
- Context-tree weighting / Bayesian context trees (exact prior predictive over variable-memory Markov models, linear time): https://doi.org/10.1109/18.382012, https://arxiv.org/abs/2007.14900 — a principled version of fallback.py and a selector for unknown rule families.
- Hierarchical Beta-Binomial pooling: Gelman et al., BDA3 ch. 5, https://sites.stat.columbia.edu/gelman/book/BDA3.pdf — how to pool cell rates across machines/types without over-shrinking.
- Proper scoring: Gneiting & Raftery 2007, https://doi.org/10.1198/016214506000001437 — posterior mean is optimal for squared error; no calibration/temperature trick can help.
- Analogues that validate our design: Kaggle RPS lag-pattern ensembles https://www.kaggle.com/competitions/rock-paper-scissors ; Santa 2020 decaying bandit posteriors https://www.kaggle.com/c/santa-2020 ; TTA under shift (use unlabeled test data) https://arxiv.org/abs/2303.15361.
- No public House Edge simulator/solution: https://github.com/tamu-datathon-org has no such repo, searches return only the kit, and gaming RNG literature is regulatory, not predictive. Our generator notes are the only source.

## 2. Ranked, testable ideas (gain = expected points/60; all gains are small, 0–0.3)
1. [R] Round-2 family zoo + evidence selection (best strategic value, zero in-family gain). Add 3-lag contexts and a "cell-swap" change model to fallback.py; score {main, 2-lag, 3-lag, swap} by total marginal likelihood on the new validation (labels if present, else unlabeled evidence).
   Why: a changed rule family is the only failure mode that can cost >1 point; exact evidence penalizes complexity (BCT/CTW), and fallback already nests arbitrary functions of its context bits.
   Effort M; test: leave-one-condition-out on main validation (fallback alone scores 53.56, so weight it only when it wins).
2. [R] Transductive empirical Bayes on the new test kit (robustness). Run fit.py's pair/size votes and cell-prior moments on the new test histories only, compare with train votes, and refit priors on test∪train if they disagree.
   Why: 192 unlabeled test episodes (main kit) estimate type-level hyper-parameters better than 96 train ones, and this adapts before any label is seen; the rules allow it because prediction already reads test histories.
   Effort S; test: on main, refit priors using validation inputs and score validation — must not lose >0.02.
3. [R] short_history: predict E[w·s_k], not E[w]·E[s_k]. joint.py:188–190 multiplies the marginal win mean by the marginal size mix, but both depend on pair/tau, so Cov(w,r1) is dropped.
   Why: the target is E[p_k] = E[w·s_k]; the covariance is largest exactly where the pair/tau posterior is spread (short history, near-flip), our worst condition (0.061).
   Effort S; test: expose per-pair/per-tau win preds, average the product over the same posterior components; require a gain > bootstrap noise.
4. [R] short_history: test shared hi/lo orientation. assign() in model.py fixes P(cell1 hi) at a 0.5 prior per machine; if orientation is correlated within an episode, add an episode-level orientation latent and reweight phi.
   Why: short-history error is dominated by which XNOR cell is high, not only by cell levels (the tested two-pass level pooling gave ≤0.01); check same-type pairs within an episode.
   Effort S to test (pairwise corr of 2·p1hi−1 vs a machine-permutation null), M to use; gain 0.1–0.3 on short if real, 0 otherwise.
5. [R] short_history: forward-filter missing lag bits. State = last 7 win bits (128 states); transition = shift + new bit drawn from the pair's two cell rates; score the ~15–25 ticks WinMachine drops because a lag is unobserved, then EM-update cell rates (forward algorithm: https://doi.org/10.1109/5.18626).
   Why: dropped ticks are still informative through the Markov bit posterior; dropping them is unbiased but wastes data precisely where we have least.
   Effort M; gain maybe 0.05–0.1 on short; test on the 48 short validation episodes only, with plug-in rates first.
6. [R] change-mechanism zoo (round-2 insurance). Current drift/shared only swap hi/lo of one fixed pair. Add (a) pair-switch at tau (pre counts under p, post under q, 3×3), (b) gradual change (cell logit moves over 8/16/32 ticks via logistic-weighted counts), (c) hierarchical tau (mixture of a latent common tau and a flat prior).
   Why: round 2 may change the temporal relationship itself or soften the flip; the current model cannot represent either, and a hard shared/drift choice can be wrong at the boundary.
   Effort M–L; zero in-family gain; gate on whether new validation prefers any variant.
7. [R] Size: predicate library + BMA. Regime is big(payout_{t−A}) XOR s; add big at the other pair lag B, win at A, and XOR of big at A and big at B, with type vectors and evidence weights.
   Why: a changed size rule is plausible in round 2; EM P(L|pair) was neutral in-family, so expect insurance only.
   Effort S–M; test: does the learned predicate posterior concentrate or stay flat on validation?
8. [R] Per-slot random effect (cheap, untested axis). machine_id recurs in every episode's 8 rows; fit a per-(type, machine_id) cell-logit offset and test it on held-out episodes.
   Why: if machine identity carries a persistent hidden offset, short-history cells can shrink toward the slot mean; if not, drop it (the episode-level version already gave ≤0.01).
   Effort S to test, S to use; gain 0–0.1, concentrated in short.
9. [R] Evaluation discipline. 57.90 vs 57.96 on 96 episodes is noise; tune blend weights and any new switch by repeated episode-level bootstrap / leave-one-episode-out, and report the CI before shipping.
   Why: with 48 episodes per condition, a per-condition weight table can overfit; shrinking toward the global w is often better than a spurious per-condition win.
   Effort S; expected gain ~0, but it prevents adopting ideas 1–8 on noise.

## 3. Round-2 order of work (14 h)
1. Run r2.py; read the per-condition weight table. 2. If fallback ≥ main anywhere, do ideas 1–2 first. 3. Re-run fit.py votes on the new kit and diff pairs/lags/cell priors; if diffuse, widen the zoo (idea 6) before touching the main model.
4. Implement only ideas whose validation gain clears the bootstrap CI; rebuild, verify the zip rerun, upload, keep the receipt.

## 4. What not to do
- No temperature/sharpen, conformal step, or pooled-baseline blend: squared error is proper and blending hurt (ε=0.01 → 57.897 vs 57.96).
- No generic ML on raw history: in-family it cannot beat the exact posterior, and it will not transfer better than the rule zoo if the generator changes.
- Do not wait for a public simulator/solution; none exists.
