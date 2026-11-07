"""Generic fallback Bayesian next-spin predictor (pure standard library).

This module is deliberately assumption-light: it does not hard-code lag pairs,
cell functions or a particular change mechanism.  The two parts are

* win part: a Beta-Binomial table for every context set of at most two lags,
  with an optional single change point (independent tables before/after tau);
* size part: Dirichlet-multinomial tables for the no-context model and for each
  single lag (bit = payout >= 2).

Cell/table priors are empirical-Bayes pooled at machine-type level, context
priors are fitted by EM, and the two concentration parameters are selected by
total marginal likelihood.  Everything is deterministic and JSON serialisable.
"""
import math
from collections import defaultdict

from .model import MAXLAG, SIZE_IDX, betaln, logsumexp

FALLBACK_DEFAULTS = {
    "max_lag": 8,
    "win_kappa_grid": [0.5, 1.0, 2.0, 4.0],
    "size_alpha_grid": [1.0, 4.0, 16.0],
    "change_prior": 0.5,
    "tau_min_back": 16,
    "tau_max_back": 128,
    "tau_step": 8,
    "ctx_smooth": 0.5,
    "em_iters": 3,
}


# --------------------------------------------------------------------------- options / contexts
def _merge_opts(opts):
    """Merge user options over the documented defaults, coercing JSON-safe types."""
    merged = dict(FALLBACK_DEFAULTS)
    if opts:
        merged.update(opts)
    out = {}
    for key, value in merged.items():
        if key in ("win_kappa_grid", "size_alpha_grid"):
            out[key] = sorted(float(x) for x in value)
        elif key in ("max_lag", "tau_min_back", "tau_max_back", "tau_step", "em_iters"):
            out[key] = int(value)
        elif key in ("change_prior", "ctx_smooth"):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def _context_list(max_lag):
    """(), (L,), and (A, B) contexts in deterministic order."""
    contexts = [()]
    for lag in range(1, max_lag + 1):
        contexts.append((lag,))
    for a in range(1, max_lag + 1):
        for b in range(a + 1, max_lag + 1):
            contexts.append((a, b))
    return contexts


def _size_context_list(max_lag):
    return [None] + list(range(1, max_lag + 1))


def _context_key(context):
    return ",".join(str(lag) for lag in context)


def _size_key(context):
    return "none" if context is None else str(context)


def _safe_rate(wins, ticks):
    if ticks <= 0:
        return 0.5
    return min(max(wins / ticks, 1e-4), 1.0 - 1e-4)


def _size_mu(counts):
    total = counts[0] + counts[1] + counts[2] + counts[3] + 4.0
    return [(counts[k] + 1.0) / total for k in range(4)]


def _normalise_vector(values):
    values = [max(1e-12, float(x)) for x in values]
    if len(values) != 4:
        values = [0.25, 0.25, 0.25, 0.25]
    total = math.fsum(values)
    if total <= 0.0:
        return [0.25, 0.25, 0.25, 0.25]
    return [value / total for value in values]


def _taus(cutoff, opts):
    """Candidate change ticks for one query, in ascending order."""
    start = cutoff - int(opts["tau_max_back"])
    stop = cutoff - int(opts["tau_min_back"])
    step = int(opts["tau_step"])
    if step <= 0 or stop < start:
        return []
    return list(range(start, stop + 1, step))


# --------------------------------------------------------------------------- Beta tables
class _BetaTable:
    """Lookup table for Beta(kappa*m, kappa*(1-m)) cell marginal likelihoods."""

    __slots__ = ("a", "b", "lga", "lgb", "lgab", "base")

    def __init__(self, m, kappa, nmax):
        m = min(max(float(m), 1e-9), 1.0 - 1e-9)
        kappa = max(float(kappa), 1e-9)
        self.a = kappa * m
        self.b = kappa * (1.0 - m)
        a = self.a
        b = self.b
        self.lga = [math.lgamma(a + n) for n in range(nmax + 1)]
        self.lgb = [math.lgamma(b + n) for n in range(nmax + 1)]
        self.lgab = [math.lgamma(a + b + n) for n in range(nmax + 1)]
        self.base = betaln(a, b)


class _SizeTable:
    """Lookup table for a Dirichlet-multinomial with prior alpha * mu."""

    __slots__ = ("alpha", "mu", "prior", "lg_alpha", "lg_alpha_n", "lgamma_mu")

    def __init__(self, mu, alpha, nmax):
        alpha = max(float(alpha), 1e-9)
        mu = _normalise_vector(mu)
        self.alpha = alpha
        self.mu = mu
        self.prior = [alpha * x for x in mu]
        self.lg_alpha = math.lgamma(alpha)
        self.lg_alpha_n = [math.lgamma(alpha + n) for n in range(nmax + 1)]
        self.lgamma_mu = []
        for k in range(4):
            base = math.lgamma(self.prior[k])
            self.lgamma_mu.append([math.lgamma(self.prior[k] + n) - base
                                   for n in range(nmax + 1)])


# --------------------------------------------------------------------------- event construction
def _machine_win_events(query, contexts):
    """Build sorted usable ticks / compact cell codes / total cell counts per context.

    A compact code is ``2 * cell + win_bit``.  Cell bits are packed in the same
    order as the lags in the context.
    """
    seq = query["seq"]
    ticks_all = sorted(seq)
    events = []
    for context in contexts:
        width = 1 << len(context)
        ticks = []
        codes = []
        count0 = [0] * width
        count1 = [0] * width
        if not context:
            for tick in ticks_all:
                y = 1 if seq[tick] > 0 else 0
                ticks.append(tick)
                codes.append(y)
                if y:
                    count1[0] += 1
                else:
                    count0[0] += 1
        else:
            for tick in ticks_all:
                cell = 0
                usable = True
                for j, lag in enumerate(context):
                    value = seq.get(tick - lag)
                    if value is None:
                        usable = False
                        break
                    if value > 0:
                        cell |= 1 << j
                if not usable:
                    continue
                y = 1 if seq[tick] > 0 else 0
                ticks.append(tick)
                codes.append((cell << 1) | y)
                if y:
                    count1[cell] += 1
                else:
                    count0[cell] += 1
        events.append((ticks, codes, count0, count1))
    return events


def _machine_size_events(query, size_contexts):
    """Counts [cell][size_index] for every size context of one query."""
    seq = query["seq"]
    wins = [(tick, SIZE_IDX[value]) for tick, value in seq.items() if value > 0]
    wins.sort(key=lambda item: item[0])
    events = []
    for context in size_contexts:
        if context is None:
            counts = [[0, 0, 0, 0]]
            for _, size_index in wins:
                counts[0][size_index] += 1
        else:
            counts = [[0, 0, 0, 0], [0, 0, 0, 0]]
            for tick, size_index in wins:
                value = seq.get(tick - context)
                if value is None:
                    continue
                bit = 1 if value >= 2 else 0
                counts[bit][size_index] += 1
        events.append(counts)
    return events


def _cutoff_win_cells(query, context, base_win):
    """Possible cutoff cell indices and their probabilities (missing bits marginalised)."""
    if not context:
        return [(0, 1.0)]
    seq = query["seq"]
    cutoff = query["cutoff"]
    out = []

    def walk(j, cell, probability):
        if j == len(context):
            out.append((cell, probability))
            return
        lag = context[j]
        value = seq.get(cutoff - lag)
        if value is None:
            walk(j + 1, cell, probability * (1.0 - base_win))
            walk(j + 1, cell | (1 << j), probability * base_win)
        else:
            bit = 1 if value > 0 else 0
            walk(j + 1, cell | (bit << j), probability)

    walk(0, 0, 1.0)
    return out


def _cutoff_size_cells(query, context, base_big):
    if context is None:
        return [(0, 1.0)]
    value = query["seq"].get(query["cutoff"] - context)
    if value is not None:
        return [(1 if value >= 2 else 0, 1.0)]
    return [(0, 1.0 - base_big), (1, base_big)]


# --------------------------------------------------------------------------- win evidence
def _predict_counts(count0, count1, cells, alpha0, beta0):
    denom0 = alpha0 + beta0
    out = 0.0
    for cell, probability in cells:
        n1 = count1[cell]
        n0 = count0[cell]
        out += probability * ((alpha0 + n1) / (denom0 + n0 + n1))
    return out


def _win_stats(event, table, taus, change_prior, cutoff_cells, want_pred):
    """Log evidence and (optionally) predictive win probability for one context.

    The change-point sweep moves usable ticks from the post-tau table to the
    pre-tau table in one pass; each moved tick updates only its own cell.
    """
    ticks, codes, total0, total1 = event
    ncell = len(total0)
    lga = table.lga
    lgb = table.lgb
    lgab = table.lgab
    base = table.base

    post0 = total0[:]
    post1 = total1[:]
    post_f = [0.0] * ncell
    h0_lml = 0.0
    for cell in range(ncell):
        n1 = post1[cell]
        n0 = post0[cell]
        value = lga[n1] + lgb[n0] - lgab[n1 + n0] - base
        post_f[cell] = value
        h0_lml += value

    pre0 = [0] * ncell
    pre1 = [0] * ncell
    pre_f = [0.0] * ncell
    pre_lml = 0.0
    post_lml = h0_lml

    h0_pred = _predict_counts(post0, post1, cutoff_cells, table.a, table.b) if want_pred else None
    terms = []
    preds = [] if want_pred else None

    j = 0
    n = len(ticks)
    for tau in taus:
        while j < n and ticks[j] < tau:
            code = codes[j]
            cell = code >> 1
            y = code & 1
            old_pre = pre_f[cell]
            old_post = post_f[cell]
            if y:
                pre1[cell] += 1
                post1[cell] -= 1
            else:
                pre0[cell] += 1
                post0[cell] -= 1
            n1 = pre1[cell]
            n0 = pre0[cell]
            new_pre = lga[n1] + lgb[n0] - lgab[n1 + n0] - base
            n1 = post1[cell]
            n0 = post0[cell]
            new_post = lga[n1] + lgb[n0] - lgab[n1 + n0] - base
            pre_f[cell] = new_pre
            post_f[cell] = new_post
            pre_lml += new_pre - old_pre
            post_lml += new_post - old_post
            j += 1
        if j >= 8 and n - j >= 8:
            terms.append(pre_lml + post_lml)
            if want_pred:
                preds.append(_predict_counts(post0, post1, cutoff_cells, table.a, table.b))

    if not terms:
        return h0_lml, h0_pred

    cp = change_prior
    if cp <= 0.0:
        return h0_lml, h0_pred

    if cp >= 1.0:
        log_parts = [-math.log(len(terms)) + value for value in terms]
        pred_parts = preds
    else:
        log_parts = [math.log1p(-cp) + h0_lml]
        log_tau = math.log(cp) - math.log(len(terms))
        log_parts.extend(log_tau + value for value in terms)
        if want_pred:
            pred_parts = [h0_pred]
            pred_parts.extend(preds)
        else:
            pred_parts = None

    log_evidence = logsumexp(log_parts)
    if not want_pred:
        return log_evidence, None
    predictive = 0.0
    for log_weight, pred in zip(log_parts, pred_parts):
        predictive += math.exp(log_weight - log_evidence) * pred
    return log_evidence, predictive


# --------------------------------------------------------------------------- size evidence
def _size_stats(counts, table, cutoff_cells, want_pred):
    """Log evidence and (optionally) the predictive size-given-win vector."""
    lg_alpha = table.lg_alpha
    lg_alpha_n = table.lg_alpha_n
    lgamma_mu = table.lgamma_mu
    log_evidence = 0.0
    for nk in counts:
        total = nk[0] + nk[1] + nk[2] + nk[3]
        value = lg_alpha - lg_alpha_n[total]
        value += lgamma_mu[0][nk[0]]
        value += lgamma_mu[1][nk[1]]
        value += lgamma_mu[2][nk[2]]
        value += lgamma_mu[3][nk[3]]
        log_evidence += value

    if not want_pred:
        return log_evidence, None

    prior = table.prior
    alpha = table.alpha
    predictive = [0.0, 0.0, 0.0, 0.0]
    for cell, probability in cutoff_cells:
        nk = counts[cell]
        total = nk[0] + nk[1] + nk[2] + nk[3]
        denom = alpha + total
        predictive[0] += probability * (prior[0] + nk[0]) / denom
        predictive[1] += probability * (prior[1] + nk[1]) / denom
        predictive[2] += probability * (prior[2] + nk[2]) / denom
        predictive[3] += probability * (prior[3] + nk[3]) / denom
    return log_evidence, predictive


# --------------------------------------------------------------------------- empirical Bayes
def _mixture_total(rows, n_contexts):
    """Sum over machines of log(sum_C uniform(C) * exp(row[C]))."""
    log_uniform = -math.log(n_contexts)
    total = 0.0
    for row in rows:
        total += logsumexp([value + log_uniform for value in row])
    return total


def _em_prior(rows, n_contexts, smooth, iters):
    """Empirical-Bayes context prior; rows are per-machine log evidences."""
    prior = [1.0 / n_contexts] * n_contexts
    n_machines = len(rows)
    if n_machines == 0 or iters <= 0:
        return prior
    for _ in range(iters):
        sums = [0.0] * n_contexts
        for row in rows:
            log_weights = [math.log(max(prior[c], 1e-300)) + row[c]
                           for c in range(n_contexts)]
            top = max(log_weights)
            weights = [math.exp(value - top) for value in log_weights]
            z = math.fsum(weights)
            if z <= 0.0:
                continue
            inv = 1.0 / z
            for c in range(n_contexts):
                sums[c] += weights[c] * inv
        denom = n_machines + smooth * n_contexts
        prior = [(smooth + sums[c]) / denom for c in range(n_contexts)]
    return prior


# --------------------------------------------------------------------------- fit
def fit_fallback(query_lists, opts=None):
    """Fit JSON-serialisable fallback parameters from a list of query lists.

    Only ``seq`` and ``machine_type`` are used; targets are never read.
    """
    options = _merge_opts(opts)
    queries = []
    for query_list in (query_lists or []):
        if query_list:
            queries.extend(query_list)
    queries.sort(key=lambda q: q["query_id"])

    max_lag = int(options["max_lag"])
    contexts = _context_list(max_lag)
    size_contexts = _size_context_list(max_lag)
    n_context = len(contexts)
    n_size_context = len(size_contexts)

    type_ticks = defaultdict(int)
    type_wins = defaultdict(int)
    type_size = defaultdict(lambda: [0, 0, 0, 0])
    global_ticks = 0
    global_wins = 0
    global_size = [0, 0, 0, 0]

    for query in queries:
        machine_type = query["machine_type"]
        for payout in query["seq"].values():
            type_ticks[machine_type] += 1
            global_ticks += 1
            if payout > 0:
                type_wins[machine_type] += 1
                global_wins += 1
                size_index = SIZE_IDX[payout]
                type_size[machine_type][size_index] += 1
                global_size[size_index] += 1

    global_rate = _safe_rate(global_wins, global_ticks)
    types = sorted(type_ticks)
    win_rate = {}
    for machine_type in types:
        win_rate[machine_type] = _safe_rate(type_wins[machine_type],
                                           type_ticks[machine_type])
    win_rate["_global"] = global_rate

    size_mu = {machine_type: _size_mu(type_size[machine_type]) for machine_type in types}
    size_mu["_global"] = _size_mu(global_size)

    table_types = list(types)
    if "_global" not in table_types:
        table_types.append("_global")

    nmax = max((len(query["seq"]) for query in queries), default=0) + 2
    kappas = [float(x) for x in options["win_kappa_grid"]]
    alphas = [float(x) for x in options["size_alpha_grid"]]

    beta_tables = {}
    for kappa in kappas:
        beta_tables[kappa] = {
            machine_type: _BetaTable(win_rate.get(machine_type, global_rate), kappa, nmax)
            for machine_type in table_types
        }
    size_tables = {}
    for alpha in alphas:
        size_tables[alpha] = {
            machine_type: _SizeTable(size_mu.get(machine_type, size_mu["_global"]),
                                     alpha, nmax)
            for machine_type in table_types
        }

    win_rows = {kappa: [] for kappa in kappas}
    size_rows = {alpha: [] for alpha in alphas}
    change_prior = float(options["change_prior"])

    for query in queries:
        machine_type = query["machine_type"]
        table_type = machine_type if machine_type in win_rate else "_global"
        cutoff = query["cutoff"]
        taus = _taus(cutoff, options)
        win_events = _machine_win_events(query, contexts)
        for kappa in kappas:
            table = beta_tables[kappa][table_type]
            row = []
            for event in win_events:
                row.append(_win_stats(event, table, taus, change_prior, None, False)[0])
            win_rows[kappa].append(row)

        size_events = _machine_size_events(query, size_contexts)
        for alpha in alphas:
            table = size_tables[alpha][table_type]
            row = [_size_stats(counts, table, None, False)[0] for counts in size_events]
            size_rows[alpha].append(row)

    if not kappas:
        kappas = [1.0]
        win_rows[1.0] = []
    if not alphas:
        alphas = [1.0]
        size_rows[1.0] = []

    kappa_total = {kappa: _mixture_total(win_rows[kappa], n_context) for kappa in kappas}
    best_kappa = kappas[0]
    best_kappa_value = kappa_total[best_kappa]
    for kappa in kappas[1:]:
        if kappa_total[kappa] > best_kappa_value:
            best_kappa = kappa
            best_kappa_value = kappa_total[kappa]

    alpha_total = {alpha: _mixture_total(size_rows[alpha], n_size_context) for alpha in alphas}
    best_alpha = alphas[0]
    best_alpha_value = alpha_total[best_alpha]
    for alpha in alphas[1:]:
        if alpha_total[alpha] > best_alpha_value:
            best_alpha = alpha
            best_alpha_value = alpha_total[alpha]

    win_prior = _em_prior(win_rows[best_kappa], n_context,
                          float(options["ctx_smooth"]), int(options["em_iters"]))
    size_prior = _em_prior(size_rows[best_alpha], n_size_context,
                           float(options["ctx_smooth"]), int(options["em_iters"]))

    win_ctx_prior = {_context_key(contexts[i]): float(win_prior[i])
                     for i in range(n_context)}
    size_ctx_prior = {_size_key(size_contexts[i]): float(size_prior[i])
                      for i in range(n_size_context)}

    return {
        "fallback_version": 1,
        "opts": options,
        "win_kappa": float(best_kappa),
        "size_alpha": float(best_alpha),
        "win_rate": {machine_type: float(win_rate[machine_type])
                     for machine_type in table_types},
        "size_mu": {machine_type: [float(x) for x in size_mu[machine_type]]
                    for machine_type in table_types},
        "win_ctx_prior": win_ctx_prior,
        "size_ctx_prior": size_ctx_prior,
    }


# --------------------------------------------------------------------------- predict
def _finalise_row(raw):
    """Clip to [1e-9, 1] and renormalise while preserving the lower bound."""
    row = [min(1.0, max(1e-9, float(value))) for value in raw]
    total = math.fsum(row)
    if total <= 0.0:
        return [0.2, 0.2, 0.2, 0.2, 0.2]
    row = [value / total for value in row]
    floor = 1e-9
    deficit = 0.0
    for i in range(5):
        if row[i] < floor:
            deficit += floor - row[i]
            row[i] = floor
    if deficit > 0.0:
        largest = max(range(5), key=lambda i: row[i])
        row[largest] = max(floor, row[largest] - deficit)
        total = math.fsum(row)
        if total <= 0.0:
            return [0.2, 0.2, 0.2, 0.2, 0.2]
        row = [value / total for value in row]
    return row


def predict_fallback(queries, fparams):
    """Return ``{query_id: [p_0, p_1, p_2, p_5, p_20]}`` for the fitted fallback."""
    options = _merge_opts(fparams.get("opts"))
    max_lag = int(options["max_lag"])
    contexts = _context_list(max_lag)
    size_contexts = _size_context_list(max_lag)
    n_context = len(contexts)
    n_size_context = len(size_contexts)

    kappa = float(fparams["win_kappa"])
    alpha = float(fparams["size_alpha"])
    win_rate = fparams.get("win_rate") or {}
    size_mu = fparams.get("size_mu") or {}
    global_rate = min(max(float(win_rate.get("_global", 0.5)), 1e-4), 1.0 - 1e-4)
    global_mu = _normalise_vector(size_mu.get("_global", [0.25, 0.25, 0.25, 0.25]))

    win_ctx_prior = fparams.get("win_ctx_prior") or {}
    size_ctx_prior = fparams.get("size_ctx_prior") or {}
    win_prior = [float(win_ctx_prior.get(_context_key(context), 1.0 / n_context))
                 for context in contexts]
    size_prior = [float(size_ctx_prior.get(_size_key(context), 1.0 / n_size_context))
                  for context in size_contexts]
    win_z = math.fsum(win_prior)
    size_z = math.fsum(size_prior)
    if win_z <= 0.0:
        win_prior = [1.0 / n_context] * n_context
    else:
        win_prior = [value / win_z for value in win_prior]
    if size_z <= 0.0:
        size_prior = [1.0 / n_size_context] * n_size_context
    else:
        size_prior = [value / size_z for value in size_prior]

    ordered = sorted(queries, key=lambda q: q["query_id"])
    nmax = max((len(query["seq"]) for query in ordered), default=0) + 2
    machine_types = sorted({query["machine_type"] for query in ordered})

    beta_tables = {}
    size_tables = {}
    for machine_type in machine_types:
        machine_rate = min(max(float(win_rate.get(machine_type, global_rate)),
                               1e-4), 1.0 - 1e-4)
        beta_tables[machine_type] = _BetaTable(machine_rate, kappa, nmax)
        machine_mu = _normalise_vector(size_mu.get(machine_type, global_mu))
        size_tables[machine_type] = _SizeTable(machine_mu, alpha, nmax)

    change_prior = float(options["change_prior"])
    out = {}
    for query in ordered:
        seq = query["seq"]
        machine_type = query["machine_type"]
        table = beta_tables[machine_type]
        size_table = size_tables[machine_type]
        n_obs = len(seq)
        n_wins = sum(1 for payout in seq.values() if payout > 0)
        n_big = sum(1 for payout in seq.values() if payout >= 2)
        base_win = (n_wins + 1.0) / (n_obs + 2.0) if n_obs else 0.5
        base_big = (n_big + 1.0) / (n_obs + 2.0) if n_obs else 0.5

        cutoff = query["cutoff"]
        taus = _taus(cutoff, options)
        win_events = _machine_win_events(query, contexts)
        log_evidence = []
        predictions = []
        for context, event in zip(contexts, win_events):
            cells = _cutoff_win_cells(query, context, base_win)
            evidence, prediction = _win_stats(event, table, taus, change_prior,
                                              cells, True)
            log_evidence.append(evidence)
            predictions.append(prediction)

        log_weights = [math.log(max(win_prior[c], 1e-300)) + log_evidence[c]
                       for c in range(n_context)]
        top = max(log_weights)
        weights = [math.exp(value - top) for value in log_weights]
        weight_sum = math.fsum(weights)
        if weight_sum <= 0.0:
            pw = predictions[0]
        else:
            inverse = 1.0 / weight_sum
            pw = math.fsum(weights[c] * inverse * predictions[c]
                           for c in range(n_context))

        size_events = _machine_size_events(query, size_contexts)
        size_log_evidence = []
        size_predictions = []
        for context, counts in zip(size_contexts, size_events):
            cells = _cutoff_size_cells(query, context, base_big)
            evidence, prediction = _size_stats(counts, size_table, cells, True)
            size_log_evidence.append(evidence)
            size_predictions.append(prediction)

        size_log_weights = [math.log(max(size_prior[c], 1e-300)) + size_log_evidence[c]
                            for c in range(n_size_context)]
        top = max(size_log_weights)
        size_weights = [math.exp(value - top) for value in size_log_weights]
        size_weight_sum = math.fsum(size_weights)
        if size_weight_sum <= 0.0:
            size_prediction = size_predictions[0]
        else:
            inverse = 1.0 / size_weight_sum
            size_prediction = [math.fsum(size_weights[c] * inverse * size_predictions[c][k]
                                         for c in range(n_size_context))
                               for k in range(4)]

        pw = min(max(pw, 1e-12), 1.0 - 1e-12)
        raw = [1.0 - pw] + [pw * value for value in size_prediction]
        out[query["query_id"]] = _finalise_row(raw)

    return out
