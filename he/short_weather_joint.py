"""Experimental short-history joint integration of machine and weather states.

Other machines supply fixed history emissions. Each own lag/sign hypothesis
integrates a persistent Gaussian intercept by five-point Gauss-Hermite quadrature,
and a binary weather path by an exact forward recursion. Missing own lag categories
still use empirical independent marginals, as in structured_state.py.
"""
import itertools
import math
from .model import DEFAULT_CFG, SIZE_IDX, _pair_prior, logsumexp
from .joint import lag_prior_marginal, lag_prior_for_pair
from .peer import peer_winners

NODES = (-2.8569700138728056, -1.355626179974266, 0., 1.355626179974266, 2.8569700138728056)
WEIGHTS = (.011257411327720688, .22207592200561266, .5333333333333333,
           .22207592200561266, .011257411327720688)


def forward(events, cutoff, hazard):
    """Likelihood and forecast hot probability for arbitrary two-state emissions."""
    hot = .5
    last = None
    likelihood = 0.
    for tick, cold_emission, hot_emission in events:
        if last is not None:
            hot = .5+(hot-.5)*(1-2*hazard)**(tick-last)
        cold, warm = (1-hot)*cold_emission, hot*hot_emission
        normal = cold+warm
        likelihood += math.log(max(normal, 1e-300))
        hot = warm/normal
        last = tick
    if last is not None:
        hot = .5+(hot-.5)*(1-2*hazard)**(cutoff-last)
    return likelihood, hot


def predict(q, machines, params, config, machine_emissions):
    cfg = dict(DEFAULT_CFG, **params['cfg'])
    pair_prior = _pair_prior(cfg)
    marginal = lag_prior_marginal(cfg)
    cf = config['coef']
    type_index = config['types'].index(q['machine_type'])
    seq, cutoff = q['seq'], q['cutoff']
    scale, hazard = config['scale'], config['hazard']
    sd = math.sqrt(config.get('machine_variance', 0.))
    quadrature = list(zip(NODES, WEIGHTS)) if sd else [(0., 1.)]
    vectors = params['size_vectors'][q['machine_type']]
    eps = cfg['size_eps']
    vectors = [[(1-eps)*vectors[r][k]+eps*vectors[1-r][k] for k in range(4)] for r in (0, 1)]
    empirical = [(sum(v == 0 for v in seq.values())+1)/(len(seq)+3),
                 (sum(v == 1 for v in seq.values())+1)/(len(seq)+3),
                 (sum(v >= 2 for v in seq.values())+1)/(len(seq)+3)]
    peer_emission = {}
    for tick, records in machine_emissions.items():
        ll = [0., 0.]
        for mid, eta, y in records:
            if mid == q['machine_id'] or tick >= cutoff:
                continue
            for state, shift in enumerate((-scale, scale)):
                if isinstance(eta, (list, tuple)):
                    probability = eta[state]
                    ll[state] += math.log(max(1e-300, probability if y else 1-probability))
                else:
                    z = eta+shift
                    ll[state] += y*z-max(z, 0)-math.log1p(math.exp(-abs(z)))
        maximum = max(ll)
        peer_emission[tick] = [math.exp(v-maximum) for v in ll]
    times = sorted(t for t in set(seq) | set(peer_emission) if t < cutoff)
    peers = {t: peer_winners(machines, q, t-int(cfg.get('peer_lag', 3)),
                            cfg.get('marginalize_missing_peers', False)) for t in list(seq)+[cutoff]}
    logs, predictions = [], []
    for (a, b), pair_log_prior in pair_prior.items():
        lp = lag_prior_for_pair((a, b), cfg, marginal)
        lag = max(lp, key=lp.get)
        contexts = {}
        for tick in list(seq)+[cutoff]:
            delays = sorted(set((a, b, lag)))
            choices = []
            for delay in delays:
                value = seq.get(tick-delay)
                choices.append(list(enumerate(empirical)) if value is None else
                               [(0 if value == 0 else (1 if value == 1 else 2), 1.)])
            mass = [0.]*4
            for combo in itertools.product(*choices):
                cats = {delay: v[0] for delay, v in zip(delays, combo)}
                state = int((cats[a] > 0) == (cats[b] > 0))
                regime = int(cats[lag] == 2)
                mass[2*state+regime] += math.prod(v[1] for v in combo)
            contexts[tick] = mass
        for orientation, sign in itertools.product((0, 1), repeat=2):
            for node, weight in quadrature:
                offset = node*sd

                def distribution(tick, weather):
                    output = [0.]*5
                    common = cf[6]*q['signal']+cf[10]*(peers[tick]-3.5)+offset+weather
                    for index, mass in enumerate(contexts[tick]):
                        if not mass:
                            continue
                        high = (index//2)^orientation
                        regime = (index%2)^sign
                        eta = cf[2*type_index+high]+cf[7+type_index]*(regime-.5)+common
                        win = 1/(1+math.exp(-eta))
                        output[0] += mass*(1-win)
                        for k in range(4):
                            output[k+1] += mass*win*vectors[regime][k]
                    return output

                events = []
                for tick in times:
                    cold, hot = peer_emission.get(tick, (1., 1.))
                    if tick in seq:
                        value = seq[tick]
                        index = 0 if value == 0 else SIZE_IDX[value]+1
                        cold *= distribution(tick, -scale)[index]
                        hot *= distribution(tick, scale)[index]
                    events.append((tick, cold, hot))
                likelihood, hot = forward(events, cutoff, hazard)
                cold_prediction = distribution(cutoff, -scale)
                hot_prediction = distribution(cutoff, scale)
                predictions.append([(1-hot)*cold+hot*warm for cold, warm in zip(cold_prediction, hot_prediction)])
                logs.append(pair_log_prior-math.log(4)+math.log(weight)+likelihood)
    normal = logsumexp(logs)
    return [math.fsum(math.exp(ll-normal)*row[k] for ll, row in zip(logs, predictions)) for k in range(5)]
