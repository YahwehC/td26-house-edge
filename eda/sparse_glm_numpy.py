"""Research-only NumPy implementation checked against the stdlib recursion."""
import math
import numpy as np
from he.sparse_joint import topology
from he.model import SIZE_IDX, _pair_prior, DEFAULT_CFG, logsumexp
from he.joint import lag_prior_marginal, lag_prior_for_pair


def components(q, peers, params, config, hot, offset=0., cavity=False, perturbation=None):
    cfg = dict(DEFAULT_CFG, **params['cfg'])
    pri = _pair_prior(cfg)
    marginal = lag_prior_marginal(cfg)
    cf = config['coef']
    i = config['types'].index(q['machine_type'])
    scale = config['scale']
    vectors = params['size_vectors'][q['machine_type']]
    eps = cfg['size_eps']
    vectors = np.array([[(1-eps)*vectors[r][k]+eps*vectors[1-r][k] for k in range(4)] for r in (0, 1)])
    seq, cutoff = q['seq'], q['cutoff']
    logs, predictions = [], []
    details = {'messages':[], 'cold':[], 'hot':[], 'wins':[]}
    for (a, b), lp in pri.items():
        lag_weights = lag_prior_for_pair((a, b), cfg, marginal)
        lag = max(lag_weights, key=lag_weights.get)
        cell, regime, dest = topology(a, b, lag)
        cell, regime, dest = np.array(cell), np.array(regime), np.array(dest)
        n = len(cell)
        for orientation in (0, 1):
            for sign in (0, 1):
                sizes = vectors[regime^sign]
                common = np.take(cf[2*i:2*i+2], cell^orientation)+cf[6]*q['signal']+cf[7+i]*((regime^sign)-.5)+offset

                def probabilities(peer, weather):
                    eta = common+cf[10]*(peer-3.5)
                    return (1-weather)/(1+np.exp(-(eta-scale)))+weather/(1+np.exp(-(eta+scale)))

                def advance(state, win, value):
                    if value is None:
                        return (np.bincount(dest[0], weights=state*(1-win), minlength=n)
                                +np.bincount(dest[1], weights=state*win*sizes[:, 0], minlength=n)
                                +np.bincount(dest[2], weights=state*win*(1-sizes[:, 0]), minlength=n))
                    category = 0 if value == 0 else (1 if value == 1 else 2)
                    weights = state*(1-win) if value == 0 else state*win*sizes[:, SIZE_IDX[value]]
                    return np.bincount(dest[category], weights=weights, minlength=n)

                state = np.full(n, 1/n)
                stationary = probabilities(3.5, .5)
                for _ in range(256):
                    nxt = advance(state, stationary, None)
                    difference = np.max(np.abs(nxt-state))
                    state = nxt
                    if difference < 1e-11:
                        break
                loglike = lp-math.log(4)
                cache = {peer: probabilities(peer, .5) for peer in set(peers.values())}
                forwards = []
                for tick in range(cutoff):
                    value = seq.get(tick)
                    if cavity:
                        forwards.append(state.copy())
                    rates = probabilities(peers[tick], perturbation[1]) if perturbation is not None and tick == perturbation[0] else cache[peers[tick]]
                    state = advance(state, rates, value)
                    if value is not None:
                        norm = state.sum()
                        loglike += math.log(norm)
                        state /= norm
                win = probabilities(peers[cutoff], hot)
                predictions.append([float(np.sum(state*(1-win)))]+[float(np.sum(state*win*sizes[:, k])) for k in range(4)])
                logs.append(loglike)
                if cavity:
                    for name, weather in (('cold',0.),('hot',1.)):
                        rate = probabilities(peers[cutoff], weather)
                        details[name].append([float(np.sum(state*(1-rate)))]+[float(np.sum(state*rate*sizes[:, k])) for k in range(4)])
                    backward = np.ones(n)
                    messages = {}
                    smoothed_wins = {}
                    for tick in range(cutoff-1, -1, -1):
                        value = seq.get(tick)
                        rate = cache[peers[tick]]
                        if value is None:
                            losing = (1-rate)*backward[dest[0]]
                            winning = rate*(sizes[:,0]*backward[dest[1]]+(1-sizes[:,0])*backward[dest[2]])
                            total = losing+winning
                            smoothed_wins[tick] = float(np.sum(forwards[tick]*winning)/np.sum(forwards[tick]*total))
                            backward = total
                        else:
                            smoothed_wins[tick] = float(value > 0)
                            category = 0 if value == 0 else (1 if value == 1 else 2)
                            downstream = backward[dest[category]]
                            emission = 1-rate if value == 0 else rate*sizes[:,SIZE_IDX[value]]
                            denominator = float(np.sum(forwards[tick]*emission*downstream))
                            if cavity is True:
                                ratio = []
                                for weather in (0.,1.):
                                    alternative = probabilities(peers[tick], weather)
                                    em = 1-alternative if value == 0 else alternative*sizes[:,SIZE_IDX[value]]
                                    ratio.append(float(np.sum(forwards[tick]*em*downstream))/denominator)
                                messages[tick] = ratio
                            backward = emission*downstream
                        backward /= np.max(backward)
                    details['messages'].append(messages)
                    details['wins'].append(smoothed_wins)
    if cavity:
        return logs, predictions, details
    return logs, predictions


def predict(q, peers, params, config, hot, offset=0.):
    if config.get('sparse_offset_quadrature'):
        from he.short_weather_joint import NODES, WEIGHTS
        sd = math.sqrt(config.get('machine_variance', 0.))
        logs, predictions = [], []
        for node, weight in zip(NODES, WEIGHTS):
            ll, pr = components(q, peers, params, config, hot, node*sd)
            logs.extend(value+math.log(weight) for value in ll)
            predictions.extend(pr)
    else:
        logs, predictions = components(q, peers, params, config, hot, offset)
    norm = logsumexp(logs)
    return [sum(math.exp(ll-norm)*pr[k] for ll, pr in zip(logs, predictions)) for k in range(5)]
