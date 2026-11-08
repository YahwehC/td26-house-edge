"""Research-only weather evidence marginalized over sparse machine hypotheses.

Each local message is the full machine-history likelihood ratio when only that
tick's weather is changed from its half/half mixture. Forward/backward messages
compute these ratios exactly for the frozen single-machine model. Multiplying
them in the common weather HMM is an approximation: messages from one machine
can be correlated through its unknown fixed rule and unobserved history.
"""
import math
from collections import defaultdict
from he.model import logsumexp
from he.global_state import state_rows
from he.machine_offset import estimate_offsets
from he.peer import peer_winners
from he.weather import filter_episode
from sparse_glm_numpy import components


def machine_messages(q, peers, params, config, offset=0.):
    logs, _, details = components(q, peers, params, config, .5, offset, cavity=True)
    normal = logsumexp(logs)
    weights = [math.exp(ll-normal) for ll in logs]
    messages = {t:[math.fsum(w*message[t][state] for w,message in zip(weights,details['messages']))
                   for state in (0,1)] for t in q['seq']}
    predictions = {name:[math.fsum(w*row[k] for w,row in zip(weights,details[name]))
                         for k in range(5)] for name in ('cold','hot')}
    return messages, predictions


def predict(queries, params):
    config = params['global_state']
    rows, _ = state_rows(queries, params)
    offsets = estimate_offsets(rows, config)
    episodes = defaultdict(list)
    for q in queries:
        episodes[q['episode_id']].append(q)
    out = {}
    for episode, qs in episodes.items():
        assert len({q['cutoff'] for q in qs}) == 1
        cutoff = qs[0]['cutoff']
        times = defaultdict(list)
        conditional = {}
        for q in qs:
            peers = {t:peer_winners(qs,q,t-int(params['cfg'].get('peer_lag',3)),True) for t in range(cutoff+1)}
            messages, prediction = machine_messages(q,peers,params,config,offsets.get((episode,q['machine_id']),0.))
            conditional[q['query_id']] = prediction
            for tick, values in messages.items():
                if tick < 8:
                    continue
                maximum = max(values)
                times[tick].append(([value/maximum for value in values],1))
        hot = filter_episode((cutoff,times),config['scale'],config['hazard'])[1]
        for q in qs:
            pred = conditional[q['query_id']]
            out[q['query_id']] = [(1-hot)*a+hot*b for a,b in zip(pred['cold'],pred['hot'])]
    return out
