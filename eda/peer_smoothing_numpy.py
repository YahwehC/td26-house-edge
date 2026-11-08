"""Research-only one-pass refinement of missing peer wins from their histories."""
import math
from collections import defaultdict
from he.model import logsumexp
from he.global_state import state_rows
from he.machine_offset import estimate_offsets
from he.peer import peer_winners
from he.weather import filter_episode
from sparse_glm_numpy import components, predict as machine_predict


def smoothed_wins(q, peers, params, config, offset):
    logs, _, details = components(q,peers,params,config,.5,offset,cavity='wins')
    normal = logsumexp(logs)
    weights = [math.exp(value-normal) for value in logs]
    return {t:float(q['seq'][t]>0) if t in q['seq'] else
            math.fsum(w*record[t] for w,record in zip(weights,details['wins']))
            for t in range(q['cutoff'])}


def predict(queries, params):
    config = params['global_state']
    cf, types = config['coef'], config['types']
    rows, _ = state_rows(queries,params)
    offsets = estimate_offsets(rows,config)
    traces = defaultdict(lambda:defaultdict(list))
    cuts = {}
    for r in rows:
        i = types.index(r['type'])
        eta = cf[2*i+r['high']]+cf[6]*r['signal']+cf[7+i]*(r['regime']-.5)+cf[10]*(r['peer']-3.5)
        eta += offsets.get((r['episode'],r['machine']),0.)
        traces[r['episode']][r['tick']].append((eta,r['y']))
        cuts[r['episode']] = r['cutoff']
    hot = {ep:filter_episode((cuts[ep],times),config['scale'],config['hazard'])[1] for ep,times in traces.items()}
    episodes = defaultdict(list)
    for q in queries:
        episodes[q['episode_id']].append(q)
    result = {}
    for episode, qs in episodes.items():
        beliefs = {}
        for q in qs:
            peers = {t:peer_winners(qs,q,t-int(params['cfg'].get('peer_lag',3)),True) for t in range(q['cutoff']+1)}
            beliefs[q['machine_id']] = smoothed_wins(q,peers,params,config,offsets.get((episode,q['machine_id']),0.))
        for q in qs:
            peers = {}
            for t in range(q['cutoff']+1):
                tick = t-int(params['cfg'].get('peer_lag',3))
                peers[t] = 0. if tick < 0 else sum(beliefs[m['machine_id']][tick] for m in qs if m['machine_id'] != q['machine_id'])
            result[q['query_id']] = machine_predict(q,peers,params,config,hot.get(episode,.5),offsets.get((episode,q['machine_id']),0.))
    return result
