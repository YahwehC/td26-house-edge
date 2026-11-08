"""Smoothed missing-win probabilities vs independently clamped complete reruns."""
import sys
import math
sys.path.insert(0, '.')
from he.api import load_params
from he.model import load_split, logsumexp
from he.peer import peer_winners
from sparse_glm_numpy import components
from peer_smoothing_numpy import smoothed_wins
p = load_params('runs/main_v13/params.json')
qs, _ = load_split('kit/public','train')
short = [q for q in qs if len(q['seq']) < .6*q['cutoff']]
for q in short[:3]:
    mates = [m for m in qs if m['episode_id'] == q['episode_id']]
    peers = {t:peer_winners(mates,q,t-3,True) for t in range(q['cutoff']+1)}
    belief = smoothed_wins(q,peers,p,p['global_state'],.13)
    normal = logsumexp(components(q,peers,p,p['global_state'],.5,.13)[0])
    missing = [t for t in range(8,q['cutoff']) if t not in q['seq']]
    for tick in (missing[0],missing[len(missing)//2],missing[-1]):
        probabilities = []
        for value in (0,1,2,5,20):
            changed = dict(q,seq=dict(q['seq']))
            changed['seq'][tick] = value
            likelihood = logsumexp(components(changed,peers,p,p['global_state'],.5,.13)[0])
            probabilities.append(math.exp(likelihood-normal))
        assert abs(sum(probabilities)-1) < 1e-10
        assert abs(sum(probabilities[1:])-belief[tick]) < 1e-10
    assert all(belief[t] == float(value>0) for t,value in q['seq'].items())
    print(q['query_id'],'smoothed missing wins match clamped likelihood sums',flush=True)
