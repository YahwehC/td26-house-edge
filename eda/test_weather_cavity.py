"""Compare backward cavity ratios against complete forward likelihood reruns."""
import sys
import math
sys.path.insert(0, '.')
from he.api import load_params
from he.model import load_split, logsumexp
from he.peer import peer_winners
from sparse_glm_numpy import components, predict
from weather_cavity_numpy import machine_messages
p = load_params('runs/main_v13/params.json')
qs, _ = load_split('kit/public', 'train')
short = [q for q in qs if len(q['seq']) < .6*q['cutoff']]
for q in short[:3]:
    mates = [r for r in qs if r['episode_id'] == q['episode_id']]
    peers = {t:peer_winners(mates,q,t-3,True) for t in range(q['cutoff']+1)}
    offset = .13
    messages, prediction = machine_messages(q,peers,p,p['global_state'],offset)
    normal = logsumexp(components(q,peers,p,p['global_state'],.5,offset)[0])
    ticks = sorted(q['seq'])
    for tick in (ticks[0],ticks[len(ticks)//2],ticks[-1]):
        for weather in (0,1):
            perturbed = components(q,peers,p,p['global_state'],.5,offset,perturbation=(tick,weather))[0]
            direct = math.exp(logsumexp(perturbed)-normal)
            assert abs(direct-messages[tick][weather]) < 1e-10, (tick,direct,messages[tick][weather])
    for hot in (.1,.8):
        direct = predict(q,peers,p,p['global_state'],hot,offset)
        mixed = [(1-hot)*a+hot*b for a,b in zip(prediction['cold'],prediction['hot'])]
        assert max(abs(a-b) for a,b in zip(direct,mixed)) < 1e-12
    print(q['query_id'],'cavity ratios and conditional forecasts match direct reruns',flush=True)
