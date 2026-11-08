"""Test whether shared-weather dwell times violate a memoryless hazard.
Uses only train histories and the train-fitted global logits. Scores likelihood
on alternating original episodes separately; no validation targets accessed.
"""
import sys,json,pickle,math
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0,'.')
rows=pickle.load(open('eda/_state_rows.pkl','rb'));cfg=json.load(open('eda/ev64_global_hmm.json'));cf=cfg['coef'];types=cfg['types'];scale=cfg['scale'];by=defaultdict(lambda:defaultdict(lambda:np.zeros(2)))
for r in rows:
    i=types.index(r['type']);eta=cf[2*i+r['high']]+cf[6]*r['signal']+cf[7+i]*(r['regime']-.5)+cf[10]*(r['peer']-3.5)
    z=eta+np.array([-scale,scale]);by[r['episode']][r['tick']]+=r['y']*z-np.logaddexp(0,z)

def score(times,minimum,hazard):
    n=minimum+1;state=np.full((2,n),.5/(minimum+1/hazard));state[:,-1]/=hazard
    evidence=0.;last=None
    def transition(state):
        out=np.zeros_like(state)
        if n>1:out[:,1:]+=state[:,:-1]
        out[:,-1]+=state[:,-1]*(1-hazard);out[::-1,0]+=state[:,-1]*hazard
        return out
    for t,em in sorted(times.items()):
        if last is not None:
            for _ in range(t-last):state=transition(state)
        shift=max(em);state*=np.exp(em-shift)[:,None];z=state.sum();state/=z;evidence+=math.log(z)+shift;last=t
    return evidence
reports=[]
for minimum in (0,4,8,16,32):
    for mean in (16,32,50,80):
        if mean<=minimum:continue
        hazard=1/(mean-minimum);values=[score(times,minimum,hazard) for ep,times in sorted(by.items())]
        reports.append({'minimum':minimum,'mean':mean,'hazard_after_minimum':hazard,'loglik':sum(values),'halves':[sum(values[::2]),sum(values[1::2])]})
reports.sort(key=lambda r:-r['loglik']);print(json.dumps(reports[:10],indent=2));Path('eda/ev73_weather_dwell.json').write_text(json.dumps(reports,indent=2))
