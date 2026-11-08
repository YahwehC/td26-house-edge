"""Train episodes: predictive log score for episode-specific weather amplitudes.
Learn mixture weights on other episode folds; condition on first 60% of each
held episode, then score its final 40%. No validation targets used.
"""
import sys,json,pickle,math
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0,'.')
from he.weather import filter_episode
rows=pickle.load(open('eda/_state_rows.pkl','rb'));cfg=json.load(open('eda/ev64_global_hmm.json'));cf=cfg['coef'];types=cfg['types'];by=defaultdict(lambda:defaultdict(list));cuts={}
for r in rows:
    i=types.index(r['type']);eta=cf[2*i+r['high']]+cf[6]*r['signal']+cf[7+i]*(r['regime']-.5)+cf[10]*(r['peer']-3.5)
    by[r['episode']][r['tick']].append((eta,r['y']));cuts[r['episode']]=r['cutoff']
eps=sorted(by);grid=[(s,h) for s in (.15,.3,.45,.6) for h in (.008,.02,.05)]
full=[];prefix=[];baseline=[]
for ep in eps:
    cut=cuts[ep];early={t:v for t,v in by[ep].items() if t<.6*cut}
    full.append([filter_episode((cut,by[ep]),s,h)[0] for s,h in grid]);prefix.append([filter_episode((cut,early),s,h)[0] for s,h in grid])
    baseline.append(filter_episode((cut,by[ep]),cfg['scale'],cfg['hazard'])[0]-filter_episode((cut,early),cfg['scale'],cfg['hazard'])[0])
full=np.array(full);prefix=np.array(prefix)
def weights(ll):
    p=np.full(ll.shape[1],1/ll.shape[1]);lik=np.exp(ll-ll.max(1)[:,None])
    for _ in range(150):
        post=lik*p;post/=post.sum(1)[:,None];p=(post.sum(0)+.5)/(len(ll)+.5*len(p))
    return p
def lse(v):
    m=max(v);return m+math.log(sum(math.exp(x-m) for x in v))
reports={}
for name,idx in [('amplitude',[j for j,(s,h) in enumerate(grid) if h==.02]),('amplitude_hazard',list(range(len(grid))))]:
    gains=[]
    for fold in range(3):
        train=np.array([i%3!=fold for i in range(len(eps))]);prior=weights(full[train][:,idx])
        for i in range(len(eps)):
            if i%3==fold:gains.append(lse(full[i,idx]+np.log(prior))-lse(prefix[i,idx]+np.log(prior))-baseline[i])
    p=weights(full[:,idx]);reports[name]={'total_predictive_gain':sum(gains),'mean':float(np.mean(gains)),'cluster_se':float(np.std(gains,ddof=1)/math.sqrt(len(gains))),'positive_episodes':int(sum(x>0 for x in gains)),'grid':[grid[i] for i in idx],'prior':p.tolist()}
print(json.dumps(reports,indent=2));Path('eda/ev79_weather_mixture.json').write_text(json.dumps(reports,indent=2))

