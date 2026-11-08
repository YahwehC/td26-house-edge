"""Adaptation train diagnostic: persistent offset variance, chronological split.
Latent state labels inferred from full histories, so this is a component diagnostic.
"""
import sys,json,math,time
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,'.')
from he.api import load_params
from he.model import load_split
from he.global_state import state_rows
from he.machine_offset import estimate_offsets
p=load_params('runs/r2_v1/params.json');cfg=p['global_state'];qs,e=load_split('kit2_x/public','train');cm={r['query_id']:r['condition'] for r in e['conditions']};stable={q['machine_id'] for q in qs if cm[q['query_id']]=='stable'}
rows,_=state_rows([q for q in qs if q['machine_id'] in stable],p);prefix=[r for r in rows if r['tick']<.6*r['cutoff']];cf=cfg['coef'];types=cfg['types'];scale=cfg['scale'];hazard=cfg['hazard']
by=defaultdict(lambda:defaultdict(list))
for r in rows:
    i=types.index(r['type']);eta=cf[2*i+r['high']]+cf[6]*r['signal']+cf[7+i]*(r['regime']-.5)+cf[10]*(r['peer']-3.5);by[r['episode']][r['tick']].append((r,eta))
sig=lambda z:1/(1+math.exp(-z));records=defaultdict(list)
for ep,times in by.items():
    for mid in {r['machine'] for obs in times.values() for r,_ in obs}:
        hot=.5;last=None
        for t,obs in sorted(times.items()):
            if last is not None:hot=.5+(hot-.5)*(1-2*hazard)**(t-last)
            ll=[0.,0.]
            for r,eta in obs:
                if r['machine']==mid:
                    if t>=.6*r['cutoff']:records[ep,mid].append((eta,hot,r['y']))
                    continue
                for j,shift in enumerate((-scale,scale)):
                    z=eta+shift;ll[j]+=r['y']*z-max(z,0)-math.log1p(math.exp(-abs(z)))
            hot=sig(math.log(max(hot,1e-15)/max(1-hot,1e-15))+ll[1]-ll[0]);last=t
report=[]
for variance in (0.,.04,.09):
    offsets=estimate_offsets(prefix,dict(cfg,machine_variance=variance));loss=defaultdict(list)
    for key,records_m in records.items():
        offset=offsets.get(key,0.)
        for eta,h,y in records_m:
            probability=(1-h)*sig(eta+offset-scale)+h*sig(eta+offset+scale);loss[key[0]].append((probability-y)**2)
    means={ep:sum(v)/len(v) for ep,v in loss.items()};report.append({'variance':variance,'brier':sum(means.values())/len(means),'means':means})
for r in report:
    d=[r['means'][ep]-report[0]['means'][ep] for ep in r['means']];m=sum(d)/len(d);r.update(delta=m,cluster_se=math.sqrt(sum((x-m)**2 for x in d)/(len(d)-1)/len(d)),improved=sum(x<0 for x in d));print({k:v for k,v in r.items() if k!='means'},flush=True)
Path('eda/r2_offset_train.json').write_text(json.dumps(report,indent=2))
