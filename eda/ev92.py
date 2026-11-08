"""Stable train prefix/tail check: peer-weather smoothing for offset estimation.
Only prefix peer observations may enter a prefix offset. Held tail weather is causal.
Structural labels still come from full training histories: component diagnostic.
"""
import sys,json,pickle,math
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,'.')
from he.model import load_split
from he.machine_offset import estimate_offsets
rows=pickle.load(open('eda/_state_rows.pkl','rb'));qs,ex=load_split('kit/public','train');cm={r['query_id']:r['condition'] for r in ex['conditions']};stable={q['machine_id'] for q in qs if cm[q['query_id']]=='stable'}
cfg=json.load(open('eda/ev64_global_hmm.json'));cfg['machine_variance']=.04;cf=cfg['coef'];scale=cfg['scale'];hazard=cfg['hazard'];types=cfg['types']
rows=[r for r in rows if r['machine'] in stable];prefix=[r for r in rows if r['tick']<.6*r['cutoff']]
base=estimate_offsets(prefix,cfg);alternative=estimate_offsets(prefix,dict(cfg,offset_smoothing=True))
by=defaultdict(lambda:defaultdict(list))
for r in rows:
    i=types.index(r['type']);eta=cf[2*i+r['high']]+cf[6]*r['signal']+cf[7+i]*(r['regime']-.5)+cf[10]*(r['peer']-3.5)
    by[r['episode']][r['tick']].append((r,eta))
sig=lambda z:1/(1+math.exp(-z));loss={name:defaultdict(list) for name in ('baseline','smoothed')}
for ep,times in by.items():
    mids={r['machine'] for obs in times.values() for r,eta in obs}
    for mid in mids:
        hot=.5;last=None
        for tick,obs in sorted(times.items()):
            if last is not None:hot=.5+(hot-.5)*(1-2*hazard)**(tick-last)
            ll=[0.,0.]
            for r,eta in obs:
                if r['machine']==mid:
                    if tick>=.6*r['cutoff']:
                        for name,offsets in [('baseline',base),('smoothed',alternative)]:
                            d=offsets[ep,mid];p=(1-hot)*sig(eta+d-scale)+hot*sig(eta+d+scale);loss[name][ep].append((p-r['y'])**2)
                    continue
                for j,shift in enumerate((-scale,scale)):
                    z=eta+shift;ll[j]+=r['y']*z-max(z,0)-math.log1p(math.exp(-abs(z)))
            hot=sig(math.log(max(hot,1e-15)/max(1-hot,1e-15))+ll[1]-ll[0]);last=tick
means={name:{ep:sum(v)/len(v) for ep,v in groups.items()} for name,groups in loss.items()}
d=[means['smoothed'][ep]-means['baseline'][ep] for ep in means['baseline']];m=sum(d)/len(d)
report={'delta':m,'cluster_se':math.sqrt(sum((v-m)**2 for v in d)/(len(d)-1)/len(d)),'improved':sum(v<0 for v in d),'episodes':len(d),'means':means}
print({k:v for k,v in report.items() if k!='means'},flush=True);Path('eda/ev92_offset_smoothing.json').write_text(json.dumps(report,indent=2))
