"""Stable train machines: chronological holdout for persistent machine offsets.
Weather at every tick is inferred from OTHER machines and PAST observations only.
No validation outcomes or targets are used for selecting the variance.
"""
import sys,json,pickle,math
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,'.')
from he.model import load_split
from he.weather import filter_episode
rows=pickle.load(open('eda/_state_rows.pkl','rb'));qs,ex=load_split('kit/public','train');cm={r['query_id']:r['condition'] for r in ex['conditions']};stable={q['machine_id'] for q in qs if cm[q['query_id']]=='stable'}
config=json.load(open('eda/ev64_global_hmm.json'));cf=config['coef'];types=config['types'];scale=config['scale'];hazard=config['hazard'];by=defaultdict(lambda:defaultdict(list))
for r in rows:
    i=types.index(r['type']);eta=cf[2*i+r['high']]+cf[6]*r['signal']+cf[7+i]*(r['regime']-.5)+cf[10]*(r['peer']-3.5)
    by[r['episode']][r['tick']].append((r['machine'],eta,r['y'],r['cutoff'],r['high']))
sig=lambda x:1/(1+math.exp(-x));records=defaultdict(list)
for ep,times in by.items():
    mids={mid for obs in times.values() for mid,eta,y,c,high in obs if mid in stable}
    for mid in mids:
        hot=.5;last=None
        for tick,obs in sorted(times.items()):
            if last is not None:hot=.5+(hot-.5)*(1-2*hazard)**(tick-last)
            ll=[0.,0.]
            for machine,eta,y,c,high in obs:
                if machine==mid:records[mid].append((tick,eta,hot,y,c,ep,high));continue
                for j,shift in enumerate((-scale,scale)):
                    z=eta+shift;ll[j]+=y*z-max(z,0)-math.log1p(math.exp(-abs(z)))
            z=math.log(max(hot,1e-15)/max(1-hot,1e-15))+ll[1]-ll[0];hot=sig(z);last=tick
reports=[]
for rho in (1.,0.,.5):
    variance=.04;losses=defaultdict(list)
    for mid,records_m in records.items():
        train=[r for r in records_m if r[0]<r[4]*.6];held=[r for r in records_m if r[0]>=r[4]*.6]
        offset=[0.,0.]
        for _ in range(10):
            if rho==1:
                score=[-offset[0]/variance,0.];information=[1/variance,0.]
            else:
                precision=1/(variance*(1-rho*rho))
                score=[-precision*(offset[z]-rho*offset[1-z]) for z in (0,1)];information=[precision,precision]
            for t,eta,h,y,c,ep,high in train:
                z=0 if rho==1 else high
                p0,p1=sig(eta+offset[z]-scale),sig(eta+offset[z]+scale);p=(1-h)*p0+h*p1
                derivative=(1-h)*p0*(1-p0)+h*p1*(1-p1)
                score[z]+=(y-p)*derivative/(p*(1-p));information[z]+=derivative*derivative/(p*(1-p))
            if rho==1:offset[0]+=score[0]/information[0];offset[1]=offset[0]
            else:
                cross=-rho*precision;det=information[0]*information[1]-cross*cross
                offset=[offset[0]+(information[1]*score[0]-cross*score[1])/det,offset[1]+(information[0]*score[1]-cross*score[0])/det]
        for t,eta,h,y,c,ep,high in held:
            p=(1-h)*sig(eta+offset[high]-scale)+h*sig(eta+offset[high]+scale)
            losses[ep].append((p-y)**2)
    means={ep:sum(v)/len(v) for ep,v in losses.items()};reports.append({'rho':rho,'brier':sum(means.values())/len(means),'episodes':means})
base=reports[0]['episodes']
for report in reports:
    d=[report['episodes'][ep]-base[ep] for ep in base];mean=sum(d)/len(d);report['delta']=mean;report['cluster_se']=math.sqrt(sum((v-mean)**2 for v in d)/(len(d)-1)/len(d))
    print({k:v for k,v in report.items() if k!='episodes'},flush=True)
Path('eda/ev80_offsets_by_state.json').write_text(json.dumps(reports,indent=2))
