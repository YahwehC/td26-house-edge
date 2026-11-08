"""Independent test: does the latent payout-size regime also affect win odds?"""
import sys,math,json
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,'.')
from he.api import load_params
from he.model import load_split,SIZE_IDX
from he.coupling import _best_pair,_logreg,pairs_from_cfg
p=load_params('runs/main_v8/params.json');qs,ex=load_split('kit/public','train')
cm={r['query_id']:r['condition'] for r in ex['conditions']};groups=defaultdict(list)
for q in qs:groups[q['episode_id']].append(q)
rows=defaultdict(lambda:([],[],[]))
for ep,ms in groups.items():
    cond=cm[ms[0]['query_id']]
    if cond=='short_history':continue
    stop=min(q['cutoff'] for q in ms)-(0 if cond=='stable' else 110)
    for q in ms:
        seq=q['seq'];sep,(a,b),counts=_best_pair(seq,stop,pairs_from_cfg(p['cfg']))
        if sep<.3:continue
        vec=p['size_vectors'][q['machine_type']];logs=[[math.log(x) for x in v] for v in vec]
        lls=[0.,0.];contrib={};cellbig=[[0.,0.] for _ in range(2)]
        for t,val in seq.items():
            if t>=stop or t-a not in seq or t-b not in seq:continue
            z=int((seq[t-a]>0)==(seq[t-b]>0));big=int(seq[t-a]>=2)
            cellbig[z][0]+=big;cellbig[z][1]+=1
            if val:
                add=[logs[big^s][SIZE_IDX[val]] for s in (0,1)];contrib[t]=add
                for s in (0,1):lls[s]+=add[s]
        X,Y,O=rows[q['machine_type']]
        for t in range(20,stop):
            if t not in seq or t-a not in seq or t-b not in seq:continue
            diff=lls[1]-lls[0]-(contrib.get(t,[0,0])[1]-contrib.get(t,[0,0])[0])
            if abs(diff)<3:continue
            sign=int(diff>0);y=int(seq[t]>0);big=int(seq[t-a]>=2);r=big^sign
            z=int((seq[t-a]>0)==(seq[t-b]>0));n=cellbig[z][1]-1
            if n<=0:continue
            mean=(cellbig[z][0]-big)/n;mean=mean if sign==0 else 1-mean
            rate=(counts[z][1]-y+.5)/(sum(counts[z]))
            others=[m for m in ms if m is not q]
            X.append([r-mean,sum(m['seq'].get(t-3,0)>0 for m in others)])
            Y.append(y);O.append(math.log(rate/(1-rate)))
report={}
for ty,(X,Y,O) in rows.items():
    coef,se,mu=_logreg(X,Y,O,[0,1]);report[ty]={'beta':coef[1],'se':se[1],'z':coef[1]/se[1],'n':len(X),'peer_beta':coef[2]}
    print(ty,report[ty],flush=True)
Path('eda/ev60_size_win.json').write_text(json.dumps(report,indent=2))
