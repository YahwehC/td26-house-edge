import sys, math, json, collections
sys.path.insert(0,'.')
import numpy as np
from he.model import *
qv,ev=load_split('kit/public','validation')
vecs=fit_size_vectors(qv,ev['targets'])
pred=predict(qv,vecs,{})
cond={r['query_id']:r['condition'] for r in ev['conditions']}
tg={r['query_id']:[float(r[k]) for k in ('p_0','p_1','p_2','p_5','p_20')] for r in ev['targets']}
eps=collections.defaultdict(list)
for q in qv: eps[q['episode_id']].append(q)
lg=lambda p: math.log(p/(1-p))
R=[];F=collections.defaultdict(list)
for q in qv:
    if cond[q['query_id']] not in ('stable','drift','shared_change'): continue
    w=min(max(1-pred[q['query_id']][0],1e-4),1-1e-4); tw=min(max(1-tg[q['query_id']][0],1e-4),1-1e-4)
    R.append(lg(tw)-lg(w)); s=q['seq']; c=q['cutoff']
    F['signal'].append(q['signal'])
    for L in range(1,9):
        F[f'own_win{L}'].append(int(s.get(c-L,0)>0)); F[f'own_big{L}'].append(int(s.get(c-L,0)>=2))
        o=[x['seq'].get(c-L,0)>0 for x in eps[q['episode_id']] if x is not q]
        F[f'oth_win{L}'].append(np.mean(o))
        o=[x['seq'].get(c-L,0)>=5 for x in eps[q['episode_id']] if x is not q]
        F[f'oth_big{L}'].append(np.mean(o))
    F['own_wins8'].append(sum(s.get(c-L,0)>0 for L in range(1,9)))
    F['oth_wins8'].append(np.mean([sum(x['seq'].get(c-L,0)>0 for L in range(1,9)) for x in eps[q['episode_id']] if x is not q]))
    F['predw'].append(w)
R=np.array(R); print('resid sd',R.std().round(3),'mean',R.mean().round(3), 'n',len(R))
cs=sorted(((abs(np.corrcoef(R,np.array(v))[0,1]),k) for k,v in F.items()),reverse=True)[:12]
print([(k,round(c,3)) for c,k in cs])
print('--- neighbor offsets')
idx=lambda q:int(q['machine_id'][-3:])
res=[]
for d in range(1,8):
    for L in range(1,9):
        x=[];r=[];k=0
        for q in qv:
            if cond[q['query_id']] not in ('stable','drift','shared_change'): continue
            e=eps[q['episode_id']]; j=(idx(q)+d)%8; o=[z for z in e if idx(z)==j][0]
            x.append(int(o['seq'].get(q['cutoff']-L,0)>0)); r.append(R[k]); k+=1
        res.append((abs(np.corrcoef(r,x)[0,1]),d,L))
for r in sorted(res,reverse=True)[:10]: print(r)
# mean of others win at lag 3: regression slope
x=np.array(F['oth_win3']); print('slope',np.polyfit(x,R,1))
