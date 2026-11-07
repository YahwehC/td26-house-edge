import sys, math, collections
sys.path.insert(0,'.')
import pandas as pd
from hx.model import *
from hx.model import _pair_prior
from hx.api import fit_params
P=fit_params('kit/public',fit_splits=('train',)); cfg=dict(DEFAULT_CFG,**P['cfg']); plp=_pair_prior(cfg); vecs=P['size_vectors']
rows=[]
for split in ['train','validation']:
    qs,ex=load_split('kit/public',split); cond={r['query_id']:r['condition'] for r in ex['conditions']}
    for q in qs:
        if cond[q['query_id']]!='stable': continue
        m=WinMachine(q,cfg,plp); pr=max(m.pairs,key=lambda p:m.lml_stable[p]); A,B=pr; s=q['seq']
        n=collections.Counter((((s[t-A]>0)==(s[t-B]>0)), s[t]>0) for t in s if t-A in s and t-B in s)
        xnor_hi = n[(True,True)]/(1+n[(True,True)]+n[(True,False)]) > n[(False,True)]/(1+n[(False,True)]+n[(False,False)])
        LV=[[math.log(x) for x in v] for v in vecs[q['machine_type']]]
        wins=[(t,SIZE_IDX[v]) for t,v in s.items() if v>0]
        best=[]
        for L in (1,2,4):
            for sg in (0,1):
                best.append((math.fsum(LV[int(s[t-L]>=2)^sg][k] for t,k in wins if t-L in s),L,sg))
        best.sort(reverse=True)
        if best[0][0]-best[1][0]<3: continue
        rows.append(dict(split=split,ep=q['episode_id'],mid=int(q['machine_id'][-3:]),type=q['machine_type'],sig=q['signal'],pair=f'{pr[0]}{pr[1]}',xnor_hi=xnor_hi,L=best[0][1],s=best[0][2]))
d=pd.DataFrame(rows); print(len(d))
print(pd.crosstab(d.pair,d.L)); print(pd.crosstab(d.type,d.s)); print(pd.crosstab(d.sig,d.s)); print(pd.crosstab(d.xnor_hi,d.s)); print(pd.crosstab(d.L,d.s))
# within-episode agreement of L and s
g=d.groupby(['split','ep']).agg(nL=('L','nunique'),ns=('s','nunique'),n=('L','size'))
print(g[g.n>=4].describe())
