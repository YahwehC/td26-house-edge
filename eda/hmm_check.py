import sys, math, collections
sys.path.insert(0,'.')
import numpy as np
from he.model import load_split, WinMachine, _pair_prior, DEFAULT_CFG
cfg=dict(DEFAULT_CFG); plp=_pair_prior(cfg)
R=collections.defaultdict(list)  # gap -> list of (resid_prev, resid_next)
for split in ['train','validation']:
    qs,ex=load_split('kit/public',split); cond={r['query_id']:r['condition'] for r in ex['conditions']}
    for q in qs:
        if cond[q['query_id']]!='stable': continue
        m=WinMachine(q,cfg,plp); A,B=max(m.pairs,key=lambda p:m.lml_stable[p]); s=q['seq']
        T=sorted(t for t in s if t-A in s and t-B in s)
        z={t:int((s[t-A]>0)==(s[t-B]>0)) for t in T}; y={t:int(s[t]>0) for t in T}
        mean={c:np.mean([y[t] for t in T if z[t]==c] or [0.5]) for c in (0,1)}
        res={t:y[t]-mean[z[t]] for t in T}
        for i,t in enumerate(T):
            for g in (1,2,3,5,10,20):
                u=t+g
                if u in res: R[g].append((res[t],res[u]))
for g in sorted(R):
    a=np.array(R[g]); print('gap',g,'n',len(a),'corr of residuals %.4f'%np.corrcoef(a[:,0],a[:,1])[0,1])
