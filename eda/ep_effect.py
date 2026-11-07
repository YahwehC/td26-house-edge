import sys, math, collections
sys.path.insert(0,'.')
import numpy as np
from he.model import load_split, WinMachine, _pair_prior, DEFAULT_CFG
cfg=dict(DEFAULT_CFG); plp=_pair_prior(cfg)
lg=lambda p: math.log(p/(1-p))
rows=[]
for split in ['train','validation']:
    qs,ex=load_split('kit/public',split); cond={r['query_id']:r['condition'] for r in ex['conditions']}
    for q in qs:
        if cond[q['query_id']]!='stable': continue
        m=WinMachine(q,cfg,plp); A,B=max(m.pairs,key=lambda p:m.lml_stable[p]); s=q['seq']
        n=collections.Counter(((s[t-A]>0)==(s[t-B]>0), s[t]>0) for t in s if t-A in s and t-B in s)
        r=[(n[(z,True)]+.5)/(n[(z,True)]+n[(z,False)]+1) for z in (True,False)]
        rows.append((split+q['episode_id'],q['machine_type'],q['signal'],lg(max(r)),lg(min(r))))
import pandas as pd
d=pd.DataFrame(rows,columns=['ep','type','sig','hi','lo'])
for c in ('hi','lo'):
    d[c+'_r']=d[c]-d.groupby(['type','sig'])[c].transform('mean')
    # leave-one-out episode mean of residuals vs own residual
    g=d.groupby('ep')[c+'_r']; loo=(g.transform('sum')-d[c+'_r'])/(g.transform('count')-1)
    print(c,'corr(own resid, episode-mates mean resid) = %.3f'%np.corrcoef(d[c+'_r'],loo)[0,1], 'n',len(d))
