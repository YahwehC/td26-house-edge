import sys, math, collections
sys.path.insert(0,'.')
import numpy as np
from he.model import *
from he.model import _pair_prior
cfg=dict(DEFAULT_CFG); plp=_pair_prior(cfg)
rows=[]
for split in ['train','validation']:
    qs,ex=load_split('kit/public',split); cond={r['query_id']:r['condition'] for r in ex['conditions']}
    for q in qs:
        if cond[q['query_id']]!='stable': continue
        m=WinMachine(q,cfg,plp); p=max(m.pairs,key=lambda p:m.lml_stable[p])
        s=q['seq']; A,B=p; n=collections.Counter()
        for t in s:
            if t-A in s and t-B in s: n[((s[t-A]>0)==(s[t-B]>0), s[t]>0)]+=1
        r1=(n[(True,True)]+.5)/(n[(True,True)]+n[(True,False)]+1); r0=(n[(False,True)]+.5)/(n[(False,True)]+n[(False,False)]+1)
        hi,lo=max(r1,r0),min(r1,r0)
        rows.append((q['machine_type'],q['signal'],hi,lo,r1>r0, n[(True,True)]+n[(True,False)], n[(False,True)]+n[(False,False)]))
lg=lambda p: math.log(p/(1-p))
for ty in ['balanced','jackpot','steady']:
    R=[r for r in rows if r[0]==ty]
    for nm,ix in (('hi',2),('lo',3)):
        x=np.array([lg(r[ix]) for r in R]); s=np.array([r[1] for r in R])
        b=np.polyfit(s,x,1)
        print(ty,nm,'n',len(R),'logit mean %.2f sd %.2f  sig slope %.2f resid sd %.2f'%(x.mean(),x.std(),b[0],(x-np.polyval(b,s)).std()), 'quantiles',np.percentile(x,[5,25,50,75,95]).round(2))
    print(ty,'frac XNOR=hi',np.mean([r[4] for r in R]).round(2))
