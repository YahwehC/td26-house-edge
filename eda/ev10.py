import sys, math, collections
sys.path.insert(0,'.')
import numpy as np, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from he.model import *
from he.model import _pair_prior
cfg=dict(DEFAULT_CFG); plp=_pair_prior(cfg)
qt,et=load_split('kit/public','train'); cond={r['query_id']:r['condition'] for r in et['conditions']}
rows=[];cols=[];vals=[];Y=[];X2=[];TY=[]
r=0
for mi,q in enumerate(qt):
    if cond[q['query_id']]!='stable': continue
    m=WinMachine(q,cfg,plp); A,B=max(m.pairs,key=lambda p:m.lml_stable[p]); s=q['seq']
    for t in sorted(s):
        if t<8 or any((t-L) not in s for L in range(1,9)): continue
        z=int((s[t-A]>0)==(s[t-B]>0))
        rows.append(r); cols.append(2*mi+z); vals.append(1.0)
        f=[float(s[t-L]>0) for L in range(1,9)]+[float(s[t-L]>=2) for L in range(1,9)]+[float(s[t-L]>=5) for L in range(1,9)]
        X2.append(f); Y.append(int(s[t]>0)); TY.append(q['machine_type']); r+=1
F=sp.csr_matrix((vals,(rows,cols)),shape=(r,2*len(qt)))
X2=np.array(X2); Y=np.array(Y); TY=np.array(TY)
for ty in ['balanced','jackpot','steady']:
    k=TY==ty
    base=LogisticRegression(C=100,max_iter=2000).fit(F[k],Y[k])
    full=LogisticRegression(C=100,max_iter=2000).fit(sp.hstack([F[k],sp.csr_matrix(X2[k])]).tocsr(),Y[k])
    ll0=-np.sum(np.log(np.where(Y[k]==1,base.predict_proba(F[k])[:,1],base.predict_proba(F[k])[:,0])))
    P=full.predict_proba(sp.hstack([F[k],sp.csr_matrix(X2[k])]).tocsr())[:,1]
    ll1=-np.sum(np.log(np.where(Y[k]==1,P,1-P)))
    co=full.coef_[0][-24:]
    print(ty,'n',k.sum(),'NLL base %.1f full %.1f'%(ll0,ll1)); print('  win',co[:8].round(2)); print('  ge2',co[8:16].round(2)); print('  ge5',co[16:].round(2))
