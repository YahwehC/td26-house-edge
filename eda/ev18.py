import sys, math, collections
sys.path.insert(0,'.')
import numpy as np, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from hx.model import *
from hx.model import _pair_prior
cfg=dict(DEFAULT_CFG); plp=_pair_prior(cfg)
rows=[];cols=[];X2=[];Y=[];TY=[]; r=0; mi=0
for split in ['train','validation']:
  qt,et=load_split('kit/public',split); cond={x['query_id']:x['condition'] for x in et['conditions']}
  for q in qt:
    if cond[q['query_id']]!='stable': continue
    m=WinMachine(q,cfg,plp); A,B=max(m.pairs,key=lambda p:m.lml_stable[p]); s=q['seq']
    n=collections.Counter((((s[t-A]>0)==(s[t-B]>0)), s[t]>0) for t in s if t-A in s and t-B in s)
    hiz = n[(True,True)]/(1+n[(True,True)]+n[(True,False)]) > n[(False,True)]/(1+n[(False,True)]+n[(False,False)])
    for t in sorted(s):
        if any((t-L) not in s for L in range(1,9)): continue
        z=((s[t-A]>0)==(s[t-B]>0)); hi=1.0 if z==hiz else -1.0
        rows.append(r); cols.append(2*mi+int(z)); r+=1
        f=[]
        for L in range(1,9):
            f+= [hi*(s[t-L]>=2), hi*(s[t-L]>=5), hi*(s[t-L]>0)]
        f+=[hi*(s[t-A]>=2), hi*(s[t-B]>=2), hi*q['signal']]
        X2.append(f); Y.append(int(s[t]>0)); TY.append(q['machine_type'])
    mi+=1
F=sp.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(r,2*mi)); X2=np.array(X2,float); Y=np.array(Y); TY=np.array(TY)
names=[f'{k}{L}' for L in range(1,9) for k in ('ge2_','ge5_','win_')]+['ge2_A','ge2_B','sig']
for ty in ['balanced','jackpot','steady']:
    k=TY==ty; Xf=sp.hstack([F[k],sp.csr_matrix(X2[k])]).tocsr()
    b=LogisticRegression(C=100,max_iter=3000).fit(F[k],Y[k]); f=LogisticRegression(C=100,max_iter=3000).fit(Xf,Y[k])
    nll=lambda mdl,X: -np.sum(np.log(np.clip(np.where(Y[k]==1,mdl.predict_proba(X)[:,1],mdl.predict_proba(X)[:,0]),1e-12,1)))
    co=f.coef_[0][-len(names):]; top=sorted(zip(np.abs(co),names,co),reverse=True)[:6]
    print(ty,'n',k.sum(),'dNLL %.1f (df %d)'%(nll(b,F[k])-nll(f,Xf),len(names)),[(n_,round(c,2)) for _,n_,c in top])
