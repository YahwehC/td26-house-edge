"""Experimental global-state calibration from training history only."""
import sys,math,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0,'.')
from he.api import load_params
from he.model import load_split,DEFAULT_CFG,WinMachine,_pair_prior,betaln,logsumexp
from he.joint import SizeEvidence,lag_prior_marginal,lag_prior_for_pair,episode_posterior_joint

def state_rows(qs,p):
    cfg=dict(DEFAULT_CFG,**p['cfg']);lp=_pair_prior(cfg);marg=lag_prior_marginal(cfg);by=defaultdict(list)
    for q in qs:by[q['episode_id']].append(q)
    rows=[];next_rows={}
    for ep,ms in by.items():
        models=[];sizes=[]
        short=sum(len(q['seq']) for q in ms)<.6*sum(q['cutoff'] for q in ms)
        for q in ms:
            se=SizeEvidence(q,p['size_vectors'],cfg);f={pa:se.factor_and_r1(lag_prior_for_pair(pa,cfg,marg),cfg)[0] for pa in lp}
            lps={pa:lp[pa]+f[pa] for pa in lp};norm=logsumexp(list(lps.values()))
            models.append(WinMachine(q,cfg,{pa:v-norm for pa,v in lps.items()},stable_only=short));sizes.append(se)
        _,diag=episode_posterior_joint(models,cfg);stable=diag['post'].get('stable',0)>.5
        for q,m,se,post in zip(ms,models,sizes,diag['pair_post']):
            seq=q['seq'];cut=q['cutoff'];pa=max(post,key=post.get);a,b=pa;tau=cut+1
            if not stable and m.taus:
                k=max(range(len(m.taus)),key=lambda i:m.lml_tau[pa][i])
                if m.lml_tau[pa][k]-math.log(len(m.taus))>m.lml_stable[pa]:tau=m.taus[k]
            lag=max(lag_prior_for_pair(pa,cfg,marg),key=lag_prior_for_pair(pa,cfg,marg).get)
            sl=[r[0] for r in se.lag[lag]];s1=math.exp(sl[1]-logsumexp(sl))
            counts=[[0,0],[0,0]]
            for t,val in seq.items():
                if t-a in seq and t-b in seq:counts[int((seq[t-a]>0)==(seq[t-b]>0))^int(t>=tau)][int(val>0)]+=1
            cp=cfg['cell_prior'][q['machine_type']];pri=[]
            for nm in ('hi','lo'):
                mu,slope,k=cp[nm];prob=1/(1+math.exp(-(mu+slope*q['signal'])));pri.append((k*prob,k*(1-prob)))
            ll=[sum(betaln(pri[z^ori][0]+w,pri[z^ori][1]+l)-betaln(*pri[z^ori]) for z,(l,w) in enumerate(counts)) for ori in (0,1)]
            ori=int(ll[1]>ll[0])
            for t in list(sorted(seq))+[cut]:
                if t<8 or t-a not in seq or t-b not in seq or t-lag not in seq:continue
                z=int((seq[t-a]>0)==(seq[t-b]>0))^int(t>=tau);high=int((z^ori)==0)
                bit=int(seq[t-lag]>=2);r=(1-s1)*bit+s1*(1-bit)
                other=sum(other['seq'].get(t-3,0)>0 for other in ms if other is not q)
                row={'episode':ep,'machine':q['machine_id'],'type':q['machine_type'],'signal':q['signal'],'high':high,'regime':r,'peer':other,'tick':t,'y':int(seq.get(t,0)>0),'cutoff':cut}
                if t==cut:next_rows[q['query_id']]=row
                else:rows.append(row)
    return rows,next_rows

if __name__=='__main__':
    p=load_params('runs/main_v8/params.json');q,e=load_split('kit/public','train');rows,_=state_rows(q,p)
    types=['balanced','jackpot','steady'];X=[];Y=[]
    for r in rows:
        x=[0.]*11;i=types.index(r['type']);x[2*i+r['high']]=1;x[6]=r['signal'];x[7+i]=r['regime']-.5;x[10]=r['peer']-3.5
        X.append(x);Y.append(r['y'])
    X=np.array(X);Y=np.array(Y)
    def loss(v):
        eta=X@v;prob=1/(1+np.exp(-eta))
        return float(np.sum(np.logaddexp(0,eta)-Y*eta)+.001*(v@v)),X.T@(prob-Y)+.002*v
    init=np.array([p['cfg']['cell_prior'][ty][nm][0] for ty in types for nm in ('lo','hi')]+[.3,-.8,-1,-.5,.09])
    result=minimize(loss,init,jac=True,method='BFGS',options={'gtol':1e-5,'maxiter':200})
    config={'types':types,'coef':result.x.tolist()};print(config,flush=True)
    Path('eda/ev63_global.json').write_text(json.dumps(config,indent=2))
    import pickle
    pickle.dump(rows,open('eda/_state_rows.pkl','wb'))
