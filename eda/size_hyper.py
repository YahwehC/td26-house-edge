"""Fit size insurance/noise parameters by train-history marginal likelihood."""
import math,copy
from scipy.optimize import minimize
from he.model import DEFAULT_CFG,WinMachine,_pair_prior,logsumexp,MAXLAG,SIZE_IDX
from he.joint import SizeEvidence,lag_prior_for_pair,lag_prior_marginal

def fit(splits,p):
    cfg=dict(DEFAULT_CFG,**p['cfg']); vecs=p['size_vectors']; plp=_pair_prior(cfg)
    rows=[]
    for qs,cm in splits:
        for q in qs:
            if cm.get(q['query_id'])!='stable':continue
            wm=WinMachine(q,cfg,plp,stable_only=True)
            known=[]; seq=q['seq']; v=vecs[q['machine_type']]
            for lag in range(1,MAXLAG+1):
                cnt=[[0]*4 for _ in range(2)]
                miss=[0]*4
                for t,value in seq.items():
                    if value>0:
                        k=SIZE_IDX[value]
                        if t-lag in seq:cnt[int(seq[t-lag]>=2)][k]+=1
                        else:miss[k]+=1
                known.append((cnt,miss))
            pb=(sum(x>=2 for x in seq.values())+1)/(len(seq)+2)
            total=[sum(x==pay for x in seq.values()) for pay in (1,2,5,20)]
            rows.append((v,known,pb,total,[plp[pair]+wm.lml_stable[pair] for pair in plp]))
    marginal=lag_prior_marginal(cfg)
    rule=p['lag_rule']['rule']
    def config(x):
        eps,pc,delta=x
        c=copy.deepcopy(p['cfg']); c.update(size_eps=float(eps),size_const=float(pc))
        if rule!='none':
            c['size_lag_given_pair']={f'{a},{b}':{str(L):delta*marginal.get(L,0)+(1-delta if L==(a if rule=='first' else b) else 0) for L in range(1,MAXLAG+1)} for a,b in plp}
        return c
    def objective(x):
        eps,pc,delta=x; out=0
        for v,known,pb,total,winll in rows:
            vn=[[(1-eps)*v[r][k]+eps*v[1-r][k] for k in range(4)] for r in (0,1)]
            lv=[[math.log(z) for z in row] for row in vn]
            constant=logsumexp([math.log(pc)+sum(n*ll for n,ll in zip(total,lv[r])) for r in (0,1)])
            lagll=[]
            for cnt,miss in known:
                lagll.append(logsumexp([sum(cnt[b][k]*lv[b^s][k] for b in (0,1) for k in range(4))+sum(miss[k]*math.log((1-pb)*vn[s][k]+pb*vn[1-s][k]) for k in range(4))-math.log(2) for s in (0,1)]))
            ll=[]
            for (a,b),wll in zip(plp,winll):
                weights={L:delta*marginal.get(L,0)+(1-delta if L==(a if rule=='first' else b) else 0) for L in range(1,MAXLAG+1)} if rule!='none' else marginal
                mixed=logsumexp([math.log(w)+lagll[L-1] for L,w in weights.items() if w>0])
                ll.append(wll+logsumexp([constant,math.log1p(-2*pc)+mixed]))
            out-=logsumexp(ll)
        return out
    start=[cfg['size_eps'],cfg['size_const'],.01]
    opt=minimize(objective,start,method='Nelder-Mead',bounds=[(0,.2),(1e-9,.2),(0,.2)],options={'maxiter':600,'xatol':1e-6,'fatol':1e-6})
    print('size hyper',opt.x,'LL gain',objective(start)-opt.fun,'converged',opt.success,flush=True)
    return config(opt.x)
