"""Exact missing payout-bit recursion under global state/size logits."""
import math
from .sparse_joint import topology
from .model import SIZE_IDX, _pair_prior, DEFAULT_CFG, logsumexp
from .joint import lag_prior_marginal, lag_prior_for_pair


def predict(q,peers,params,config,hot,offset=0.):
    cfg=dict(DEFAULT_CFG,**params['cfg']);pri=_pair_prior(cfg);marg=lag_prior_marginal(cfg);cf=config['coef'];i=config['types'].index(q['machine_type']);scale=config['scale']
    V=params['size_vectors'][q['machine_type']];eps=cfg['size_eps'];V=[[(1-eps)*V[r][k]+eps*V[1-r][k] for k in range(4)] for r in (0,1)]
    seq=q['seq'];cut=q['cutoff'];logs=[];predictions=[]
    for (a,b),lp in pri.items():
        lag=max(lag_prior_for_pair((a,b),cfg,marg),key=lag_prior_for_pair((a,b),cfg,marg).get)
        cells,regimes,dest=topology(a,b,lag);n=len(cells)
        for orientation in (0,1):
            for sign in (0,1):
                def probabilities(peer,weather):
                    rates=[[0.,0.],[0.,0.]]
                    for z in (0,1):
                        for r in (0,1):
                            eta=cf[2*i+(z^orientation)]+cf[6]*q['signal']+cf[7+i]*(r-.5)+cf[10]*(peer-3.5)+offset
                            rates[z][r]=(1-weather)/(1+math.exp(-(eta-scale)))+weather/(1+math.exp(-(eta+scale)))
                    return [rates[z][r^sign] for z,r in zip(cells,regimes)]
                sizes=[V[r^sign] for r in regimes]
                def advance(state,win,value):
                    out=[0.]*n
                    if value is None:
                        for j,mass in enumerate(state):
                            if mass:
                                w=win[j];v=sizes[j];out[dest[0][j]]+=mass*(1-w);out[dest[1][j]]+=mass*w*v[0];out[dest[2][j]]+=mass*w*(1-v[0])
                    else:
                        category=0 if value==0 else (1 if value==1 else 2);k=SIZE_IDX.get(value)
                        for j,mass in enumerate(state):
                            if mass:out[dest[category][j]]+=mass*((1-win[j]) if value==0 else win[j]*sizes[j][k])
                    return out
                state=[1/n]*n;stationary=probabilities(3.5,.5)
                for _ in range(256):
                    nxt=advance(state,stationary,None);diff=max(abs(x-y) for x,y in zip(nxt,state));state=nxt
                    if diff<1e-11:break
                loglike=lp-math.log(4);cache={peer:probabilities(peer,.5) for peer in set(peers.values())}
                for t in range(cut):
                    value=seq.get(t);state=advance(state,cache[peers[t]],value)
                    if value is not None:
                        norm=sum(state);loglike+=math.log(norm);state=[v/norm for v in state]
                win=probabilities(peers[cut],hot)
                pr=[sum(mass*(1-w) for mass,w in zip(state,win))]+[sum(mass*w*v[k] for mass,w,v in zip(state,win,sizes)) for k in range(4)]
                logs.append(loglike);predictions.append(pr)
    norm=logsumexp(logs)
    return [sum(math.exp(ll-norm)*pr[k] for ll,pr in zip(logs,predictions)) for k in range(5)]
