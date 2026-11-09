"""Research joint filtering of hidden payout categories and persistent weather.

Peer emissions come from fixed history-inferred machine states, excluding own
outcomes. Own lag-rule uncertainty and both hidden processes are integrated jointly.
Missing peer wins still use empirical expectations; this is not an exact 8-machine
joint model. No validation-fitted parameters or new hyperparameters are introduced.
"""
import math
import numpy as np
from collections import defaultdict
from he.model import DEFAULT_CFG, SIZE_IDX, _pair_prior, logsumexp
from he.joint import lag_prior_marginal, lag_prior_for_pair
from he.sparse_joint import topology
from he.global_state import state_rows
from he.machine_offset import estimate_offsets
from he.peer import peer_winners


def forward(seq,cutoff,pair,lag,orientation,sign,vectors,base,levels,size_beta,
            peer_beta,peers,scale,hazard,peer_emissions):
    cells,regimes,dest=topology(*pair,lag)
    cells,regimes,dest=np.array(cells),np.array(regimes),np.array(dest)
    n=len(cells);sizes=vectors[regimes^sign]
    eta=np.take(levels,cells^orientation)+base+size_beta*((regimes^sign)-.5)
    def rates(peer):return 1/(1+np.exp(-(eta[None,:]+peer_beta*(peer-3.5)+np.array([-scale,scale])[:,None])))
    def transition(state,win,value):
        out=np.zeros_like(state)
        for weather in (0,1):
            mass=state[weather];w=win[weather]
            if value is None:
                out[weather]=(np.bincount(dest[0],weights=mass*(1-w),minlength=n)
                              +np.bincount(dest[1],weights=mass*w*sizes[:,0],minlength=n)
                              +np.bincount(dest[2],weights=mass*w*(1-sizes[:,0]),minlength=n))
            else:
                category=0 if value==0 else (1 if value==1 else 2)
                weight=1-w if value==0 else w*sizes[:,SIZE_IDX[value]]
                out[weather]=np.bincount(dest[category],weights=mass*weight,minlength=n)
        return out
    # State is joint P(weather_t, own lag bits before tick t).
    state=np.full((2,n),.5/n);stationary=rates(3.5)
    for _ in range(256):
        nxt=transition(state,stationary,None);nxt=(1-hazard)*nxt+hazard*nxt[::-1]
        difference=np.max(np.abs(nxt-state));state=nxt
        if difference<1e-11:break
    ll=0.;cache={peer:rates(peer) for peer in set(peers.values())}
    for tick in range(cutoff):
        state=transition(state,cache[peers[tick]],seq.get(tick))
        state*=np.asarray(peer_emissions.get(tick,(1.,1.)))[:,None]
        norm=state.sum();ll+=math.log(norm);state/=norm
        state=(1-hazard)*state+hazard*state[::-1]
    win=cache[peers[cutoff]]
    prediction=[float(np.sum(state*(1-win)))]+[float(np.sum(state*win*sizes[:,k])) for k in range(4)]
    return ll,prediction


def machine_predict(q,peers,params,config,offset,emissions):
    cfg=dict(DEFAULT_CFG,**params['cfg']);prior=_pair_prior(cfg);marginal=lag_prior_marginal(cfg)
    cf=config['coef'];i=config['types'].index(q['machine_type']);V=params['size_vectors'][q['machine_type']];eps=cfg['size_eps']
    vectors=np.array([[(1-eps)*V[r][k]+eps*V[1-r][k] for k in range(4)] for r in (0,1)])
    logs=[];predictions=[]
    for pair,lp in prior.items():
        lag_weights=lag_prior_for_pair(pair,cfg,marginal);lag=max(lag_weights,key=lag_weights.get)
        for orientation in (0,1):
            for sign in (0,1):
                ll,pred=forward(q['seq'],q['cutoff'],pair,lag,orientation,sign,vectors,
                                cf[6]*q['signal']+offset,cf[2*i:2*i+2],cf[7+i],cf[10],peers,
                                config['scale'],config['hazard'],emissions)
                logs.append(lp-math.log(4)+ll);predictions.append(pred)
    normal=logsumexp(logs)
    return [math.fsum(math.exp(ll-normal)*pred[k] for ll,pred in zip(logs,predictions)) for k in range(5)]


def predict(queries,params):
    config=params['global_state'];cf=config['coef'];types=config['types'];scale=config['scale']
    rows,_=state_rows(queries,params);offsets=estimate_offsets(rows,config)
    times=defaultdict(lambda:defaultdict(list));episodes=defaultdict(list)
    for r in rows:
        i=types.index(r['type']);eta=cf[2*i+r['high']]+cf[6]*r['signal']+cf[7+i]*(r['regime']-.5)+cf[10]*(r['peer']-3.5)+offsets.get((r['episode'],r['machine']),0.)
        ll=[]
        for shift in (-scale,scale):
            z=eta+shift;ll.append(r['y']*z-max(z,0)-math.log1p(math.exp(-abs(z))))
        times[r['episode']][r['tick']].append((r['machine'],ll))
    for q in queries:episodes[q['episode_id']].append(q)
    result={}
    for ep,qs in episodes.items():
        for q in qs:
            emissions={}
            for tick,records in times[ep].items():
                ll=[sum(row[state] for mid,row in records if mid!=q['machine_id']) for state in (0,1)]
                maximum=max(ll);emissions[tick]=[math.exp(value-maximum) for value in ll]
            peers={t:peer_winners(qs,q,t-int(params['cfg'].get('peer_lag',3)),True) for t in range(q['cutoff']+1)}
            result[q['query_id']]=machine_predict(q,peers,params,config,offsets.get((ep,q['machine_id']),0.),emissions)
    return result
