"""Vectorized fixed-rate forward passes for EDA quadrature only."""
import numpy as np

def forward_batch(seq,cutoff,pair,rate_rows):
    a,b=pair;n=1<<b;half=n//2
    cells=np.array([int(((s>>(a-1))&1)==((s>>(b-1))&1)) for s in range(n)])
    probabilities=np.asarray(rate_rows)[:,cells]
    state=np.full(probabilities.shape,1/n)
    def advance(state,observation):
        out=np.zeros_like(state)
        if observation is None or observation==0:
            losses=state*(1-probabilities);out[:,::2]=losses[:,:half]+losses[:,half:]
        if observation is None or observation>0:
            wins=state*probabilities;out[:,1::2]=wins[:,:half]+wins[:,half:]
        return out
    active=np.ones(len(rate_rows),dtype=bool)
    for _ in range(256):
        nxt=advance(state,None);diff=np.max(np.abs(nxt-state),axis=1)
        state[active]=nxt[active];active&=diff>=1e-11
        if not active.any():break
    ll=np.zeros(len(rate_rows))
    for t in range(cutoff):
        value=seq.get(t);state=advance(state,value)
        if value is not None:
            z=state.sum(axis=1);ll+=np.log(z);state/=z[:,None]
    return list(zip(ll.tolist(),(state*probabilities).sum(axis=1).tolist()))
