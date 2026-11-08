"""Sparse-history win likelihood: integrate hidden lag bits (standard library).
Uses posterior-mean cell rates for the additional likelihood, retaining the
Beta-binomial integral for fully observed contexts. No EM/imputed counts.

Enabled explicitly by cfg['sparse_win']; dense histories retain the original
model. The hidden-bit forward pass is exact at fixed rates. Using posterior
mean rates and a numerically approximated stationary initial distribution is
an approximation to the full Bayesian model, not an exact Beta-HMM integral.
"""
import math
from he.model import WinMachine, betaln, logsumexp


def forward(seq, cutoff, pair, rates):
    a,b=pair; size=1<<b; mask=size-1
    probs=[rates[int(((s>>(a-1))&1)==((s>>(b-1))&1))] for s in range(size)]
    nxt=[(s<<1)&mask for s in range(size)]
    state=[1/size]*size
    # Stationary initial distribution, before the first observed tick.
    for _ in range(256):
        out=[0.0]*size
        for s,v in enumerate(state):
            p=probs[s]; j=nxt[s]; out[j]+=v*(1-p); out[j+1]+=v*p
        diff=max(abs(x-y) for x,y in zip(out,state)); state=out
        if diff<1e-11: break
    ll=0.0
    for t in range(cutoff):
        observed=seq.get(t)
        out=[0.0]*size
        for s,v in enumerate(state):
            if not v: continue
            p=probs[s]; j=nxt[s]
            if observed is None or observed==0: out[j]+=v*(1-p)
            if observed is None or observed>0: out[j+1]+=v*p
        z=sum(out)
        if observed is not None:
            ll+=math.log(z)
            state=[v/z for v in out]
        else: state=out
    prediction=sum(v*p for v,p in zip(state,probs))
    return ll,prediction


class SparseMachine(WinMachine):
    """Augment pair/orientation evidence using formerly discarded observations.

    For each orientation, multiply the existing Beta-binomial likelihood by
    P(all observed wins | fixed rates) / P(complete-context wins | fixed rates).
    The denominator prevents counting the already-used observations twice.
    Updated pair evidence also informs the existing conditional size model.
    """
    def __init__(self,q,cfg,pair_logprior,**kwargs):
        super().__init__(q,cfg,pair_logprior,**kwargs)
        seq=q['seq']; c=q['cutoff']
        if len(seq)>=.6*c: return
        cp=cfg['cell_prior'][q['machine_type']]
        pri=[]
        for name in ('hi','lo'):
            mu,slope,k=cp[name]; k*=cfg.get('kappa_scale',1)
            mean=1/(1+math.exp(-(mu+slope*q['signal'])))
            pri.append((k*mean,k*(1-mean)))
        for pair in self.pairs:
            a,b=pair; counts=[[0,0],[0,0]]
            for t,value in seq.items():
                if t-a in seq and t-b in seq:
                    z=int((seq[t-a]>0)==(seq[t-b]>0)); counts[z][int(value>0)]+=1
            logs=[]; preds=[]
            for orient in (0,1):
                rates=[]; base=-math.log(2); known=0
                for z,(l,w) in enumerate(counts):
                    alpha,beta=pri[z^orient]
                    rate=(alpha+w)/(alpha+beta+l+w); rates.append(rate)
                    base+=betaln(alpha+w,beta+l)-betaln(alpha,beta)
                    known+=w*math.log(rate)+l*math.log1p(-rate)
                full,pred=forward(seq,c,pair,rates)
                logs.append(base+full-known); preds.append(pred)
            total=logsumexp(logs)
            self.lml_stable[pair]=total
            self.pred_stable[pair]=sum(math.exp(ll-total)*pr for ll,pr in zip(logs,preds))
