"""Joint sparse-history win/size inference, standard library only.

Tracks win bits and big-payout bits together, preserving their dependence at
unobserved historical ticks. Rates use posterior means of complete-context
Beta cells. The dominant learned size lag is used per pair, plus the original
constant-size insurance components. This intentionally drops the tiny
cross-lag insurance mass and is enabled only by explicit configuration.
"""
import math
from functools import lru_cache
from .model import SIZE_IDX,betaln,logsumexp,_pair_prior,DEFAULT_CFG

@lru_cache(None)
def topology(a,b,lag):
    keys=[]
    for win in range(1<<b):
        big=win&((1<<lag)-1)
        sub=big
        while True:
            keys.append((win,sub))
            if sub==0:break
            sub=(sub-1)&big
    index={v:i for i,v in enumerate(keys)}
    dest=[[],[],[]];cells=[];reg=[]
    for win,big in keys:
        cells.append(int(((win>>(a-1))&1)==((win>>(b-1))&1)))
        reg.append((big>>(lag-1))&1)
        w=(win<<1)&((1<<b)-1);g=(big<<1)&((1<<lag)-1)
        for r,key in enumerate(((w,g),(w|1,g),(w|1,g|1))): dest[r].append(index[key])
    return cells,reg,dest

def forward(seq, cutoff, pair, lag, rates, vectors, sign):
    cell, regime, destinations = topology(*pair, lag)
    win = [rates[z] for z in cell]
    sizes = [vectors[r ^ sign] for r in regime]
    weights = ([1-p for p in win],
               [p*v[0] for p,v in zip(win,sizes)],
               [p*(1-v[0]) for p,v in zip(win,sizes)])
    n = len(cell)
    state = [1/n] * n

    def transition(state):
        out = [0.0] * n
        for dest, weight in zip(destinations, weights):
            for i, value in enumerate(state):
                out[dest[i]] += value * weight[i]
        return out

    for _ in range(256):
        nxt = transition(state)
        difference = max(abs(x-y) for x,y in zip(nxt,state))
        state = nxt
        if difference < 1e-11:
            break
    log_likelihood = 0.0
    for t in range(cutoff):
        value = seq.get(t)
        if value is None:
            state = transition(state)
        else:
            if value == 0:
                weight, dest = weights[0], destinations[0]
            else:
                k = SIZE_IDX[value]
                weight = [p*v[k] for p,v in zip(win,sizes)]
                dest = destinations[1 if value == 1 else 2]
            out = [0.0] * n
            for i, mass in enumerate(state):
                out[dest[i]] += mass * weight[i]
            total = sum(out)
            log_likelihood += math.log(total)
            state = [x/total for x in out]
    prediction = [sum(v*(1-p) for v,p in zip(state,win))]
    prediction += [sum(v*p*s[k] for v,p,s in zip(state,win,sizes)) for k in range(4)]
    return log_likelihood, prediction


def predict(q,vecs,cfg):
    cfg=dict(DEFAULT_CFG,**cfg);plp=_pair_prior(cfg)
    seq=q['seq'];c=q['cutoff'];cp=cfg['cell_prior'][q['machine_type']];pri=[]
    for nm in ('hi','lo'):
        mu,slope,k=cp[nm];k*=cfg.get('kappa_scale',1);m=1/(1+math.exp(-(mu+slope*q['signal'])))
        pri.append((k*m,k*(1-m)))
    v=vecs[q['machine_type']];eps=cfg['size_eps'];vn=[[(1-eps)*v[r][k]+eps*v[1-r][k] for k in range(4)] for r in (0,1)]
    logs=[];preds=[]
    for (a,b),lp in plp.items():
        counts=[[0,0],[0,0]]
        for t,val in seq.items():
            if t-a in seq and t-b in seq:counts[int((seq[t-a]>0)==(seq[t-b]>0))][int(val>0)]+=1
        row=cfg['size_lag_given_pair'][f'{a},{b}'];lag=int(max(row,key=row.get))
        for ori in (0,1):
            base=lp-math.log(2);known=0;rates=[]
            for z,(loss,w) in enumerate(counts):
                alpha,beta=pri[z^ori];rate=(alpha+w)/(alpha+beta+loss+w);rates.append(rate)
                base+=betaln(alpha+w,beta+loss)-betaln(alpha,beta)
                known+=w*math.log(rate)+loss*math.log1p(-rate)
            pc=cfg['size_const']
            for sign in (0,1):
                ll,pr=forward(seq,c,(a,b),lag,rates,vn,sign)
                logs.append(base-known+ll+math.log((1-2*pc)*.5));preds.append(pr)
            # Constant-size insurance: no latent big bit affects the emissions.
            from .sparse import forward as win_forward
            winll,wp=win_forward(seq,c,(a,b),rates)
            for r in (0,1):
                ll=sum(math.log(vn[r][SIZE_IDX[val]]) for val in seq.values() if val>0)
                logs.append(base-known+winll+ll+math.log(pc));preds.append([1-wp]+[wp*x for x in vn[r]])
    total=logsumexp(logs)
    return [sum(math.exp(ll-total)*pr[k] for ll,pr in zip(logs,preds)) for k in range(5)]
