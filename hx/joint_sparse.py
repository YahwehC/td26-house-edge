"""Joint missing win/size forward model pilot; NumPy only for exploration."""
import math
import numpy as np
from functools import lru_cache
from he.model import SIZE_IDX,betaln,logsumexp,_pair_prior,DEFAULT_CFG

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
    return np.array(cells),np.array(reg),[np.array(x) for x in dest]

def forward(seq,c,pair,lag,rates,vectors,sign):
    cell,reg,dest=topology(*pair,lag)
    win=np.array(rates)[cell]; sz=np.array(vectors)[reg^sign]
    weights=[1-win,win*sz[:,0],win*(1-sz[:,0])]
    n=len(cell); state=np.full(n,1/n)
    def transition(state):
        return sum(np.bincount(d,weights=state*p,minlength=n) for d,p in zip(dest,weights))
    for _ in range(256):
        nxt=transition(state);diff=np.max(np.abs(nxt-state));state=nxt
        if diff<1e-11:break
    ll=0
    for t in range(c):
        v=seq.get(t)
        if v is None:state=transition(state)
        else:
            if v==0:wt=weights[0];d=dest[0]
            else:wt=win*sz[:,SIZE_IDX[v]];d=dest[1 if v==1 else 2]
            state=np.bincount(d,weights=state*wt,minlength=n)
            z=float(state.sum());ll+=math.log(z);state/=z
    pred=[float(state@(1-win))]+[float(state@(win*sz[:,k])) for k in range(4)]
    return ll,pred

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
            from hx.sparse import forward as win_forward
            winll,wp=win_forward(seq,c,(a,b),rates)
            for r in (0,1):
                ll=sum(math.log(vn[r][SIZE_IDX[val]]) for val in seq.values() if val>0)
                logs.append(base-known+winll+ll+math.log(pc));preds.append([1-wp]+[wp*x for x in vn[r]])
    total=logsumexp(logs)
    return [sum(math.exp(ll-total)*pr[k] for ll,pr in zip(logs,preds)) for k in range(5)]
