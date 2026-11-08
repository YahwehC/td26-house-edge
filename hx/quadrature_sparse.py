"""Integrate cell-rate uncertainty as well as missing win bits (pilot)."""
import math
from scipy.special import roots_jacobi
from he.model import WinMachine,betaln,logsumexp
from hx.forward_batch import forward_batch

class QuadratureMachine(WinMachine):
    def __init__(self,q,cfg,lp,**kwargs):
        super().__init__(q,cfg,lp,**kwargs)
        seq=q['seq'];c=q['cutoff']
        if len(seq)>=.6*c:return
        cp=cfg['cell_prior'][q['machine_type']];pri=[]
        for nm in ('hi','lo'):
            mu,slope,k=cp[nm];k*=cfg.get('kappa_scale',1);m=1/(1+math.exp(-(mu+slope*q['signal'])))
            pri.append((k*m,k*(1-m)))
        for a,b in self.pairs:
            counts=[[0,0],[0,0]]
            for t,v in seq.items():
                if t-a in seq and t-b in seq:counts[int((seq[t-a]>0)==(seq[t-b]>0))][int(v>0)]+=1
            logs=[];preds=[]
            for ori in (0,1):
                base=-math.log(2);grids=[]
                for z,(l,w) in enumerate(counts):
                    alpha,beta=pri[z^ori]
                    base+=betaln(alpha+w,beta+l)-betaln(alpha,beta)
                    nodes,weights=roots_jacobi(3,beta+l-1,alpha+w-1)
                    norm=sum(weights);grids.append([(float((x+1)/2),float(v/norm)) for x,v in zip(nodes,weights)])
                rows=[([p0,p1],w0*w1) for p0,w0 in grids[0] for p1,w1 in grids[1]]
                values=forward_batch(seq,c,(a,b),[rates for rates,weight in rows])
                for (rates,weight),(ll,pr) in zip(rows,values):
                    known=sum(w*math.log(rates[z])+l*math.log1p(-rates[z]) for z,(l,w) in enumerate(counts))
                    logs.append(base+math.log(weight)+ll-known);preds.append(pr)
            total=logsumexp(logs);self.lml_stable[(a,b)]=total
            self.pred_stable[(a,b)]=sum(math.exp(ll-total)*pr for ll,pr in zip(logs,preds))
