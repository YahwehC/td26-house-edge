"""Marginal-likelihood cell prior pilot (SciPy is experiment-only)."""
import math
from collections import defaultdict
from scipy.optimize import minimize
from he.model import _pair_prior, betaln, logsumexp


def fit(splits, cfg):
    pairs = _pair_prior(cfg)
    rows = defaultdict(list)
    for qs, conditions in splits:
        for q in qs:
            if conditions.get(q['query_id']) != 'stable': continue
            seq = q['seq']; counts = []
            for a,b in pairs:
                cells = [[0,0],[0,0]]
                for t,v in seq.items():
                    if t-a in seq and t-b in seq:
                        z = int((seq[t-a]>0) == (seq[t-b]>0))
                        cells[z][int(v>0)] += 1
                counts.append(cells)
            rows[q['machine_type']].append((q['signal'], counts))
    result = {}
    for ty, data in sorted(rows.items()):
        old = cfg['cell_prior'][ty]
        start = [x for nm in ('hi','lo') for x in (*old[nm][:2], math.log(old[nm][2]))]
        def objective(x):
            pri = {}
            for signal in {r[0] for r in data}:
                beta=[]
                for offset in (0,3):
                    mu = 1/(1+math.exp(-(x[offset]+x[offset+1]*signal)))
                    k=math.exp(x[offset+2]); a=k*mu; b=k*(1-mu)
                    beta.append((a,b,betaln(a,b)))
                pri[signal]=beta
            loss=0
            for signal, counts in data:
                hypotheses=[]
                for lp,cells in zip(pairs.values(),counts):
                    for orient in (0,1):
                        ll=lp-math.log(2)
                        for z,(l,w) in enumerate(cells):
                            a,b,norm=pri[signal][z^orient]
                            ll += betaln(a+w,b+l)-norm
                        hypotheses.append(ll)
                loss-=logsumexp(hypotheses)
            return loss
        opt=minimize(objective,start,method='Nelder-Mead',bounds=[(-8,8),(-1,1),(math.log(2),math.log(500))]*2,
                     options={'maxiter':2000,'xatol':1e-5,'fatol':1e-7})
        result[ty]={nm:[*opt.x[i:i+2],math.exp(opt.x[i+2])] for nm,i in [('hi',0),('lo',3)]}
        print(ty, 'likelihood improvement', objective(start)-opt.fun, 'converged', opt.success, result[ty],flush=True)
    return result
