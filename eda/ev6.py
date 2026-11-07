import sys, math, json, collections
sys.path.insert(0,'.')
import numpy as np
from he.model import *
qv,ev=load_split('kit/public','validation')
vecs=fit_size_vectors(qv,ev['targets'])
pred=predict(qv,vecs,{})
cond={r['query_id']:r['condition'] for r in ev['conditions']}
tg={r['query_id']:[float(r[k]) for k in ('p_0','p_1','p_2','p_5','p_20')] for r in ev['targets']}
eps=collections.defaultdict(list)
for q in qv: eps[q['episode_id']].append(q)
lg=lambda p: math.log(p/(1-p)); sg=lambda x:1/(1+math.exp(-x))
LAGS=[int(x) for x in sys.argv[1].split(',')] if len(sys.argv)>1 else [3,6]
rows=[]
for q in qv:
    c=q['cutoff']; w=min(max(1-pred[q['query_id']][0],1e-4),1-1e-4); tw=min(max(1-tg[q['query_id']][0],1e-4),1-1e-4)
    f=[np.mean([x['seq'].get(c-L,0)>0 for x in eps[q['episode_id']] if x is not q]) for L in LAGS]
    rows.append((q,w,tw,f))
# fit beta minimizing squared error in prob space via simple grid/gradient on fold A, test on fold B
epl=sorted(eps); fold={e:i%2 for i,e in enumerate(epl)}
def apply(b,r):
    q,w,tw,f=r; return sg(lg(w)+sum(bi*(fi-0.5) for bi,fi in zip(b,f)))
def loss(b,rs): return sum((apply(b,r)-r[2])**2 for r in rs)
from scipy.optimize import minimize
newp={}
for k in (0,1):
    tr=[r for r in rows if fold[r[0]['episode_id']]!=k]; te=[r for r in rows if fold[r[0]['episode_id']]==k]
    b=minimize(lambda b:loss(b,tr),np.zeros(len(LAGS))).x; print('fold',k,'beta',b.round(3))
    for r in te:
        q=r[0]; p=pred[q['query_id']]; w=1-p[0]; nw=apply(b,r); newp[q['query_id']]=[1-nw]+[x/w*nw for x in p[1:]]
print('base',official_score(pred,ev,qv)); print('adj ',official_score(newp,ev,qv))
