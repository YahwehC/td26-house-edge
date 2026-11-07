import sys, math, collections
sys.path.insert(0,'.')
import numpy as np
from he.model import *
from he.model import _pair_prior
from he.fit import fit_cell_prior
qt,et=load_split('kit/public','train'); qv,ev=load_split('kit/public','validation')
cp=fit_cell_prior([(qt,{r['query_id']:r['condition'] for r in et['conditions']})])
cfg=dict(DEFAULT_CFG,cell_prior=cp); plp=_pair_prior(cfg)
vecs=fit_size_vectors(qv,ev['targets']); pred=predict(qv,vecs,cfg)
cond={r['query_id']:r['condition'] for r in ev['conditions']}
tg={r['query_id']:float(r['p_0']) for r in ev['targets']}
lg=lambda p: math.log(max(p,1e-4)/(1-min(p,1-1e-4)))
cat=lambda v: 0 if v==0 else (1 if v==1 else (2 if v==2 else 3))
g=collections.defaultdict(list)
for q in qv:
    if cond[q['query_id']]!='stable': continue
    m=WinMachine(q,cfg,plp); A,B=max(m.pairs,key=lambda p:m.lml_stable[p]); s=q['seq']; c=q['cutoff']
    r=lg(1-tg[q['query_id']])-lg(1-pred[q['query_id']][0])
    g[(cat(s[c-A]),cat(s[c-B]))].append(r)
    g[('sumwins8',sum(s.get(c-L,0)>0 for L in range(1,9)))].append(r)
    g[('lastcat',cat(s[c-1]))].append(r)
for k in sorted(g,key=str): print(k,len(g[k]),round(np.mean(g[k]),2),round(np.std(g[k]),2))
