import sys, time, json
sys.path.insert(0,'.')
from he.model import *
from he.fit import fit_cell_prior
qt,et=load_split('kit/public','train'); qv,ev=load_split('kit/public','validation')
cp=fit_cell_prior([(qt,{r['query_id']:r['condition'] for r in et['conditions']})])
print({k:{n:[round(x,2) for x in v] for n,v in d.items()} for k,d in cp.items()})
vecs=fit_size_vectors(qv,ev['targets'])
cfg=json.loads(sys.argv[1]) if len(sys.argv)>1 else {}
t0=time.time()
print(official_score(predict(qv,vecs,dict(cfg,cell_prior=cp)),ev,qv),'%.1fs'%(time.time()-t0))
