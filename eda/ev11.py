import sys, time, json
sys.path.insert(0,'.')
from he.model import *
from he.fit import fit_cell_prior, fit_structure_priors
qt,et=load_split('kit/public','train'); qv,ev=load_split('kit/public','validation')
ct={r['query_id']:r['condition'] for r in et['conditions']}
vecs=fit_size_vectors(qv,ev['targets'])
sp=fit_structure_priors([(qt,ct)],vecs); print(sp['_votes'])
cp=fit_cell_prior([(qt,ct)])
over=json.loads(sys.argv[1]) if len(sys.argv)>1 else {}
cfg={k:v for k,v in sp.items() if not k.startswith('_')}; cfg.update(cell_prior=cp); cfg.update(over)
t0=time.time(); print(official_score(predict(qv,vecs,cfg),ev,qv),'%.1fs'%(time.time()-t0))
