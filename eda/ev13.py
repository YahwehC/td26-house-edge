import sys, time, json
sys.path.insert(0,'.')
from he.model import *
from he.fit import fit_cell_prior, fit_structure_priors, fit_tau_priors
qt,et=load_split('kit/public','train'); qv,ev=load_split('kit/public','validation')
ct={r['query_id']:r['condition'] for r in et['conditions']}
vecs=fit_size_vectors(qv,ev['targets']); sp=fit_structure_priors([(qt,ct)],vecs)
cfg=dict(DEFAULT_CFG,**{k:v for k,v in sp.items() if not k.startswith('_')}); cfg['cell_prior']=fit_cell_prior([(qt,ct)])
tp,sc=fit_tau_priors([(qt,ct)],cfg); print('short episodes by cond',sc)
over=json.loads(sys.argv[1]) if len(sys.argv)>1 else {}
cfg.update(tp); cfg.update(over)
t0=time.time(); print(official_score(predict(qv,vecs,cfg),ev,qv),'%.1fs'%(time.time()-t0))
