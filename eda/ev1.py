import sys, time, json
sys.path.insert(0,'.')
from he.model import *
t0=time.time()
qv,ev=load_split('kit/public','validation')
vecs=fit_size_vectors(qv,ev['targets'])
print({k:[[round(x,3) for x in v] for v in vv] for k,vv in vecs.items()})
cfg=json.loads(sys.argv[1]) if len(sys.argv)>1 else {}
pred=predict(qv,vecs,cfg)
print(official_score(pred,ev,qv), 'time %.1fs'%(time.time()-t0))
