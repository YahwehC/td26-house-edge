import sys, json, time, collections
sys.path.insert(0,'.')
from hx.api import fit_params
from hx.model import load_split, official_score
from hx.joint import predict_joint
p=fit_params('kit/public',fit_splits=('train',))
cfg=dict(p['cfg']); cfg.update(json.loads(sys.argv[1]) if len(sys.argv)>1 else {})
if False:
    cfg['size_lag_given_pair']={'1,3':{'1':1.0},'2,5':{'2':1.0},'4,7':{'4':1.0}}
qv,ev=load_split('kit/public','validation')
t=time.time(); pred=predict_joint(qv,p['size_vectors'],cfg); print(official_score(pred,ev,qv),'%.1fs'%(time.time()-t))
