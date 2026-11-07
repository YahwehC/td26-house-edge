import sys, json
sys.path.insert(0,'.')
from hx.api import fit_params, _cond
from hx.model import load_split, official_score
from hx.joint import predict_joint
from hx.pool import fit_slopes, episode_offsets
P=fit_params('kit/public',fit_splits=('train',)); cfg=dict(P['cfg'])
qt,et=load_split('kit/public','train')
sl=fit_slopes([(qt,_cond(et))],cfg); print('slopes',sl)
qv,ev=load_split('kit/public','validation')
print('base',official_score(predict_joint(qv,P['size_vectors'],cfg),ev,qv)['points'])
for mult in (0.5,1.0):
  for km in (1.0,1.5,2.0):
    s2={k:v*mult for k,v in sl.items()}
    offs,_=episode_offsets(qv,cfg,s2)
    q2=[dict(q,_cell_offset=offs[q['query_id']]) for q in qv]
    s=official_score(predict_joint(q2,P['size_vectors'],dict(cfg,offset_kappa_mult=km)),ev,qv); print('slope x',mult,'kappa x',km,round(s['points'],3),s['ratios'])
