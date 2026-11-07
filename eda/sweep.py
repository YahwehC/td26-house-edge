import sys, json
sys.path.insert(0,'.')
from hx.api import fit_params
from hx.model import load_split, official_score
from hx.joint import predict_joint
p=fit_params('kit/public',fit_splits=('train',))
qv,ev=load_split('kit/public','validation')
eps=sorted({q['episode_id'] for q in qv}); half={e:i%2 for i,e in enumerate(eps)}
def sub(ex,h):
    keep={q['query_id'] for q in qv if half[q['episode_id']]==h}
    return {k:[r for r in v if r['query_id'] in keep] for k,v in ex.items()}
grid=[{}, {"size_eps":0.005},{"size_eps":0.05},{"size_const":0.01},{"size_const":0.15},{"kappa_scale":0.5},{"kappa_scale":2.0},
      {"drift_rho":0.97,"shared_rho":0.97},{"drift_rho":0.75,"shared_rho":0.75},{"model_prior":{"stable":0.34,"drift":0.33,"shared":0.33}},
      {"model_prior_short":{"stable":0.9,"drift":0.05,"shared":0.05}}]
for g in grid:
    cfg=dict(p['cfg'],**g); pred=predict_joint(qv,p['size_vectors'],cfg)
    s=official_score(pred,ev,qv); h=[official_score(pred,sub(ev,i),qv)['points'] for i in (0,1)]
    print(json.dumps(g)[:70].ljust(72),'%.3f  halves %.3f %.3f'%(s['points'],*h), s['ratios'])
