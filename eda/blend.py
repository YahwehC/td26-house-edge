import sys
sys.path.insert(0,'.')
from he.api import fit_params, predict_split
from he.model import official_score
p=fit_params('kit/public',fit_splits=('train',))
qv,ev,pred=predict_split('kit/public','validation',p)
ref={r['query_id']:[float(r[k]) for k in ('p_0','p_1','p_2','p_5','p_20')] for r in ev['reference']}
for e in (0,0.01,0.03,0.1):
    bl={q:[(1-e)*a+e*b for a,b in zip(v,ref[q])] for q,v in pred.items()}
    s=official_score(bl,ev,qv); print(e, round(s['points'],3), s['ratios'])
