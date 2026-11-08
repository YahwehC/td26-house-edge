"""Episode-held-out supervised calibration, no validation-set parameter sweep.
Fits only an interpretable 11-term logit model and a binary episode shift;
the held-out episode targets never enter its fit. Training-history checkpoint
provides initialization and state thresholds. All predictions use histories.
"""
import sys,json,math,copy
import numpy as np
from pathlib import Path
sys.path.insert(0,'.')
from he.api import load_params,predict_queries,write_submission
from he.model import load_split,official_score

def calibrate(qs,ex,initial,selected,params):
    types=initial['types'];coef=np.array(initial['coef']);qm={q['query_id']:q for q in qs};X=[];Y=[];groups=[]
    for target in ex['targets']:
        q=qm[target['query_id']]
        if q['episode_id'] not in selected:continue
        win=1-float(target['p_0']);logit=math.log(win/(1-win));i=types.index(q['machine_type']);V=params['size_vectors'][q['machine_type']]
        regime=int(float(target['p_1'])/win>(V[0][0]+V[1][0])/2)
        state=int(logit>(coef[2*i]+coef[2*i+1])/2+coef[6]*q['signal']+coef[7+i]*(regime-.5))
        other=[m for m in qs if m['episode_id']==q['episode_id'] and m is not q]
        peer=sum(m['seq'].get(q['cutoff']-3,0)>0 for m in other)
        row=[0.]*11;row[2*i+state]=1;row[6]=q['signal'];row[7+i]=regime-.5;row[10]=peer-3.5
        X.append(row);Y.append(logit);groups.append(q['episode_id'])
    X=np.array(X);Y=np.array(Y);groupids={e:i for i,e in enumerate(sorted(set(groups)))};g=np.array([groupids[e] for e in groups]);scale=initial['scale']
    for _ in range(20):
        residual=Y-X@coef
        means=np.bincount(g,weights=residual)/np.bincount(g)
        signs=np.where(means>=0,1.,-1.)
        design=np.column_stack([X,signs[g]])
        prior=np.r_[initial['coef'],initial['scale']]
        penalty=np.eye(12)*.1
        fitted=np.linalg.solve(design.T@design+penalty,design.T@Y+penalty@prior)
        coef=fitted[:11];scale=abs(float(fitted[11]))
    cfg=dict(initial,coef=coef.tolist(),scale=scale)
    return cfg

if __name__=='__main__':
    p=load_params('runs/main_v9/params.json');q,e=load_split('kit/public','validation');eps=sorted({r['episode_id'] for r in q});prediction={};configs=[]
    for fold in range(3):
        train={ep for i,ep in enumerate(eps) if i%3!=fold};held=[r for r in q if r['episode_id'] not in train]
        config=calibrate(q,e,p['global_state'],train,p);configs.append(config);pp=copy.deepcopy(p);pp['global_state']=config
        prediction.update(predict_queries(held,pp));print('fold',fold,'finished',config,flush=True)
    score=official_score(prediction,e,q);print(score,flush=True)
    Path('eda/ev68_crossfit.json').write_text(json.dumps({'score':score,'configs':configs},indent=2));write_submission('eda/ev68_predictions.csv',prediction)
