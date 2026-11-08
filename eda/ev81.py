"""Common high-low logit gap across types: constrained episode cross-fit."""
import sys,json,copy
from pathlib import Path
sys.path.insert(0,'.')
from he.api import load_params,predict_queries,write_submission
from he.model import load_split,official_score
# Same fixed calibration as ev68, but a shared high-low gap removes two coefficients.
import ev68
source=Path('eda/ev68.py').read_text(encoding='utf-8');body=source[source.index('def calibrate'):source.index("if __name__=='__main__':")]
body=body.replace('design=np.column_stack([X,signs[g]])','''mapping=np.zeros((12,10))
        for i in range(3):
            mapping[2*i,i]=1;mapping[2*i+1,i]=1;mapping[2*i+1,3]=1
        for i in range(6,12):mapping[i,i-2]=1
        full_design=np.column_stack([X,signs[g]])
        design=full_design@mapping''')
body=body.replace("penalty=np.eye(12)*.1\n        fitted=np.linalg.solve(design.T@design+penalty,design.T@Y+penalty@prior)","penalty=np.eye(12)*.1\n        reduced=np.linalg.solve(design.T@design+mapping.T@penalty@mapping,design.T@Y+mapping.T@penalty@prior)\n        fitted=mapping@reduced")
exec(body,ev68.__dict__)
p=load_params('runs/main_v11/params.json');q,e=load_split('kit/public','validation');eps=sorted({r['episode_id'] for r in q});initial=json.load(open('eda/ev64_global_hmm.json'));pred={};configs=[]
print('train gaps', [initial['coef'][2*i+1]-initial['coef'][2*i] for i in range(3)],flush=True)
for fold in range(3):
    selected={ep for i,ep in enumerate(eps) if i%3!=fold};config=ev68.calibrate(q,e,initial,selected,p);config['machine_variance']=.04;configs.append(config)
    pp=copy.deepcopy(p);pp['global_state']=config;held=[r for r in q if r['episode_id'] not in selected]
    pred.update(predict_queries(held,pp));print('fold',fold,'done',flush=True)
score=official_score(pred,e,q);print(score,flush=True);Path('eda/ev81_common_gap.json').write_text(json.dumps({'score':score,'configs':configs},indent=2));write_submission('eda/ev81_predictions.csv',pred)
