"""Marginalized orientation and size uncertainty in shared-weather emissions."""
import sys,json,copy
from pathlib import Path
sys.path.insert(0,'.')
from he.api import load_params,predict_queries,write_submission
from he.model import load_split,official_score
p=load_params('runs/main_v10/params.json');q,e=load_split('kit/public','validation');eps=sorted({r['episode_id'] for r in q});configs=json.load(open('eda/ev68_crossfit.json'))['configs'];pred={}
for fold,config in enumerate(configs):
    pp=copy.deepcopy(p);pp['global_state']=dict(config,soft_weather=True)
    held=[r for r in q if eps.index(r['episode_id'])%3==fold]
    pred.update(predict_queries(held,pp));print('fold',fold,'finished',flush=True)
s=official_score(pred,e,q);print(s,flush=True);Path('eda/ev74_validation.json').write_text(json.dumps(s,indent=2));write_submission('eda/ev74_predictions.csv',pred)

