"""Refit fixed global-state calibration on released labels, package test forecast.
The reported validation prediction is the episode-held-out result from ev68.
"""
import sys,json,copy
from pathlib import Path
sys.path.insert(0,'.')
from he.api import load_params,save_params,predict_split,write_submission
from he.model import load_split
from check_csv import check
from ev68 import calibrate
p=load_params('runs/main_v9/params.json');q,e=load_split('kit/public','validation')
p['global_state']=calibrate(q,e,p['global_state'],{r['episode_id'] for r in q},p)
p['global_state_fit']={'method':'fixed linear logit plus binary episode offset','source':'released validation targets','evaluation':'3-fold episode-held-out; eda/ev68_crossfit.json','history_initialization':'train outcomes; eda/ev64.py'}
out=Path('runs/main_v10');out.mkdir(exist_ok=True);save_params(out/'params.json',p)
report=json.load(open('eda/ev68_crossfit.json'));(out/'validation_score.json').write_text(json.dumps(report['score'],indent=2))
(out/'validation_predictions.csv').write_bytes(Path('eda/ev68_predictions.csv').read_bytes())
q,e,pr=predict_split('kit/public','test',p);write_submission(out/'submission.csv',pr)
errors=check(out/'submission.csv','kit/public/test/queries.csv');assert not errors,errors
print('main_v10 test checked',len(pr),flush=True)
