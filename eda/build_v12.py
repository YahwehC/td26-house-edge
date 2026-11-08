"""Freeze the train-selected unblended global-state model and test artifact."""
import sys
from pathlib import Path
sys.path.insert(0,'.')
from he.api import load_params,save_params,predict_split,write_submission
from check_csv import check
p=load_params('runs/main_v11/params.json');p.pop('blend',None);p.pop('fallback',None)
p['global_state_fit']['blend']='Pure main model; episode-isolated train folds select 1.0, 0.99920035, 1.0; full fit 1.0 (eda/ev94.py supersedes ev85)'
out=Path('runs/main_v12');out.mkdir(exist_ok=True);save_params(out/'params.json',p)
(out/'validation_score.json').write_bytes(Path('eda/ev87_validation.json').read_bytes());(out/'validation_predictions.csv').write_bytes(Path('eda/ev87_predictions.csv').read_bytes())
q,e,pr=predict_split('kit/public','test',p);write_submission(out/'submission.csv',pr)
errors=check(out/'submission.csv','kit/public/test/queries.csv');assert not errors,errors
print('main_v12 test checked',len(pr),flush=True)
