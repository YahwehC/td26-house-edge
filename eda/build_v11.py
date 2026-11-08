"""Freeze training-selected persistent-offset variant and build its test CSV."""
import sys,json
from pathlib import Path
sys.path.insert(0,'.')
from he.api import load_params,save_params,predict_split,write_submission
from check_csv import check
p=load_params('runs/main_v10/params.json');p['global_state']['machine_variance']=.04
p['global_state_fit']['machine_variance']='0.04; chronological train-history comparison in eda/ev77.py'
out=Path('runs/main_v11');out.mkdir(exist_ok=True);save_params(out/'params.json',p)
(out/'validation_score.json').write_bytes(Path('eda/ev78_validation.json').read_bytes());(out/'validation_predictions.csv').write_bytes(Path('eda/ev78_predictions.csv').read_bytes())
q,e,pr=predict_split('kit/public','test',p);write_submission(out/'submission.csv',pr)
errors=check(out/'submission.csv','kit/public/test/queries.csv');assert not errors,errors
print('main_v11 test checked',len(pr),flush=True)
