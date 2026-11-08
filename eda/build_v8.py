"""Build the fixed joint sparse candidate using unchanged main_v6 fitted parameters."""
import sys,json
from pathlib import Path
sys.path.insert(0,'.')
from he.api import load_params,save_params,predict_split,write_submission
from he.model import official_score
from check_csv import check
p=load_params('runs/main_v6/params.json');p['cfg']['sparse_joint']=True
out=Path('runs/main_v8');out.mkdir(exist_ok=True)
save_params(out/'params.json',p)
q,e,pr=predict_split('kit/public','validation',p)
s=official_score(pr,e,q);print(s,flush=True)
(out/'validation_score.json').write_text(json.dumps(s,indent=2))
write_submission(out/'validation_predictions.csv',pr)
q,e,pr=predict_split('kit/public','test',p);write_submission(out/'submission.csv',pr)
errors=check(out/'submission.csv','kit/public/test/queries.csv');assert not errors,errors
print('test CSV checked',len(pr),flush=True)

