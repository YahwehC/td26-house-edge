"""Package the frozen history-fitted joint-state candidate."""
import sys,json
from pathlib import Path
sys.path.insert(0,'.')
from he.api import load_params,save_params,predict_split,write_submission
from he.model import official_score
from check_csv import check
name=sys.argv[1] if len(sys.argv)>1 else 'main_v9'
p=load_params('runs/main_v8/params.json');p['cfg'].pop('sparse_joint',None)
p['global_state']=json.load(open('eda/ev64_global_hmm.json'))
if '--exact' in sys.argv:p['cfg']['exact_sparse_glm']=True
out=Path('runs')/name;out.mkdir(exist_ok=True);save_params(out/'params.json',p)
q,e,pr=predict_split('kit/public','validation',p);score=official_score(pr,e,q);print(score,flush=True)
(out/'validation_score.json').write_text(json.dumps(score,indent=2));write_submission(out/'validation_predictions.csv',pr)
q,e,pr=predict_split('kit/public','test',p);write_submission(out/'submission.csv',pr)
errors=check(out/'submission.csv','kit/public/test/queries.csv');assert not errors,errors
print(name,'test checked',len(pr),flush=True)
