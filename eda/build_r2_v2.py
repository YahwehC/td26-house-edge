"""Freeze pooled calibration and train-selected adaptation inference settings."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,'.')
from he.api import load_params,save_params,predict_split,write_submission
from check_csv import check
start=time.perf_counter()
p=load_params('eda/r2_pooled_params.json')
out=Path('runs/r2_v2');out.mkdir(exist_ok=True)
save_params(out/'params.json',p)
report=json.load(open('eda/r2_pooled_calibration.json'))
(out/'validation_score.json').write_text(json.dumps(report['score'],indent=2))
(out/'validation_predictions.csv').write_bytes(Path('eda/r2_pooled_predictions.csv').read_bytes())
q,e,pred=predict_split('kit2_x/public','test',p)
write_submission(out/'submission.csv',pred)
errors=check(out/'submission.csv','kit2_x/public/test/queries.csv')
assert not errors,errors
print('r2_v2 test checked',len(pred),'seconds',time.perf_counter()-start,flush=True)
