"""Frozen train-fitted size/win correction, first validation check."""
import sys,json
from pathlib import Path
sys.path.insert(0,'.')
from he.api import load_params,write_submission
from he.model import load_split,official_score,read_csv
from he.size_win import apply
p=load_params('runs/main_v8/params.json');qs,ex=load_split('kit/public','validation')
base={r['query_id']:[float(r['p_'+str(k)]) for k in (0,1,2,5,20)] for r in read_csv('runs/main_v8/validation_predictions.csv')}
effects=json.load(open('eda/ev60_size_win.json'));candidate=apply(qs,base,p,effects)
report={n:official_score(pr,ex,qs) for n,pr in [('baseline',base),('size_win',candidate)]}
print(json.dumps(report,indent=2),flush=True)
Path('eda/ev61_validation.json').write_text(json.dumps(report,indent=2))
write_submission('eda/ev61_predictions.csv',candidate)
