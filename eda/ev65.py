import sys,json
from pathlib import Path
sys.path.insert(0,'.')
from he.api import load_params,write_submission
from he.model import load_split,read_csv,official_score
from he.global_state import predict
p=load_params('runs/main_v8/params.json');q,e=load_split('kit/public','validation')
base={r['query_id']:[float(r['p_'+str(k)]) for k in (0,1,2,5,20)] for r in read_csv('runs/main_v8/validation_predictions.csv')}
config=json.load(open('eda/ev64_global_hmm.json'));pr=predict(q,base,p,config);score=official_score(pr,e,q)
print(score,flush=True);Path('eda/ev65_validation.json').write_text(json.dumps(score,indent=2));write_submission('eda/ev65_predictions.csv',pr)
