import sys,json
from pathlib import Path
sys.path.insert(0,'.')
from he.api import load_params,write_submission
from he.model import load_split,official_score
from he.structured_state import predict
from he.fallback import predict_fallback
p=load_params('runs/main_v8/params.json');q,e=load_split('kit/public','validation');config=json.load(open('eda/ev64_global_hmm.json'));p['cfg']['exact_sparse_glm']=True
pr=predict(q,p,config);fb=predict_fallback(q,p['fallback']);w=p['blend']['w_main']
pr={k:[w*a+(1-w)*b for a,b in zip(row,fb[k])] for k,row in pr.items()}
s=official_score(pr,e,q);print(s,flush=True);Path('eda/ev67_validation.json').write_text(json.dumps(s,indent=2));write_submission('eda/ev67_predictions.csv',pr)

