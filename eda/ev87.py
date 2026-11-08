"""One held-episode validation of training-selected pure global model.
Recover already computed OOF main predictions from the known linear blend.
"""
import sys,json,csv
from pathlib import Path
sys.path.insert(0,'.')
from he.api import load_params,write_submission,predict_queries
from he.model import load_split,official_score
from he.fallback import predict_fallback
p=load_params('runs/main_v11/params.json');q,e=load_split('kit/public','validation')
old={r['query_id']:[float(r[k]) for k in ('p_0','p_1','p_2','p_5','p_20')] for r in csv.DictReader(open('eda/ev78_predictions.csv'))}
fb=predict_fallback(q,p['fallback']);pred={k:[(v-.1*b)/.9 for v,b in zip(row,fb[k])] for k,row in old.items()}
# Verify the algebra against an independent direct inference on a full held episode.
eps=sorted({r['episode_id'] for r in q});p['blend']['w_main']=1.;p['global_state']=dict(json.load(open('eda/ev68_crossfit.json'))['configs'][0],machine_variance=.04)
held=[r for r in q if r['episode_id']==eps[0]];direct=predict_queries(held,p)
maximum=max(abs(a-b) for k in direct for a,b in zip(direct[k],pred[k]));assert maximum<1e-12,maximum
s=official_score(pred,e,q);print('direct check',maximum,'score',s,flush=True)
Path('eda/ev87_validation.json').write_text(json.dumps(s,indent=2));write_submission('eda/ev87_predictions.csv',pred)
