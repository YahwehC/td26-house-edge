"""One validation check for joint hidden-win/size model after train evaluation."""
import sys,json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,'.')
import he.api as api
from he.model import load_split,official_score
from hx.joint_sparse import predict
p=api.load_params('runs/main_v7/params.json');q,e=load_split('kit/public','validation')
original=api.predict

def candidate(qs,vecs,cfg):
    dense=[q for q in qs if len(q['seq'])>=.6*q['cutoff']]
    result=original(dense,vecs,cfg)
    result.update({q['query_id']:predict(q,vecs,cfg) for q in qs if len(q['seq'])<.6*q['cutoff']})
    return result
with patch.object(api,'predict',candidate):pr=api.predict_queries(q,p)
s=official_score(pr,e,q);print(s,flush=True)
Path('eda/ev58_validation.json').write_text(json.dumps(s,indent=2))
