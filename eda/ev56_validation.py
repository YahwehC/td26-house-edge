"""Single validation check after frozen sparse candidate training comparison."""
import sys,json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,'.')
import he.joint as joint
from hx.sparse import SparseMachine
from he.api import load_params,predict_queries
from he.model import load_split,official_score
p=load_params('runs/main_v6/params.json');qs,ex=load_split('kit/public','validation');report={}
for name,cls in [('base',joint.WinMachine),('sparse',SparseMachine)]:
    with patch.object(joint,'WinMachine',cls):pred=predict_queries(qs,p)
    report[name]=official_score(pred,ex,qs);print(name,report[name],flush=True)
Path('eda/ev56_validation.json').write_text(json.dumps(report,indent=2))
