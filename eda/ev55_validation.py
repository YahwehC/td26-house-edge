"""Single validation check of cell-prior candidate after train episode CV."""
import sys,copy,json
from pathlib import Path
sys.path.insert(0,'.')
from he.api import load_params,predict_queries
from he.model import DEFAULT_CFG,load_split,official_score
from he.fit import fit_cell_prior
from eb_prior import fit
p=load_params('runs/main_v6/params.json'); tq,te=load_split('kit/public','train');q,e=load_split('kit/public','validation')
splits=[(tq,{r['query_id']:r['condition'] for r in te['conditions']})]
p['cfg']['cell_prior']=fit_cell_prior(splits,dict(DEFAULT_CFG,**p['cfg']))
candidate=copy.deepcopy(p);candidate['cfg']['cell_prior']=fit(splits,dict(DEFAULT_CFG,**p['cfg']))
report={}
for name,params in [('moments',p),('marginal',candidate)]:
    report[name]=official_score(predict_queries(q,params),e,q)
    print(name,report[name],flush=True)
Path('eda/ev55_validation.json').write_text(json.dumps(report,indent=2))
Path('eda/eb_train_params.json').write_text(json.dumps(candidate,indent=2))
