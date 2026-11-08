"""Fit a latent episode state on training histories; validate once."""
import sys,json
from pathlib import Path
sys.path.insert(0,'.')
from he.api import load_params,write_submission
from he.model import load_split,read_csv,official_score
from he.weather import traces,filter_episode,apply
p=load_params('runs/main_v8/params.json');effects=json.load(open('eda/ev60_size_win.json'))
q,e=load_split('kit/public','train');data=traces(q,p,effects)
rows=[]
for scale in (0.,.2,.4,.6,.8):
    for hazard in (.005,.02,.08,.25,.5):
        if scale==0 and hazard!=.5:continue
        value=sum(filter_episode(tr,scale,hazard)[0] for tr in data.values())
        rows.append({'scale':scale,'hazard':hazard,'loglik':value})
rows.sort(key=lambda r:-r['loglik']);print(rows[:8],flush=True)
config=rows[0];q,e=load_split('kit/public','validation')
base={r['query_id']:[float(r['p_'+str(k)]) for k in (0,1,2,5,20)] for r in read_csv('eda/ev61_predictions.csv')}
pr=apply(q,base,p,effects,config);score=official_score(pr,e,q);print(score,flush=True)
Path('eda/ev62_weather.json').write_text(json.dumps({'train_search':rows,'config':config,'validation':score},indent=2))
write_submission('eda/ev62_predictions.csv',pr)
