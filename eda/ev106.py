"""Single OOF validation of train-screened cavity weather, short histories only."""
import sys,json,csv,copy
from pathlib import Path
sys.path.insert(0,'.')
from he.api import load_params,write_submission
from he.model import load_split,official_score
from weather_cavity_numpy import predict
p=load_params('runs/main_v13/params.json');qs,e=load_split('kit/public','validation');eps=sorted({q['episode_id'] for q in qs});configs=json.load(open('eda/ev68_crossfit.json'))['configs'];columns=('p_0','p_1','p_2','p_5','p_20')
pred={r['query_id']:[float(r[k]) for k in columns] for r in csv.DictReader(open('runs/main_v13/validation_predictions.csv'))}
for fold,config in enumerate(configs):
    candidate=copy.deepcopy(p);candidate['global_state']=dict(config,machine_variance=.04)
    held=[q for q in qs if eps.index(q['episode_id'])%3==fold and len(q['seq'])<.6*q['cutoff']]
    pred.update(predict(held,candidate));print('fold',fold,'finished',flush=True)
score=official_score(pred,e,qs);print(score,flush=True);Path('eda/ev106_validation.json').write_text(json.dumps(score,indent=2));write_submission('eda/ev106_predictions.csv',pred)
