"""Adaptation round: fixed global-logit calibration with episode-held-out evaluation.
Same 11-term model used for main v10; no sweep of validation hyperparameters.
Reported validation predictions exclude that episode's targets from calibration.
Existing base size vectors and priors keep their previously documented provenance.
"""
import sys,json,copy,time
from pathlib import Path
sys.path.insert(0,'.')
from he.api import load_params,predict_queries,write_submission,save_params
from he.model import load_split,official_score
from ev68 import calibrate
p=load_params('runs/r2_v1/params.json');qs,e=load_split('kit2_x/public','validation');eps=sorted({q['episode_id'] for q in qs});pred={};configs=[];fold_scores=[];start=time.perf_counter()
assert p['cfg'].get('peer_lag',3)==3
for fold in range(3):
    selected={ep for i,ep in enumerate(eps) if i%3!=fold};held=[q for q in qs if q['episode_id'] not in selected]
    config=calibrate(qs,e,p['global_state'],selected,p);configs.append(config);candidate=copy.deepcopy(p);candidate['global_state']=config
    pr=predict_queries(held,candidate);pred.update(pr);print('fold',fold,'done',time.perf_counter()-start,flush=True)
score=official_score(pred,e,qs,20.);report={'score':score,'configs':configs,'protocol':'3-fold episode-held-out global calibration, frozen 11 logits plus binary shared amplitude; existing base priors/size vectors unchanged'}
Path('eda/r2_calibration.json').write_text(json.dumps(report,indent=2));write_submission('eda/r2_calibration_predictions.csv',pred)
p['global_state']=calibrate(qs,e,p['global_state'],set(eps),p);p['global_state_fit']={'method':'r2g.py train-history GLM/HMM followed by fixed 11-coefficient released-target calibration','source':'adaptation validation targets; reported score uses 3 episode folds; test checkpoint refits all released target episodes','evaluation':'eda/r2_calibration.json'}
save_params('eda/r2_calibrated_params.json',p);print(report,flush=True)
