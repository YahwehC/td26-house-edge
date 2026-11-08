"""Adaptation OOF calibration pooling released main and adaptation targets.
Fixed hypothesis: non-temporal type/signal/size/coupling coefficients shared across
rounds; lag-pair priors remain adaptation-specific. Held adaptation episode excluded.
"""
import sys,json,copy
from pathlib import Path
sys.path.insert(0,'.')
from he.api import load_params,predict_queries,write_submission,save_params
from he.model import load_split,official_score
from ev68 import calibrate
p=load_params('runs/r2_v1/params.json');qs,e=load_split('kit2_x/public','validation');main,me=load_split('kit/public','validation');eps=sorted({q['episode_id'] for q in qs});main_eps={q['episode_id'] for q in main};assert not main_eps.intersection(eps)
allq=main+qs;targets={'targets':me['targets']+e['targets']};pred={};configs=[];weight=json.load(open('eda/r2_blend_train.json'))['full_weight']
for fold in range(3):
    selected=main_eps|{ep for i,ep in enumerate(eps) if i%3!=fold};config=calibrate(allq,targets,p['global_state'],selected,p);configs.append(config);candidate=copy.deepcopy(p);candidate['global_state']=dict(config,machine_variance=.04);candidate['blend']['w_main']=weight
    held=[q for q in qs if eps.index(q['episode_id'])%3==fold];pred.update(predict_queries(held,candidate));print('fold',fold,'finished',flush=True)
score=official_score(pred,e,qs,20.);report={'score':score,'configs':configs,'main_weight':weight,'machine_variance':.04,'protocol':'All main validation targets plus two-thirds adaptation target episodes; held adaptation episode targets excluded; existing base size vectors unchanged'}
Path('eda/r2_pooled_calibration.json').write_text(json.dumps(report,indent=2));write_submission('eda/r2_pooled_predictions.csv',pred)
p['global_state']=dict(calibrate(allq,targets,p['global_state'],main_eps|set(eps),p),machine_variance=.04);p['blend']['w_main']=weight
p['global_state_fit']={'method':'Joint 11-coefficient calibration + common binary weather amplitude on released main/adaptation targets; history EM initialization','source':'main and adaptation released validation targets; reported adaptation predictions exclude the held adaptation episode','evaluation':'eda/r2_pooled_calibration.json','machine_variance':'0.04; eda/r2_offset_train.json','blend':'adaptation training original-episode CV; eda/r2_blend_train.json'}
if weight==1.:p.pop('blend',None);p.pop('fallback',None)
save_params('eda/r2_pooled_params.json',p);print(score,flush=True)
