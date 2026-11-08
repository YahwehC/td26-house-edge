"""One OOF check of adaptation train-selected offsets and fallback weight."""
import sys,json,copy,time
from pathlib import Path
sys.path.insert(0,'.')
from he.api import load_params,predict_queries,write_submission,save_params
from he.model import load_split,official_score
p=load_params('eda/r2_calibrated_params.json');qs,e=load_split('kit2_x/public','validation');eps=sorted({q['episode_id'] for q in qs});cal=json.load(open('eda/r2_calibration.json'));train=json.load(open('eda/r2_blend_train.json'));weight=train['full_weight'];pred={}
for fold,config in enumerate(cal['configs']):
    candidate=copy.deepcopy(p);candidate['global_state']=dict(config,machine_variance=.04);candidate['blend']['w_main']=weight
    held=[q for q in qs if eps.index(q['episode_id'])%3==fold];pred.update(predict_queries(held,candidate));print('fold',fold,'finished',flush=True)
score=official_score(pred,e,qs,20.);report={'score':score,'main_weight':weight,'machine_variance':.04,'selection':'offset variance and weight selected on adaptation train histories; validation logits fit excluding held episode'}
print(report,flush=True);Path('eda/r2_selected_validation.json').write_text(json.dumps(report,indent=2));write_submission('eda/r2_selected_predictions.csv',pred)
p['global_state']['machine_variance']=.04;p['blend']['w_main']=weight;p['global_state_fit']['machine_variance']='0.04; adaptation stable train first60/tail40 component check eda/r2_offset_train.json';p['global_state_fit']['blend']='Weight selected by original-episode 3fold matched pseudo-short adaptation train check eda/r2_blend_train.json'
if weight==1.:p.pop('blend',None);p.pop('fallback',None)
save_params('eda/r2_selected_params.json',p)
