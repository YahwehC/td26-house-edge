"""Anatomy only: oracle high/size/weather from released targets, no fitted knobs.
Separates structural state errors from shared-state filtering and residual levels.
"""
import sys,json,math
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,'.')
from he.model import load_split,official_score,read_csv
from he.api import load_params
q,e=load_split('kit/public','validation');p=load_params('runs/main_v10/params.json');cfg=p['global_state'];cf=cfg['coef'];types=cfg['types'];targets={r['query_id']:r for r in e['targets']};groups=defaultdict(list)
rows=[]
for r in q:
    t=targets[r['query_id']];win=1-float(t['p_0']);logit=math.log(win/(1-win));i=types.index(r['machine_type']);V=p['size_vectors'][r['machine_type']];regime=int(float(t['p_1'])/win>(V[0][0]+V[1][0])/2)
    state=int(logit>(cf[2*i]+cf[2*i+1])/2+cf[6]*r['signal']+cf[7+i]*(regime-.5));peers=sum(other['seq'].get(r['cutoff']-3,0)>0 for other in q if other['episode_id']==r['episode_id'] and other is not r)
    eta=cf[2*i+state]+cf[6]*r['signal']+cf[7+i]*(regime-.5)+cf[10]*(peers-3.5);groups[r['episode_id']].append(logit-eta);rows.append((r,eta,logit,regime,state))
signs={ep:1 if sum(v)>0 else -1 for ep,v in groups.items()};pred={};residuals=[];type_res=defaultdict(list)
for r,eta,logit,regime,state in rows:
    corrected=eta+signs[r['episode_id']]*cfg['scale'];win=1/(1+math.exp(-corrected));V=p['size_vectors'][r['machine_type']][regime];pred[r['query_id']]=[1-win]+[win*v for v in V];residuals.append(logit-corrected);type_res[r['machine_type'],state].append(logit-corrected)
report={'oracle_all_states':official_score(pred,e,q),'residual_logit_sd':(sum(x*x for x in residuals)/len(residuals))**.5,'by_type_state':{str(k):{'mean':sum(v)/len(v),'rms':(sum(x*x for x in v)/len(v))**.5} for k,v in type_res.items()}}
print(json.dumps(report,indent=2));Path('eda/ev76_anatomy.json').write_text(json.dumps(report,indent=2))
