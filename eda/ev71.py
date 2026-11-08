"""Train-only evidence for abrupt versus gradual reversal, with equal tau support."""
import sys,math,json
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0,'.')
from he.api import load_params
from he.model import load_split
from he.global_state import state_rows
p=load_params('runs/main_v10/params.json');q,e=load_split('kit/public','train');cm={r['query_id']:r['condition'] for r in e['conditions']};conditions={r['machine_id']:cm[r['query_id']] for r in q}
# Recovered high-state traces are used only to define the likely pair/sign;
# then compare likelihood under ramp widths, marginalizing onset time.
rows,_=state_rows(q,p);by=defaultdict(list)
for r in rows:
    if conditions[r['machine']] in ('drift','shared_change'):by[r['machine']].append(r)
cfg=json.load(open('eda/ev64_global_hmm.json'));cf=cfg['coef'];scale=cfg['scale'];types=cfg['types'];widths=[0,4,8,16,32];scores=defaultdict(lambda:np.zeros(len(widths)));machines=defaultdict(int)
for mid,rr in by.items():
    if len(rr)<60:continue
    r=rr[0];i=types.index(r['type']);t=np.array([r['tick'] for r in rr]);y=np.array([r['y'] for r in rr]);high=np.array([r['high'] for r in rr])
    # high in recovered trace already contains a flip. Reconstruct raw XNOR
    # directly using best pair from observed history (joint sign not relevant).
    seq=next(q['seq'] for q in q if q['machine_id']==mid)
    from he.coupling import _best_pair,pairs_from_cfg
    _,(a,b),_= _best_pair(seq,min(t)+max(50,len(t)//3),pairs_from_cfg(p['cfg']))
    raw=np.array([int((seq.get(int(tick)-a,0)>0)==(seq.get(int(tick)-b,0)>0)) for tick in t])
    offset=cf[6]*r['signal']+cf[7+i]*(np.array([r['regime'] for r in rr])-.5)+cf[10]*(np.array([r['peer'] for r in rr])-3.5)
    eta0=np.array(cf[2*i:2*i+2])[raw]+offset;eta1=np.array(cf[2*i:2*i+2])[1-raw]+offset
    logistic=lambda x:1/(1+np.exp(-x))
    p0=.5*(logistic(eta0-scale)+logistic(eta0+scale));p1=.5*(logistic(eta1-scale)+logistic(eta1+scale))
    taus=np.arange(8,r['cutoff']+1);prob0=np.where(y,p0,1-p0);prob1=np.where(y,p1,1-p1)
    values=[]
    for width in widths:
        mix=(t[:,None]>=taus).astype(float) if width==0 else np.clip(.5+(t[:,None]-taus)/(2*width),0,1)
        ll0=np.log(prob0[:,None]*(1-mix)+prob1[:,None]*mix).sum(0)
        ll1=np.log(prob1[:,None]*(1-mix)+prob0[:,None]*mix).sum(0)
        likelihood=np.r_[ll0,ll1];maxll=likelihood.max();values.append(maxll+math.log(np.exp(likelihood-maxll).mean()))
    scores[conditions[mid]]+=np.array(values)-values[0];machines[conditions[mid]]+=1
report={'widths':widths,'delta_log_evidence':{c:values.tolist() for c,values in scores.items()},'machines':dict(machines)}
print(report);Path('eda/ev71_ramps.json').write_text(json.dumps(report,indent=2))
