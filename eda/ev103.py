"""Train component diagnostic: does regime noise jointly affect win and payout size?
One scalar noise rate fit on other episode folds; held tail evaluated once.
Frozen full-history structural labels remain a limitation; no validation data.
"""
import sys,json,pickle,math
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize_scalar
sys.path.insert(0,'.')
from he.api import load_params
from he.model import load_split,SIZE_IDX
from he.machine_offset import estimate_offsets
rows=pickle.load(open('eda/_state_rows.pkl','rb'));cfg=json.load(open('eda/ev64_global_hmm.json'));cfg['machine_variance']=.04
p=load_params('runs/main_v13/params.json');qs,ex=load_split('kit/public','train');qm={q['machine_id']:q for q in qs};cf=cfg['coef'];types=cfg['types'];scale=cfg['scale'];hazard=cfg['hazard'];off=estimate_offsets([r for r in rows if r['tick']<.6*r['cutoff']],cfg)
eps=sorted({r['episode'] for r in rows});eid={ep:i for i,ep in enumerate(eps)};by=defaultdict(lambda:defaultdict(list));eta=[];regime=[];labels=[];groups=[];tail=[];vectors=[]
for k,r in enumerate(rows):
    i=types.index(r['type']);base=cf[2*i+r['high']]+cf[6]*r['signal']+cf[10]*(r['peer']-3.5)+off.get((r['episode'],r['machine']),0.)
    eta.append(base);regime.append(r['regime']);value=qm[r['machine']]['seq'][r['tick']];labels.append(0 if value==0 else SIZE_IDX[value]+1);groups.append(eid[r['episode']]);tail.append(r['tick']>=.6*r['cutoff']);vectors.append(p['size_vectors'][r['type']]);by[r['episode']][r['tick']].append((k,base+cf[7+i]*(r['regime']-.5)))
H=np.full(len(rows),.5)
for ep,times in by.items():
    mids={rows[k]['machine'] for observations in times.values() for k,_ in observations}
    for mid in mids:
        hot=.5;last=None
        for tick,observations in sorted(times.items()):
            if last is not None:hot=.5+(hot-.5)*(1-2*hazard)**(tick-last)
            ll=[0.,0.]
            for k,z in observations:
                if rows[k]['machine']==mid:H[k]=hot;continue
                for j,shift in enumerate((-scale,scale)):
                    value=z+shift;ll[j]+=rows[k]['y']*value-max(value,0)-math.log1p(math.exp(-abs(value)))
            hot=float(expit(math.log(max(hot,1e-15)/max(1-hot,1e-15))+ll[1]-ll[0]));last=tick
eta=np.array(eta);r=np.array(regime);V=np.array(vectors);labels=np.array(labels);groups=np.array(groups);tail=np.array(tail);beta=np.array([cf[7+types.index(row['type'])] for row in rows])
win=[];dist=[]
for state in (0,1):
    z=eta+beta*(state-.5);w=(1-H)*expit(z-scale)+H*expit(z+scale);win.append(w);dist.append(np.column_stack([1-w,w[:,None]*V[:,state]]))
base=(1-r[:,None])*dist[0]+r[:,None]*dist[1]
ind=np.column_stack([np.zeros(len(r)),((1-r)*win[0]-r*win[1])[:,None]*(V[:,1]-V[:,0])])
joint=(1-2*r)[:,None]*(dist[1]-dist[0]);truth=np.eye(5)[labels];baseline=base+.02*ind
baseline_loss=np.sum((baseline-truth)**2,axis=1);report={}
for name,direction in [('size_only',ind),('joint_regime',joint)]:
    loss=np.zeros(len(rows));rates=[]
    for fold in range(3):
        train=groups%3!=fold;held=(~train)&tail;prob=base[np.arange(len(rows)),labels];slope=direction[np.arange(len(rows)),labels]
        objective=lambda rate:-np.log(np.maximum(1e-15,prob[train]+rate*slope[train])).sum()
        fit=minimize_scalar(objective,bounds=(0.,.2),method='bounded',options={'xatol':1e-8});rate=float(fit.x) if fit.fun<objective(0.) else 0.;rates.append(rate)
        prediction=base[held]+rate*direction[held];loss[held]=np.sum((prediction-truth[held])**2,axis=1)
    full_objective=lambda rate:-np.log(np.maximum(1e-15,prob+rate*slope)).sum()
    full_fit=minimize_scalar(full_objective,bounds=(0.,.2),method='bounded',options={'xatol':1e-8})
    full_rate=float(full_fit.x) if full_fit.fun<full_objective(0.) else 0.
    differences=np.array([np.mean((loss-baseline_loss)[(groups==i)&tail]) for i in range(len(eps))]);report[name]={'fold_rates':rates,'full_rate':full_rate,'delta':float(differences.mean()),'cluster_se':float(differences.std(ddof=1)/math.sqrt(len(eps))),'improved':int(sum(differences<0))}
    print(name,report[name],flush=True)
Path('eda/ev103_regime_noise.json').write_text(json.dumps(report,indent=2))
