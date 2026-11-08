"""Train-only episode-held-out test of omitted own/peer temporal effects.
Fixed full-history inferred structural labels / initial weather checkpoint make
this a component diagnostic, not an end-to-end causal backtest. No validation data.
"""
import sys,json,pickle,math,time
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
sys.path.insert(0,'.')
from he.model import load_split
rows=pickle.load(open('eda/_state_rows.pkl','rb'));cfg=json.load(open('eda/ev64_global_hmm.json'));types=cfg['types'];cf=np.array(cfg['coef']);scale=cfg['scale'];hazard=cfg['hazard']
qs,extra=load_split('kit/public','train');qm={q['machine_id']:q for q in qs};eps=sorted({r['episode'] for r in rows});epid={ep:i for i,ep in enumerate(eps)}
mates=defaultdict(list)
for q in qs:mates[q['episode_id']].append(q)
rates={q['machine_id']:(sum(v>0 for v in q['seq'].values())+1)/(len(q['seq'])+2) for q in qs}
def lag_win(q,t):
    value=q['seq'].get(t)
    return rates[q['machine_id']] if value is None else float(value>0)
by=defaultdict(lambda:defaultdict(list));X=[];Y=[];groups=[];tail=[];features=defaultdict(list)
for k,r in enumerate(rows):
    i=types.index(r['type']);x=[0.]*11;x[2*i+r['high']]=1;x[6]=r['signal'];x[7+i]=r['regime']-.5;x[10]=r['peer']-3.5
    X.append(x);Y.append(r['y']);groups.append(epid[r['episode']]);tail.append(r['tick']>=.6*r['cutoff'])
    signal=r['signal'];state=r['high']-.5;regime=r['regime']-.5;peer=r['peer']-3.5
    own=qm[r['machine']];t=r['tick'];others=[q for q in mates[r['episode']] if q['machine_id']!=r['machine']]
    features['own_lags'].append([lag_win(own,t-lag)-rates[r['machine']] for lag in range(1,9)])
    values=[sum(lag_win(q,t-lag) for q in others)-3.5 for lag in (1,2,4,5,6,7,8)]
    values.append(sum(q['seq'].get(t-lag,1)-1 for q in others for lag in range(1,9))/8)
    features['peer_lags'].append(values)
    by[r['episode']][r['tick']].append(k)
X=np.array(X);Y=np.array(Y);groups=np.array(groups);tail=np.array(tail);eta=X@cf;H=np.full(len(rows),.5)
# Shared state for each observation uses other machines' strictly earlier ticks.
for ep,times in by.items():
    mids={rows[k]['machine'] for ids in times.values() for k in ids}
    for mid in mids:
        hot=.5;last=None
        for tick,ids in sorted(times.items()):
            if last is not None:hot=.5+(hot-.5)*(1-2*hazard)**(tick-last)
            ll=[0.,0.]
            for k in ids:
                if rows[k]['machine']==mid:H[k]=hot;continue
                for j,shift in enumerate((-scale,scale)):
                    z=eta[k]+shift;ll[j]+=Y[k]*z-max(z,0)-math.log1p(math.exp(-abs(z)))
            hot=float(expit(math.log(max(hot,1e-15)/max(1-hot,1e-15))+ll[1]-ll[0]));last=tick
print('prepared',len(rows),'rows',flush=True)
features={k:np.array(v) for k,v in features.items()};losses={};parameters={};start=time.perf_counter()
for name in ['baseline']+list(features):
    design=X if name=='baseline' else np.column_stack([X,features[name]])
    initial=np.r_[cf,np.zeros(design.shape[1]-11)];prediction=np.zeros(len(Y));parameters[name]=[]
    for fold in range(3):
        train=groups%3!=fold;held=(~train)&tail;A=design[train];y=Y[train];hot=H[train]
        def objective(v):
            z=A@v;p0=expit(z-scale);p1=expit(z+scale);p=(1-hot)*p0+hot*p1
            deriv=(1-hot)*p0*(1-p0)+hot*p1*(1-p1);p=np.clip(p,1e-12,1-1e-12)
            value=-np.sum(y*np.log(p)+(1-y)*np.log1p(-p))+.5*np.sum((v-initial)**2)
            gradient=A.T@((p-y)*deriv/(p*(1-p)))+(v-initial)
            return value,gradient
        fit=minimize(objective,initial,jac=True,method='L-BFGS-B',options={'maxiter':150,'ftol':1e-10,'gtol':1e-5})
        z=design[held]@fit.x;prediction[held]=(1-H[held])*expit(z-scale)+H[held]*expit(z+scale)
        parameters[name].append({'converged':bool(fit.success),'extra':fit.x[11:].tolist(),'coef':fit.x[:11].tolist()})
    losses[name]={ep:float(np.mean((prediction[(groups==j)&tail]-Y[(groups==j)&tail])**2)) for ep,j in epid.items() if np.any((groups==j)&tail)}
    print('finished',name,'seconds',time.perf_counter()-start,flush=True)
report={}
for name,loss in losses.items():
    d=np.array([loss[ep]-losses['baseline'][ep] for ep in loss])
    report[name]={'brier':float(np.mean(list(loss.values()))),'delta':float(d.mean()),'cluster_se':float(d.std(ddof=1)/math.sqrt(len(d))),'improved':int(sum(d<0)),'parameters':parameters[name]}
    print(name,{k:v for k,v in report[name].items() if k!='parameters'},flush=True)
Path('eda/ev91_temporal.json').write_text(json.dumps(report,indent=2))
