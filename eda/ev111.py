"""Train-only test of changed probability levels and offset memory after a flip.
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
from he.machine_offset import estimate_offsets
rows=pickle.load(open('eda/_state_rows.pkl','rb'));cfg=json.load(open('eda/ev64_global_hmm.json'));types=cfg['types'];cf=np.array(cfg['coef']);scale=cfg['scale'];hazard=cfg['hazard']
qs,extra=load_split('kit/public','train');qm={q['machine_id']:q for q in qs};eps=sorted({r['episode'] for r in rows});epid={ep:i for i,ep in enumerate(eps)}
conditions={r['query_id']:r['condition'] for r in extra['conditions']}
condition_by_ep={q['episode_id']:conditions[q['query_id']] for q in qs}
records=defaultdict(list)
for r in rows:records[r['machine']].append(r)
phases={}
for mid,rs in records.items():
    seq=qm[mid]['seq'];candidates=[]
    for a,b in ((1,3),(2,5),(4,7)):
        known=[(r['tick'],r['high']^int((seq[r['tick']-a]>0)==(seq[r['tick']-b]>0))) for r in rs if r['tick']-a in seq and r['tick']-b in seq]
        changes=[known[j][0] for j in range(1,len(known)) if known[j][1]!=known[j-1][1]]
        if known and len(changes)<=1:candidates.append((len(known),changes[0] if changes else None))
    if candidates:
        support=max(n for n,tau in candidates);taus={tau for n,tau in candidates if n==support}
        if len(taus)==1:phases[mid]=taus.pop()
offsets=estimate_offsets([r for r in rows if r['tick']<.6*r['cutoff']],dict(cfg,machine_variance=.04))
# Remove uncertain transition bands from level comparison, preserving all stable rows.
rows=[r for r in rows if r['machine'] in phases and (phases[r['machine']] is None or abs(r['tick']-phases[r['machine']])>16)]
by=defaultdict(lambda:defaultdict(list));X=[];Y=[];groups=[];tail=[];features=defaultdict(list)
fixed=[]
for k,r in enumerate(rows):
    i=types.index(r['type']);x=[0.]*11;x[2*i+r['high']]=1;x[6]=r['signal'];x[7+i]=r['regime']-.5;x[10]=r['peer']-3.5
    X.append(x);Y.append(r['y']);groups.append(epid[r['episode']]);tail.append(r['tick']>=.6*r['cutoff'])
    signal=r['signal'];state=r['high']-.5;regime=r['regime']-.5;peer=r['peer']-3.5
    offset=offsets.get((r['episode'],r['machine']),0.);fixed.append(offset)
    tau=phases[r['machine']];post=float(tau is not None and r['tick']>=tau)
    features['post_flip'].append([post,post*state,post*offset])
    by[r['episode']][r['tick']].append(k)
X=np.array(X);Y=np.array(Y);groups=np.array(groups);tail=np.array(tail);fixed=np.array(fixed);eta=X@cf+fixed;H=np.full(len(rows),.5)
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
            z=A@v+fixed[train];p0=expit(z-scale);p1=expit(z+scale);p=(1-hot)*p0+hot*p1
            deriv=(1-hot)*p0*(1-p0)+hot*p1*(1-p1);p=np.clip(p,1e-12,1-1e-12)
            value=-np.sum(y*np.log(p)+(1-y)*np.log1p(-p))+.5*np.sum((v-initial)**2)
            gradient=A.T@((p-y)*deriv/(p*(1-p)))+(v-initial)
            return value,gradient
        fit=minimize(objective,initial,jac=True,method='L-BFGS-B',options={'maxiter':150,'ftol':1e-10,'gtol':1e-5})
        z=design[held]@fit.x+fixed[held];prediction[held]=(1-H[held])*expit(z-scale)+H[held]*expit(z+scale)
        parameters[name].append({'converged':bool(fit.success),'extra':fit.x[11:].tolist(),'coef':fit.x[:11].tolist()})
    losses[name]={ep:float(np.mean((prediction[(groups==j)&tail]-Y[(groups==j)&tail])**2)) for ep,j in epid.items() if np.any((groups==j)&tail)}
    print('finished',name,'seconds',time.perf_counter()-start,flush=True)
report={}
for name,loss in losses.items():
    d=np.array([loss[ep]-losses['baseline'][ep] for ep in loss])
    report[name]={'brier':float(np.mean(list(loss.values()))),'delta':float(d.mean()),'cluster_se':float(d.std(ddof=1)/math.sqrt(len(d))),'improved':int(sum(d<0)),'parameters':parameters[name]}
    report[name]['by_condition']={}
    for condition in sorted(set(condition_by_ep.values())):
        sub=np.array([loss[ep]-losses['baseline'][ep] for ep in loss if condition_by_ep[ep]==condition])
        report[name]['by_condition'][condition]={'delta':float(sub.mean()),'episode_se':float(sub.std(ddof=1)/math.sqrt(len(sub)))}
    print(name,{k:v for k,v in report[name].items() if k!='parameters'},flush=True)
Path('eda/ev111_post_flip.json').write_text(json.dumps(report,indent=2))
