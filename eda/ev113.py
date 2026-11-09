"""Train episode-held-out test of heterogeneous responses to common weather.
Frozen inferred structure and prefix offsets: component diagnostic only.
"""
import sys,json,pickle,math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0,'.')
from he.machine_offset import estimate_offsets
rows=pickle.load(open('eda/_state_rows.pkl','rb'));cfg=json.load(open('eda/ev64_global_hmm.json'));cfg['machine_variance']=.04
prefix=[r for r in rows if r['tick']<.6*r['cutoff']];off=estimate_offsets(prefix,cfg)
eps=sorted({r['episode'] for r in rows});eid={ep:i for i,ep in enumerate(eps)};N=len(eps);cuts=np.zeros(N,dtype=int)
for r in rows:cuts[eid[r['episode']]]=r['cutoff']
T=int(cuts.max());eta=[];y=[];indices=[];typeidx=[];high=[]
for r in rows:
    i=cfg['types'].index(r['type']);cf=cfg['coef'];eta.append(cf[2*i+r['high']]+cf[6]*r['signal']+cf[7+i]*(r['regime']-.5)+cf[10]*(r['peer']-3.5)+off.get((r['episode'],r['machine']),0.));y.append(r['y']);indices.append(eid[r['episode']]*T+r['tick']);typeidx.append(i);high.append(r['high'])
eta=np.array(eta);y=np.array(y);indices=np.array(indices);typeidx=np.array(typeidx);high=np.array(high);hazard=cfg['hazard']
def scores(v,kind):
    if kind=='common':amplitude=np.full(len(rows),v[0])
    elif kind=='type':amplitude=np.asarray(v)[typeidx]
    else:amplitude=np.asarray(v)[high]
    em=np.zeros((N*T,2))
    for j,sign in enumerate((-1,1)):
        z=eta+sign*amplitude;ll=y*z-np.logaddexp(0,z);em[:,j]=np.bincount(indices,weights=ll,minlength=N*T)
    em=em.reshape(N,T,2);shift=em.max(2);lik=np.exp(em-shift[:,:,None]);hot=np.full(N,.5);full=np.zeros(N);pre=np.zeros(N)
    for t in range(T):
        if t:hot=hazard+(1-2*hazard)*hot
        cold=(1-hot)*lik[:,t,0];warm=hot*lik[:,t,1];norm=cold+warm;hot=warm/norm;increment=np.log(norm)+shift[:,t];full+=increment*(t<cuts);pre+=increment*(t<.6*cuts)
    return full,pre
f,p=scores([cfg['scale']],'common');baseline=f-p;report={}
for kind,size in [('common',1),('type',3),('state',2)]:
    gains=np.zeros(N);parameters=[]
    for fold in range(3):
        train=np.arange(N)%3!=fold;held=~train;initial=np.full(size,cfg['scale'])
        def objective(v):return -scores(v,kind)[0][train].sum()+2*np.sum((np.array(v)-cfg['scale'])**2)
        fit=minimize(objective,initial,method='L-BFGS-B',bounds=[(.01,1.2)]*size,options={'ftol':1e-10});a,b=scores(fit.x,kind);gains[held]=(a-b-baseline)[held];parameters.append({'scale':fit.x.tolist(),'converged':bool(fit.success)})
    report[kind]={'logscore_gain':float(gains.mean()),'cluster_se':float(gains.std(ddof=1)/math.sqrt(N)),'improved':int(sum(gains>0)),'fold_parameters':parameters,'gains':gains.tolist()};print(kind,{k:v for k,v in report[kind].items() if k!='gains'},flush=True)
Path('eda/ev113_weather_loadings.json').write_text(json.dumps(report,indent=2))
