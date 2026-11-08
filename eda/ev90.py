"""Train episode cross-fit: continuous AR(1) weather vs binary Markov weather.
Frozen full-history structural labels and intercepts: component diagnostic only.
No validation files/targets. Hold-out score uses the last 40% conditional on prefix.
"""
import sys,json,pickle,math,time
from pathlib import Path
import numpy as np
from scipy.special import ndtr,expit
from scipy.optimize import minimize
sys.path.insert(0,'.')
from he.machine_offset import estimate_offsets
rows=pickle.load(open('eda/_state_rows.pkl','rb'));cfg=json.load(open('eda/ev64_global_hmm.json'));cfg['machine_variance']=.04
off=estimate_offsets(rows,cfg);eps=sorted({r['episode'] for r in rows});eid={ep:i for i,ep in enumerate(eps)};cuts=np.zeros(len(eps),dtype=int)
eta=[];y=[];indices=[]
for r in rows:cuts[eid[r['episode']]]=r['cutoff']
T=int(cuts.max());N=len(eps)
for r in rows:
    i=cfg['types'].index(r['type']);cf=cfg['coef'];eta.append(cf[2*i+r['high']]+cf[6]*r['signal']+cf[7+i]*(r['regime']-.5)+cf[10]*(r['peer']-3.5)+off.get((r['episode'],r['machine']),0.))
    y.append(r['y']);indices.append(eid[r['episode']]*T+r['tick'])
eta=np.array(eta);y=np.array(y);indices=np.array(indices);grid=np.linspace(-3.5,3.5,21);boundaries=(grid[:-1]+grid[1:])/2;boundaries=np.r_[-np.inf,boundaries,np.inf]
initial_prior=np.diff(ndtr(boundaries))

def scores(v,kind):
    amplitude=math.exp(v[0])
    if kind=='binary':
        h=float(expit(v[1]));states=np.array([-1.,1.]);transition=np.array([[1-h,h],[h,1-h]]);prior=np.array([.5,.5])
    else:
        rho=v[1];states=grid;transition=np.diff(ndtr((boundaries[None,:]-rho*grid[:,None])/math.sqrt(1-rho*rho)),axis=1);prior=initial_prior
    K=len(states);em=np.zeros((N*T,K))
    for j,state in enumerate(states):
        z=eta+amplitude*state;ll=y*z-np.logaddexp(0,z)
        em[:,j]=np.bincount(indices,weights=ll,minlength=N*T)
    em=em.reshape(N,T,K);shift=em.max(axis=2);lik=np.exp(em-shift[:,:,None]);prob=np.tile(prior,(N,1));full=np.zeros(N);prefix=np.zeros(N)
    for t in range(T):
        if t:prob=prob@transition
        prob*=lik[:,t];norm=prob.sum(axis=1);prob/=norm[:,None];increment=np.log(norm)+shift[:,t]
        full+=increment*(t<cuts);prefix+=increment*(t<.6*cuts)
    return full,prefix
start=time.perf_counter();initial=[math.log(cfg['scale']),math.log(cfg['hazard']/(1-cfg['hazard']))];bf,bp=scores(initial,'binary');baseline=bf-bp
report={}
for kind in ('binary','ar1'):
    gains=np.zeros(N);parameters=[]
    for fold in range(3):
        train=np.arange(N)%3!=fold;held=~train
        x0=initial if kind=='binary' else [initial[0],.96]
        bounds=[(math.log(.05),math.log(1.2)),(-7,-.5)] if kind=='binary' else [(math.log(.05),math.log(1.2)),(.05,.995)]
        def objective(v):return -scores(v,kind)[0][train].sum()+.1*np.sum((np.array(v)-x0)**2)
        fit=minimize(objective,x0,method='L-BFGS-B',bounds=bounds,options={'maxiter':60,'ftol':1e-8})
        f,p=scores(fit.x,kind);gains[held]=(f-p-baseline)[held]
        params={'scale':math.exp(fit.x[0]),'hazard' if kind=='binary' else 'rho':float(expit(fit.x[1])) if kind=='binary' else float(fit.x[1]),'converged':bool(fit.success)}
        parameters.append(params);print(kind,fold,params,'elapsed',time.perf_counter()-start,flush=True)
    report[kind]={'logscore_gain':float(gains.mean()),'episode_se':float(gains.std(ddof=1)/math.sqrt(N)),'improved':int(sum(gains>0)),'parameters':parameters,'gains':gains.tolist()}
    print(kind,{k:v for k,v in report[kind].items() if k not in ('parameters','gains')},flush=True)
Path('eda/ev90_continuous_weather.json').write_text(json.dumps(report,indent=2))
