"""Train cross-fit: binary persistent weather plus shared IID tick-level shocks.
No validation files; full-history inferred structure/offsets remain a diagnostic caveat.
"""
import sys,json,pickle,math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
sys.path.insert(0,'.')
from he.machine_offset import estimate_offsets
from he.short_weather_joint import NODES,WEIGHTS
rows=pickle.load(open('eda/_state_rows.pkl','rb'));cfg=json.load(open('eda/ev64_global_hmm.json'));cfg['machine_variance']=.04;off=estimate_offsets(rows,cfg)
eps=sorted({r['episode'] for r in rows});eid={ep:i for i,ep in enumerate(eps)};cuts=np.zeros(len(eps),dtype=int)
for r in rows:cuts[eid[r['episode']]]=r['cutoff']
T=int(cuts.max());N=len(eps);eta=[];y=[];indices=[]
for r in rows:
    i=cfg['types'].index(r['type']);cf=cfg['coef'];eta.append(cf[2*i+r['high']]+cf[6]*r['signal']+cf[7+i]*(r['regime']-.5)+cf[10]*(r['peer']-3.5)+off.get((r['episode'],r['machine']),0.));y.append(r['y']);indices.append(eid[r['episode']]*T+r['tick'])
eta=np.array(eta);y=np.array(y);indices=np.array(indices);hazard=cfg['hazard'];scale=cfg['scale']
def scores(sd):
    quadrature=list(zip(NODES,WEIGHTS)) if sd else [(0.,1.)];em=np.zeros((N*T,2,len(quadrature)))
    for state,shift in enumerate((-scale,scale)):
        for j,(node,weight) in enumerate(quadrature):
            z=eta+shift+sd*node;ll=y*z-np.logaddexp(0,z);em[:,state,j]=np.bincount(indices,weights=ll,minlength=N*T)+math.log(weight)
    em=logsumexp(em,axis=2).reshape(N,T,2);shift=em.max(2);lik=np.exp(em-shift[:,:,None]);hot=np.full(N,.5);full=np.zeros(N);prefix=np.zeros(N)
    for t in range(T):
        if t:hot=hazard+(1-2*hazard)*hot
        cold=(1-hot)*lik[:,t,0];warm=hot*lik[:,t,1];norm=cold+warm;hot=warm/norm;increment=np.log(norm)+shift[:,t]
        full+=increment*(t<cuts);prefix+=increment*(t<.6*cuts)
    return full,prefix
f,p=scores(0.);baseline=f-p;gains=np.zeros(N);parameters=[]
for fold in range(3):
    train=np.arange(N)%3!=fold;held=~train
    fit=minimize_scalar(lambda sd:-scores(sd)[0][train].sum(),bounds=(0.,1.),method='bounded',options={'xatol':1e-4})
    sd=float(fit.x) if fit.fun < -f[train].sum() else 0.;parameters.append(sd);a,b=scores(sd);gains[held]=(a-b-baseline)[held];print('fold',fold,'shared_shock_sd',sd,flush=True)
report={'logscore_gain':float(gains.mean()),'episode_se':float(gains.std(ddof=1)/math.sqrt(N)),'improved':int(sum(gains>0)),'fold_sd':parameters,'gains':gains.tolist()}
print({k:v for k,v in report.items() if k!='gains'},flush=True);Path('eda/ev97_shared_shocks.json').write_text(json.dumps(report,indent=2))
