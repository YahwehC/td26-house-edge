"""Train-only three-fold test of asymmetric and history-dependent weather switches.
No validation files. Score final 40% of each held episode conditional on its prefix.
Latent machine states are frozen full-history estimates: component diagnostic only.
"""
import sys,json,pickle,math
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0,'.')
from he.model import load_split
rows=pickle.load(open('eda/_state_rows.pkl','rb'));cfg=json.load(open('eda/ev64_global_hmm.json'));cf=cfg['coef'];scale=cfg['scale'];types=cfg['types'];qs,_=load_split('kit/public','train')
eps=sorted({q['episode_id'] for q in qs});index={ep:i for i,ep in enumerate(eps)};cuts=np.array([next(q['cutoff'] for q in qs if q['episode_id']==ep) for ep in eps]);T=int(cuts.max());N=len(eps)
em=np.zeros((N,T,2));feature=np.zeros((N,T));net=np.zeros((N,T));seen=np.zeros((N,T))
for q in qs:
    i=index[q['episode_id']]
    for t,v in q['seq'].items():
        if t<T:feature[i,t]+=v>0;net[i,t]+=v-1;seen[i,t]+=1
feature=np.where(seen>0,feature/np.maximum(seen,1)-.5,0)
feature=np.concatenate([np.zeros((N,1)),feature[:,:-1]],axis=1)
for r in rows:
    i=types.index(r['type']);eta=cf[2*i+r['high']]+cf[6]*r['signal']+cf[7+i]*(r['regime']-.5)+cf[10]*(r['peer']-3.5);z=eta+np.array([-scale,scale]);em[index[r['episode']],r['tick']]+=r['y']*z-np.logaddexp(0,z)
shift=em.max(2);lik=np.exp(em-shift[:,:,None]);sig=lambda x:1/(1+np.exp(-x))
def scores(v,kind):
    up=sig(v[0]);down=up if kind=='symmetric' else sig(v[1]);hot=np.full(N,up/(up+down));full=np.zeros(N);prefix=np.zeros(N)
    for t in range(T):
        if kind=='triggered':u=sig(v[0]+v[2]*feature[:,t]);d=sig(v[1]-v[2]*feature[:,t])
        else:u,d=up,down
        if t:hot=hot*(1-d)+(1-hot)*u
        cold=(1-hot)*lik[:,t,0];warm=hot*lik[:,t,1];normal=cold+warm;increment=np.log(normal)+shift[:,t]
        full+=increment*(t<cuts);prefix+=increment*(t<.6*cuts);hot=warm/normal
    return full,prefix
base=np.log(cfg['hazard']/(1-cfg['hazard']));bfull,bpre=scores([base],'symmetric');baseline=bfull-bpre
report={}
for kind,size in [('symmetric',1),('asymmetric',2),('triggered',3)]:
    gains=np.zeros(N);parameters=[]
    for fold in range(3):
        train=np.arange(N)%3!=fold;held=~train;initial=[base]*min(size,2)+([0.] if size==3 else [])
        def objective(v):return -scores(v,kind)[0][train].sum()+.1*np.sum((np.array(v)-initial)**2)
        fit=minimize(objective,initial,method='L-BFGS-B',bounds=[(-7,-.5)]*min(size,2)+([(-3,3)] if size==3 else []))
        full,pre=scores(fit.x,kind);gains[held]=(full-pre-baseline)[held];parameters.append(fit.x.tolist())
    report[kind]={'mean_logscore_gain':float(gains.mean()),'episode_se':float(gains.std(ddof=1)/math.sqrt(N)),'improved_episodes':int(sum(gains>0)),'fold_parameters':parameters,'gains':gains.tolist()}
    print(kind,{k:v for k,v in report[kind].items() if k!='gains'},flush=True)
Path('eda/ev82_weather_switches.json').write_text(json.dumps(report,indent=2))
