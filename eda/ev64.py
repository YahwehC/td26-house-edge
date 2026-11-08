"""Fit shared binary-state GLM by EM on train histories (no validation labels)."""
import sys,json,pickle,math
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,'.')
from he.weather import filter_episode
rows=pickle.load(open('eda/_state_rows.pkl','rb'));config=json.load(open('eda/ev63_global.json'));types=config['types']
X=[];Y=[];by=defaultdict(lambda:defaultdict(list))
for k,r in enumerate(rows):
    x=[0.]*11;i=types.index(r['type']);x[2*i+r['high']]=1;x[6]=r['signal'];x[7+i]=r['regime']-.5;x[10]=r['peer']-3.5
    X.append(x);Y.append(r['y']);by[r['episode']][r['tick']].append(k)
X=np.array(X);Y=np.array(Y);coef=np.array(config['coef']);eta=X@coef
best=None
for scale in (.2,.4,.6,.8):
    for hazard in (.005,.02,.08,.25):
        value=0
        for ep,times in by.items():
            trace=(max(times)+1,{t:[(float(eta[k]),int(Y[k])) for k in ids] for t,ids in times.items()})
            value+=filter_episode(trace,scale,hazard)[0]
        if best is None or value>best[0]:best=(value,scale,hazard)
print('initial',best,flush=True);_,scale,hazard=best

def smooth(eta,scale):
    expected=np.zeros(len(Y));loglike=0.
    loss0=Y*(eta-scale)-np.logaddexp(0,eta-scale);loss1=Y*(eta+scale)-np.logaddexp(0,eta+scale)
    for ep,times in by.items():
        ticks=sorted(times);em=np.array([[loss0[times[t]].sum(),loss1[times[t]].sum()] for t in ticks]);shift=em.max(1);em=np.exp(em-shift[:,None]);f=[];norms=[];prob=np.array([.5,.5])
        for j,t in enumerate(ticks):
            if j:
                stay=(1+(1-2*hazard)**(t-ticks[j-1]))/2
                prob=np.array([stay*prob[0]+(1-stay)*prob[1],(1-stay)*prob[0]+stay*prob[1]])
            prob=prob*em[j];norm=prob.sum();prob/=norm;f.append(prob.copy());norms.append(norm);loglike+=math.log(norm)+shift[j]
        back=np.ones(2)
        for j in range(len(ticks)-1,-1,-1):
            post=f[j]*back;post/=post.sum();expected[times[ticks[j]]]=post[1]
            if j:
                stay=(1+(1-2*hazard)**(ticks[j]-ticks[j-1]))/2;v=em[j]*back
                back=np.array([stay*v[0]+(1-stay)*v[1],(1-stay)*v[0]+stay*v[1]])/norms[j]
    return expected,loglike
for iteration in range(6):
    hot,ll=smooth(X@coef,scale)
    def objective(v):
        e=X@v[:11];s=v[11];lo=e-s;hi=e+s
        p0=1/(1+np.exp(-lo));p1=1/(1+np.exp(-hi))
        value=np.sum((1-hot)*(np.logaddexp(0,lo)-Y*lo)+hot*(np.logaddexp(0,hi)-Y*hi))+.001*np.dot(v[:11],v[:11])
        residual=(1-hot)*(p0-Y)+hot*(p1-Y)
        grad=np.r_[X.T@residual+.002*v[:11],np.sum(-(1-hot)*(p0-Y)+hot*(p1-Y))]
        return value,grad
    opt=minimize(objective,np.r_[coef,scale],jac=True,method='L-BFGS-B',bounds=[(None,None)]*11+[(.01,1.5)],options={'maxiter':150,'ftol':1e-10})
    coef=opt.x[:11];scale=float(opt.x[11]);print(iteration,ll,scale,flush=True)
config.update(coef=coef.tolist(),scale=scale,hazard=hazard)
Path('eda/ev64_global_hmm.json').write_text(json.dumps(config,indent=2));print(config,flush=True)
