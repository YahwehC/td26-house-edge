"""Episode-level binary latent logit state; all fitting uses history outcomes."""
import math
from collections import defaultdict
from .model import DEFAULT_CFG,WinMachine,_pair_prior,logsumexp,betaln
from .joint import SizeEvidence,lag_prior_marginal,lag_prior_for_pair,episode_posterior_joint


def traces(queries,params,effects):
    cfg=dict(DEFAULT_CFG,**params['cfg']);vecs=params['size_vectors'];lp=_pair_prior(cfg)
    marginal=lag_prior_marginal(cfg);episodes=defaultdict(list)
    for q in queries:episodes[q['episode_id']].append(q)
    out={}
    for ep,qs in sorted(episodes.items()):
        qs=sorted(qs,key=lambda q:q['machine_id']);machines=[];sizes=[]
        short=sum(len(q['seq']) for q in qs)<.6*sum(q['cutoff'] for q in qs)
        for q in qs:
            se=SizeEvidence(q,vecs,cfg)
            f={pair:se.factor_and_r1(lag_prior_for_pair(pair,cfg,marginal),cfg)[0] for pair in lp}
            wt={pair:lp[pair]+f[pair] for pair in lp};norm=logsumexp(list(wt.values()))
            machines.append(WinMachine(q,cfg,{pair:v-norm for pair,v in wt.items()},stable_only=short));sizes.append(se)
        _,diag=episode_posterior_joint(machines,cfg)
        stable=diag['post'].get('stable',0)>.5
        times=defaultdict(list)
        for q,m,se,post in zip(qs,machines,sizes,diag['pair_post']):
            seq=q['seq'];cut=q['cutoff'];pair=max(post,key=post.get);a,b=pair
            tau=cut+1
            if not stable and m.taus:
                idx=max(range(len(m.taus)),key=lambda i:m.lml_tau[pair][i])
                if m.lml_tau[pair][idx]-math.log(len(m.taus))>m.lml_stable[pair]:tau=m.taus[idx]
            lag=max(lag_prior_for_pair(pair,cfg,marginal),key=lag_prior_for_pair(pair,cfg,marginal).get)
            sl=[row[0] for row in se.lag[lag]];sg=1/(1+math.exp(max(-700,min(700,sl[0]-sl[1]))))
            counts=[[0,0],[0,0]];reg=[0.,0.];records=[]
            for t,val in seq.items():
                if t<8 or t-a not in seq or t-b not in seq or t-lag not in seq:continue
                z=int((seq[t-a]>0)==(seq[t-b]>0))^int(t>=tau)
                bit=int(seq[t-lag]>=2);r=(1-sg)*bit+sg*(1-bit)
                counts[z][int(val>0)]+=1;reg[z]+=r;records.append((t,z,r,int(val>0)))
            cp=cfg['cell_prior'][q['machine_type']];pri=[]
            for nm in ('hi','lo'):
                mu,slope,k=cp[nm];p=1/(1+math.exp(-(mu+slope*q['signal'])));pri.append((k*p,k*(1-p)))
            ll=[]
            for orientation in (0,1):
                ll.append(sum(betaln(pri[z^orientation][0]+w,pri[z^orientation][1]+l)-betaln(*pri[z^orientation]) for z,(l,w) in enumerate(counts)))
            ori=int(ll[1]>ll[0]);beta=effects[q['machine_type']]['beta'];corr=params.get('logit_corr',{})
            for t,z,r,y in records:
                alpha,beta0=pri[z^ori];l,w=counts[z];n=l+w
                # Leave-one-out cell rate keeps the outcome out of its own offset.
                mean=(alpha+w-y)/(alpha+beta0+n-1)
                rbar=(reg[z]-r+.5*(alpha+beta0))/(n-1+alpha+beta0)
                eta=math.log(mean/(1-mean))+beta*(r-rbar)
                for name,coef,center in zip(corr.get('names',[]),corr.get('beta',[]),corr.get('mu',[])):
                    kind,_,ks=name.rpartition('_');k=int(ks);others=[other for other in qs if other is not q]
                    if kind=='oth_win':value=sum(other['seq'].get(t-k,0)>0 for other in others)
                    elif kind=='oth_net':value=sum(other['seq'].get(t-j,1)-1 for other in others for j in range(1,k+1))/k
                    else:continue
                    eta+=coef*(value-center)
                times[t].append((eta,y))
        out[ep]=(max(q['cutoff'] for q in qs),dict(times))
    return out


def filter_episode(trace,scale,hazard):
    cutoff,times=trace;hot=.5;last=None;likelihood=0.
    for tick,rows in sorted(times.items()):
        if last is not None:hot=.5+(hot-.5)*(1-2*hazard)**(tick-last)
        emissions=[]
        for sign in (-1,1):
            ll=0.
            for eta,y in rows:
                if isinstance(eta, (tuple, list)):
                    probability = eta[0 if sign < 0 else 1]
                    ll += math.log(probability if y else 1-probability)
                else:
                    z=eta+sign*scale
                    ll+=y*z-(max(z,0)+math.log1p(math.exp(-abs(z))))
            emissions.append(ll)
        terms=[math.log(max(1-hot,1e-300))+emissions[0],math.log(max(hot,1e-300))+emissions[1]]
        norm=logsumexp(terms);hot=math.exp(terms[1]-norm);likelihood+=norm;last=tick
    if last is not None:hot=.5+(hot-.5)*(1-2*hazard)**(cutoff-last)
    return likelihood,hot


def apply(queries,pred,params,effects,config):
    data=traces(queries,params,effects);result={};scale=config['scale'];hazard=config['hazard']
    hot={ep:filter_episode(tr,scale,hazard)[1] for ep,tr in data.items()}
    for q in queries:
        key=q['query_id'];old=pred[key];w=1-old[0];eta=math.log(max(1e-12,w)/max(1e-12,1-w));h=hot[q['episode_id']]
        new=(1-h)/(1+math.exp(-(eta-scale)))+h/(1+math.exp(-(eta+scale)))
        result[key]=[1-new]+[new*x/w for x in old[1:]] if w>1e-12 else old
    return result
