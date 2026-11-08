"""Payout-regime correction to win probabilities, learned from train outcomes.

Independent implementation prompted by the public observation that win and
size regimes interact. No external source code or fitted parameters are used.
"""
import math
from collections import defaultdict
from .model import DEFAULT_CFG,WinMachine,_pair_prior,logsumexp
from .joint import SizeEvidence,lag_prior_marginal,lag_prior_for_pair,episode_posterior_joint


def apply(queries,pred,params,effects):
    cfg=dict(DEFAULT_CFG,**params['cfg']);vecs=params['size_vectors'];lp=_pair_prior(cfg)
    marginal=lag_prior_marginal(cfg);episodes=defaultdict(list);result={}
    for q in queries:episodes[q['episode_id']].append(q)
    for ep,qs in sorted(episodes.items()):
        qs=sorted(qs,key=lambda q:q['machine_id']);machines=[];sizes=[]
        short=sum(len(q['seq']) for q in qs)<.6*sum(q['cutoff'] for q in qs)
        for q in qs:
            se=SizeEvidence(q,vecs,cfg)
            factors={pair:se.factor_and_r1(lag_prior_for_pair(pair,cfg,marginal),cfg)[0] for pair in lp}
            weights={pair:lp[pair]+factors[pair] for pair in lp};norm=logsumexp(list(weights.values()))
            machines.append(WinMachine(q,cfg,{pair:v-norm for pair,v in weights.items()},stable_only=short))
            sizes.append(se)
        _,diagnostics=episode_posterior_joint(machines,cfg)
        changing=diagnostics['post'].get('stable',0)<.5
        for q,m,se,pairpost in zip(qs,machines,sizes,diagnostics['pair_post']):
            seq=q['seq'];cut=q['cutoff'];key=q['query_id'];old=pred[key];win=1-old[0]
            if not 1e-10<win<1-1e-10:result[key]=old;continue
            conditional=[v/win for v in old[1:]];v0,v1=se.Vn
            norm=sum((b-a)**2 for a,b in zip(v0,v1))
            regime=min(1.,max(0.,sum((p-a)*(b-a) for p,a,b in zip(conditional,v0,v1))/norm))
            cp=cfg['cell_prior'][q['machine_type']]
            level='hi' if abs(win-1/(1+math.exp(-(cp['hi'][0]+cp['hi'][1]*q['signal']))))<abs(win-1/(1+math.exp(-(cp['lo'][0]+cp['lo'][1]*q['signal'])))) else 'lo'
            kappa=cp[level][2]*cfg.get('kappa_scale',1)
            center=0.
            for (a,b),weight in pairpost.items():
                if cut-a not in seq or cut-b not in seq:
                    center+=weight*.5;continue
                zc=(seq[cut-a]>0)==(seq[cut-b]>0)
                pri=lag_prior_for_pair((a,b),cfg,marginal);lag=max(pri,key=pri.get)
                sl=[row[0] for row in se.lag[lag]];sn=logsumexp(sl);s1=math.exp(sl[1]-sn)
                total=count=0
                for t in seq:
                    if changing and t<cut-32:continue
                    if t-a in seq and t-b in seq and t-lag in seq and ((seq[t-a]>0)==(seq[t-b]>0))==zc:
                        bit=float(seq[t-lag]>=2);total+=(1-s1)*bit+s1*(1-bit);count+=1
                center+=weight*(total+.5*kappa)/(count+kappa)
            beta=effects[q['machine_type']]['beta'];eta=math.log(win/(1-win))
            w0=1/(1+math.exp(-(eta-beta*center)));w1=1/(1+math.exp(-(eta+beta*(1-center))))
            mass0=(1-regime)*w0;mass1=regime*w1;newwin=mass0+mass1
            residual=[p-((1-regime)*a+regime*b) for p,a,b in zip(conditional,v0,v1)]
            result[key]=[1-newwin]+[mass0*a+mass1*b+newwin*d for a,b,d in zip(v0,v1,residual)]
    return result
