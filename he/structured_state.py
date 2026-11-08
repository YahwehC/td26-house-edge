"""Joint win/size structural hypotheses under history-fitted global logits."""
import math
from collections import defaultdict
from .model import DEFAULT_CFG, _pair_prior, SIZE_IDX
from .joint import episode_posterior_joint,lag_prior_marginal,lag_prior_for_pair
from .global_state import state_rows
from .weather import filter_episode


def predict(queries,params,config):
    cfg=dict(DEFAULT_CFG,**params['cfg']);pri=_pair_prior(cfg);marg=lag_prior_marginal(cfg);cf=config['coef'];types=config['types'];scale=config['scale']
    by=defaultdict(list)
    for q in queries:by[q['episode_id']].append(q)
    # Obtain a history-based shared-state estimate before the forecast tick.
    rows,_=state_rows(queries,params);traces=defaultdict(lambda:defaultdict(list));cuts={}
    for r in rows:
        i=types.index(r['type']);eta=cf[2*i+r['high']]+cf[6]*r['signal']+cf[7+i]*(r['regime']-.5)+cf[10]*(r['peer']-3.5)
        traces[r['episode']][r['tick']].append((eta,r['y']));cuts[r['episode']]=r['cutoff']
    hot={ep:filter_episode((cuts[ep],ts),scale,config['hazard'])[1] for ep,ts in traces.items()}
    result={}
    class Machine:
        pass
    for ep,qs in by.items():
        machines=[];regimes=[];short=sum(len(q['seq']) for q in qs)<.6*sum(q['cutoff'] for q in qs)
        if short and cfg.get('exact_sparse_glm'):
            from .sparse_glm import predict as sparse_predict
            for q in qs:
                peers={t:sum(other['seq'].get(t-3,0)>0 for other in qs if other is not q) for t in range(q['cutoff']+1)}
                result[q['query_id']]=sparse_predict(q,peers,params,config,hot.get(ep,.5))
            continue
        for q in qs:
            seq=q['seq'];cut=q['cutoff'];m=Machine();m.q=q;m.taus=[] if short else list(range(cfg['tau_lo'],cut+1));m.pairs=[];m.pair_logprior={};m.lml_stable={};m.lml_tau={};m.pred_stable={};m.pred_tau={};reg={}
            V=params['size_vectors'][q['machine_type']];eps=cfg['size_eps'];V=[[(1-eps)*V[r][k]+eps*V[1-r][k] for k in range(4)] for r in (0,1)]
            typeidx=types.index(q['machine_type']);ticks=sorted(seq);emp=[(sum(v==0 for v in seq.values())+1)/(len(seq)+3),(sum(v==1 for v in seq.values())+1)/(len(seq)+3),(sum(v>=2 for v in seq.values())+1)/(len(seq)+3)]
            peers={t:sum(other['seq'].get(t-3,0)>0 for other in qs if other is not q) for t in ticks+[cut]}
            for (a,b),lp in pri.items():
                lag=max(lag_prior_for_pair((a,b),cfg,marg),key=lag_prior_for_pair((a,b),cfg,marg).get)
                for orientation in (0,1):
                    for sign in (0,1):
                        key=(a,b,orientation,sign);m.pairs.append(key);m.pair_logprior[key]=lp-math.log(4)
                        def distribution(t,flip,weather):
                            lagset=sorted(set((a,b,lag)));choices=[]
                            for delay in lagset:
                                val=seq.get(t-delay)
                                choices.append([(0 if val==0 else (1 if val==1 else 2),1.)] if val is not None else list(enumerate(emp)))
                            import itertools
                            output=[0.]*5
                            for combo in itertools.product(*choices):
                                cats={delay:value[0] for delay,value in zip(lagset,combo)};weight=math.prod(value[1] for value in combo)
                                state=int((cats[a]>0)==(cats[b]>0))^orientation^flip;r=int(cats[lag]==2)^sign
                                eta=cf[2*typeidx+state]+cf[6]*q['signal']+cf[7+typeidx]*(r-.5)+cf[10]*(peers[t]-3.5)
                                pw=(1-weather)/(1+math.exp(-(eta-scale)))+weather/(1+math.exp(-(eta+scale)))
                                output[0]+=weight*(1-pw)
                                for k in range(4):output[k+1]+=weight*pw*V[r][k]
                            return output
                        before=[0.];after=[0.]
                        for t in ticks:
                            value=seq[t];idx=0 if value==0 else SIZE_IDX[value]+1
                            before.append(before[-1]+math.log(max(distribution(t,0,.5)[idx],1e-300)))
                            after.append(after[-1]+math.log(max(distribution(t,1,.5)[idx],1e-300)))
                        nxt0=distribution(cut,0,hot.get(ep,.5));nxt1=distribution(cut,1,hot.get(ep,.5))
                        m.lml_stable[key]=before[-1];m.pred_stable[key]=1-nxt0[0];m.pred_tau[key]=[1-nxt1[0]]*len(m.taus)
                        j=0;ll=[]
                        for tau in m.taus:
                            while j<len(ticks) and ticks[j]<tau:j+=1
                            ll.append(before[j]+after[-1]-after[j])
                        m.lml_tau[key]=ll;reg[key]=(nxt0,nxt1)
            machines.append(m);regimes.append((reg,V))
        win,_=episode_posterior_joint(machines,cfg)
        predictions=[[1-w] for w in win]
        for k in range(1,5):
            for m,(reg,V) in zip(machines,regimes):
                for key,(a,b) in reg.items():
                    m.pred_stable[key]=a[k];m.pred_tau[key]=[b[k]]*len(m.taus)
            prob,_=episode_posterior_joint(machines,cfg)
            for row,value in zip(predictions,prob):row.append(value)
        result.update({q['query_id']:pr for q,pr in zip(qs,predictions)})
    return result
