"""Global type/state model with a shared dynamic episode effect (stdlib)."""
import math
from collections import defaultdict
from .model import DEFAULT_CFG, WinMachine, _pair_prior, betaln, logsumexp
from .joint import SizeEvidence, lag_prior_marginal, lag_prior_for_pair, episode_posterior_joint
from .weather import filter_episode

def state_rows(qs,p):
    cfg=dict(DEFAULT_CFG,**p['cfg']);lp=_pair_prior(cfg);marg=lag_prior_marginal(cfg);by=defaultdict(list)
    for q in qs:by[q['episode_id']].append(q)
    rows=[];next_rows={}
    for ep,ms in by.items():
        models=[];sizes=[]
        short=sum(len(q['seq']) for q in ms)<.6*sum(q['cutoff'] for q in ms)
        for q in ms:
            se=SizeEvidence(q,p['size_vectors'],cfg);f={pa:se.factor_and_r1(lag_prior_for_pair(pa,cfg,marg),cfg)[0] for pa in lp}
            lps={pa:lp[pa]+f[pa] for pa in lp};norm=logsumexp(list(lps.values()))
            models.append(WinMachine(q,cfg,{pa:v-norm for pa,v in lps.items()},stable_only=short));sizes.append(se)
        _,diag=episode_posterior_joint(models,cfg);stable=diag['post'].get('stable',0)>.5
        for q,m,se,post in zip(ms,models,sizes,diag['pair_post']):
            seq=q['seq'];cut=q['cutoff'];pa=max(post,key=post.get);a,b=pa;tau=cut+1
            if not stable and m.taus:
                k=max(range(len(m.taus)),key=lambda i:m.lml_tau[pa][i])
                if m.lml_tau[pa][k]-math.log(len(m.taus))>m.lml_stable[pa]:tau=m.taus[k]
            lag=max(lag_prior_for_pair(pa,cfg,marg),key=lag_prior_for_pair(pa,cfg,marg).get)
            sl=[r[0] for r in se.lag[lag]];s1=math.exp(sl[1]-logsumexp(sl))
            counts=[[0,0],[0,0]]
            for t,val in seq.items():
                if t-a in seq and t-b in seq:counts[int((seq[t-a]>0)==(seq[t-b]>0))^int(t>=tau)][int(val>0)]+=1
            cp=cfg['cell_prior'][q['machine_type']];pri=[]
            for nm in ('hi','lo'):
                mu,slope,k=cp[nm];prob=1/(1+math.exp(-(mu+slope*q['signal'])));pri.append((k*prob,k*(1-prob)))
            ll=[sum(betaln(pri[z^ori][0]+w,pri[z^ori][1]+l)-betaln(*pri[z^ori]) for z,(l,w) in enumerate(counts)) for ori in (0,1)]
            ori=int(ll[1]>ll[0])
            orientation_zero = math.exp(ll[0]-logsumexp(ll))
            for t in list(sorted(seq))+[cut]:
                if t<8 or t-a not in seq or t-b not in seq or t-lag not in seq:continue
                z=int((seq[t-a]>0)==(seq[t-b]>0))^int(t>=tau);high=int((z^ori)==0)
                bit=int(seq[t-lag]>=2);r=(1-s1)*bit+s1*(1-bit)
                other=sum(other['seq'].get(t-int(cfg.get('peer_lag',3)),0)>0 for other in ms if other is not q)
                row={'episode':ep,'machine':q['machine_id'],'type':q['machine_type'],'signal':q['signal'],'high':high,'regime':r,'peer':other,'tick':t,'y':int(seq.get(t,0)>0),'cutoff':cut}
                row['high_probability'] = orientation_zero if z==0 else 1-orientation_zero
                row['pair_confidence'] = post[pa]
                if t==cut:next_rows[q['query_id']]=row
                else:rows.append(row)
    return rows,next_rows


def high_probabilities(queries,params):
    cfg=dict(DEFAULT_CFG,**params['cfg']);lp=_pair_prior(cfg);marg=lag_prior_marginal(cfg);by=defaultdict(list);out={}
    for q in queries:by[q['episode_id']].append(q)
    for qs in by.values():
        machines=[];short=sum(len(q['seq']) for q in qs)<.6*sum(q['cutoff'] for q in qs)
        for q in qs:
            seq=q['seq'];cut=q['cutoff'];se=SizeEvidence(q,params['size_vectors'],cfg)
            wt={pa:lp[pa]+se.factor_and_r1(lag_prior_for_pair(pa,cfg,marg),cfg)[0] for pa in lp};norm=logsumexp(list(wt.values()))
            m=WinMachine(q,cfg,{pa:v-norm for pa,v in wt.items()},stable_only=short)
            cp=cfg['cell_prior'][q['machine_type']];pri=[]
            for nm in ('hi','lo'):
                mu,slope,k=cp[nm];pr=1/(1+math.exp(-(mu+slope*q['signal'])));pri.append((k*pr,k*(1-pr)))
            def probability(cells,active):
                terms=[]
                for ori in (0,1):
                    terms.append(sum(betaln(pri[z^ori][0]+w,pri[z^ori][1]+l)-betaln(*pri[z^ori]) for z,(l,w) in enumerate(cells)))
                p0=math.exp(terms[0]-logsumexp(terms))
                return p0 if active==0 else 1-p0
            for pa in m.pairs:
                a,b=pa;ticks=sorted(t for t in seq if t-a in seq and t-b in seq);pref=[[[0,0],[0,0]]]
                for t in ticks:
                    new=[x[:] for x in pref[-1]];new[int((seq[t-a]>0)==(seq[t-b]>0))][int(seq[t]>0)]+=1;pref.append(new)
                total=pref[-1]
                if cut-a not in seq or cut-b not in seq:zc=.5
                else:zc=int((seq[cut-a]>0)==(seq[cut-b]>0))
                m.pred_stable[pa]=probability(total,zc) if zc!=.5 else .5
                j=0;values=[]
                for tau in m.taus:
                    while j<len(ticks) and ticks[j]<tau:j+=1
                    pre=pref[j];cells=[[pre[z][y]+total[1-z][y]-pre[1-z][y] for y in (0,1)] for z in (0,1)]
                    values.append(probability(cells,1-zc) if zc!=.5 else .5)
                m.pred_tau[pa]=values
            machines.append(m)
        probabilities,_=episode_posterior_joint(machines,cfg)
        out.update({q['query_id']:v for q,v in zip(qs,probabilities)})
    return out


def predict(queries,base,params,config):
    rows,next_rows=state_rows(queries,params);high=high_probabilities(queries,params);cf=config['coef'];types=config['types'];times=defaultdict(lambda:defaultdict(list));cutoffs={}
    def eta(row,state=None,regime=None):
        i=types.index(row['type']);state=row['high'] if state is None else state;regime=row['regime'] if regime is None else regime
        return cf[2*i+state]+cf[6]*row['signal']+cf[7+i]*(regime-.5)+cf[10]*(row['peer']-3.5)
    for row in rows:
        times[row['episode']][row['tick']].append((eta(row),row['y']));cutoffs[row['episode']]=row['cutoff']
    hot={ep:filter_episode((cutoffs[ep],ts),config['scale'],config['hazard'])[1] for ep,ts in times.items()}
    result={}
    for q in queries:
        key=q['query_id'];old=base[key];win=1-old[0]
        if key not in next_rows or win<=1e-12:result[key]=old;continue
        V=params['size_vectors'][q['machine_type']];eps=params['cfg'].get('size_eps',.02);v0,v1=[[(1-eps)*V[r][k]+eps*V[1-r][k] for k in range(4)] for r in (0,1)]
        conditional=[v/win for v in old[1:]];den=sum((b-a)**2 for a,b in zip(v0,v1));r1=min(1.,max(0.,sum((v-a)*(b-a) for v,a,b in zip(conditional,v0,v1))/den))
        h=high[key];weather=hot.get(q['episode_id'],.5);mass=[0.,0.]
        for state,ps in ((0,1-h),(1,h)):
            for regime,pr in ((0,1-r1),(1,r1)):
                e=eta(next_rows[key],state,regime)
                prob=(1-weather)/(1+math.exp(-(e-config['scale'])))+weather/(1+math.exp(-(e+config['scale'])))
                mass[regime]+=ps*pr*prob
        new=sum(mass);residual=[v-((1-r1)*a+r1*b) for v,a,b in zip(conditional,v0,v1)]
        result[key]=[1-new]+[mass[0]*a+mass[1]*b+new*d for a,b,d in zip(v0,v1,residual)]
    return result
