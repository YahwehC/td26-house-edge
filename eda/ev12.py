import sys, collections
sys.path.insert(0,'.')
from he.model import *
from he.model import _pair_prior
from he.fit import fit_cell_prior, fit_structure_priors
qt,et=load_split('kit/public','train'); qv,ev=load_split('kit/public','validation')
ct={r['query_id']:r['condition'] for r in et['conditions']}
vecs=fit_size_vectors(qv,ev['targets']); sp=fit_structure_priors([(qt,ct)],vecs)
cfg=dict(DEFAULT_CFG,**{k:v for k,v in sp.items() if not k.startswith('_')}); cfg['cell_prior']=fit_cell_prior([(qt,ct)])
plp=_pair_prior(cfg)
eps=collections.defaultdict(list)
for q in qt: eps[q['episode_id']].append(q)
H={'drift':collections.Counter(),'shared_change':collections.Counter()}; post=collections.defaultdict(list)
for e,qs in eps.items():
    cd=ct[qs[0]['query_id']]
    ms=[WinMachine(q,cfg,plp) for q in sorted(qs,key=lambda q:q['machine_id'])]
    _,d=episode_posterior(ms,cfg); post[cd].append({k:round(v,2) for k,v in d['post'].items()})
    c=ms[0].q['cutoff']; taus=d['taus']
    if cd=='drift':
        for tp in d['tpost_drift']:
            for tau,w in zip(taus,tp): H[cd][(c-tau)//8]+=w
    if cd=='shared_change' and d['tpost_shared']:
        for tau,w in zip(taus,d['tpost_shared']): H[cd][(c-tau)//8]+=w
for k,h in H.items():
    tot=sum(h.values()); print(k,[(b*8,round(h[b]/tot,3)) for b in sorted(h)])
for k,v in post.items(): print(k,v[:12])
