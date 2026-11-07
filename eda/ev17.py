import sys, math, collections
sys.path.insert(0,'.')
from hx.model import *
from hx.model import _pair_prior
from hx.api import fit_params
P=fit_params('kit/public',fit_splits=('train',)); cfg=dict(DEFAULT_CFG,**P['cfg']); plp=_pair_prior(cfg); vecs=P['size_vectors']
qs,ex=load_split('kit/public','train'); cond={r['query_id']:r['condition'] for r in ex['conditions']}
out=collections.Counter(); ex_=[]
for q in qs:
    if cond[q['query_id']]!='stable': continue
    m=WinMachine(q,cfg,plp); r=sorted(m.pairs,key=lambda p:-m.lml_stable[p])
    if m.lml_stable[r[0]]-m.lml_stable[r[1]]<=2: continue
    A=r[0][0]; s=q['seq']; V=vecs[q['machine_type']]; eps=0.02
    LV=[[math.log((1-eps)*V[x][k]+eps*V[1-x][k]) for k in range(4)] for x in (0,1)]
    wins=[(t,SIZE_IDX[v]) for t,v in s.items() if v>0]
    def ll(L): 
        xs=[(s[t-L]>=2,k) for t,k in wins if t-L in s]
        return max(math.fsum(LV[int(b)^g][k] for b,k in xs) for g in (0,1))
    const=max(math.fsum(LV[x][k] for _,k in wins) for x in (0,1))
    lA=ll(A); others=max(ll(L) for L in range(1,9) if L!=A)
    d=lA-others; dc=lA-const
    out[('A_wins' if d>0 else 'other_wins', 'vs_const>3' if dc>3 else 'vs_const<=3')]+=1
    if d<0: ex_.append((r[0],round(d,1),round(dc,1),len(wins), q['machine_type']))
print(out); print(ex_[:20])
