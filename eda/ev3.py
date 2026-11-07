import sys, math
sys.path.insert(0,'.')
from he.model import *
from he.model import _pair_prior
qv,ev=load_split('kit/public','validation')
cfg=dict(DEFAULT_CFG); plp=_pair_prior(cfg)
for suffix in ['87-m006-next','67-m002-next']:
    q=[q for q in qv if q['query_id'].endswith(suffix)][0]
    m=WinMachine(q,cfg,plp)
    for p in m.pairs:
        i=max(range(len(m.taus)),key=lambda i:m.lml_tau[p][i])
        print(suffix,p,'stable lml %.1f pred %.2f'%(m.lml_stable[p],m.pred_stable[p]),'best tau',m.taus[i],'lml %.1f pred %.2f'%(m.lml_tau[p][i],m.pred_tau[p][i]), 'cutoff',q['cutoff'])
    s=q['seq']; c=q['cutoff']
    print(''.join('W' if s.get(t,-1)>0 else ('.' if t in s else '?') for t in range(c-80,c)))
