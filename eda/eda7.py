import numpy as np, pandas as pd, collections
exec(open('eda/eda3.py').read().split('q,S=seqs')[0])
SZ={'balanced':[(0.5,0.348,0.13,0.022),(0.214,0.547,0.205,0.034)],
    'jackpot':[(0.335,0.245,0.315,0.105),(0.121,0.324,0.416,0.139)],
    'steady':[(0.754,0.197,0.048,0.001),(0.455,0.436,0.107,0.002)]}
IDX={1:0,2:1,5:2,20:3}
q,S=seqs('validation'); t=pd.read_csv(P+'validation/targets.csv').set_index('query_id'); c=pd.read_csv(P+'validation/conditions.csv').set_index('query_id').condition
res=[]
for _,r in q.iterrows():
    a=S[(r.episode_id,r.machine_id)][:r.cutoff]; tt=t.loc[r.query_id]; w=1-tt.p_0
    true=int(abs(tt.p_1/w-SZ[r.machine_type][1][0])<abs(tt.p_1/w-SZ[r.machine_type][0][0]))
    L0,L1=np.log(SZ[r.machine_type][0]),np.log(SZ[r.machine_type][1])
    ticks=np.where(a>0)[0]
    llr=np.array([L1[IDX[a[k]]]-L0[IDX[a[k]]] for k in ticks])
    rec=ticks>=r.cutoff-40
    res.append(dict(cond=c[r.query_id],type=r.machine_type,true=true,all=int(llr.sum()>0),recent=int(llr[rec].sum()>0),nwin=len(ticks),
        first_half=int(llr[ticks<r.cutoff/2].sum()>0), second=int(llr[ticks>=r.cutoff/2].sum()>0)))
d=pd.DataFrame(res)
for k in ['all','recent','first_half','second']: d['acc_'+k]=(d[k]==d.true)
print(d.groupby('cond')[[c for c in d if c.startswith('acc')]].mean().round(3))
# does regime differ within an episode
print(d.assign(ep=q.episode_id).groupby(['cond','ep']).true.mean().groupby('cond').apply(lambda s: s.round(2).value_counts().head(5).to_dict()))
