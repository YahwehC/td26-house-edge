import numpy as np, pandas as pd, collections, itertools
exec(open('eda/eda7.py').read().split('res=[]')[0])
out=[]
for _,r in q.iterrows():
    a=S[(r.episode_id,r.machine_id)][:r.cutoff]; cu=r.cutoff; tt=t.loc[r.query_id]; w=1-tt.p_0
    true=int(abs(tt.p_1/w-SZ[r.machine_type][1][0])<abs(tt.p_1/w-SZ[r.machine_type][0][0]))
    L0,L1=np.log(SZ[r.machine_type][0]),np.log(SZ[r.machine_type][1])
    T=np.where(a>0)[0]; T=T[T>=8]; llr=np.array([L1[IDX[a[k]]]-L0[IDX[a[k]]] for k in T])
    best=None
    for L in range(1,9):
        m=a[T-L]>=0; x=a[T[m]-L]>=2; l=llr[m]
        s1,s0=l[x].sum(),l[~x].sum(); g=abs(s1)+abs(s0)
        if best is None or g>best[0]: best=(g,L,s1,s0)
    g,L,s1,s0=best
    v=a[cu-L]
    pred=None if v<0 else int((s1 if v>=2 else s0)>0)
    out.append(dict(cond=c[r.query_id],L=L,true=true,pred=pred,g=g,signal=r.signal,type=r.machine_type,flip=int(np.sign(s1)!=np.sign(s0))))
d=pd.DataFrame(out); d['ok']=(d.pred==d.true)
print(d.groupby('cond').agg(acc=('ok','mean'),nmiss=('pred',lambda s:s.isna().sum())))
print(d[d.cond=='stable'].groupby('L').ok.agg(['mean','size']))
print('flip frac',d.groupby('cond').flip.mean())
