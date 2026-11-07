import numpy as np, pandas as pd, collections, itertools
exec(open('eda/eda7.py').read().split('res=[]')[0])
PAIRS=[(1,3),(2,5),(4,7)]
out=[]
for _,r in q.iterrows():
    if c[r.query_id] not in ('stable',): continue
    a=S[(r.episode_id,r.machine_id)][:r.cutoff]; cu=r.cutoff; tt=t.loc[r.query_id]; w=1-tt.p_0
    T=np.arange(8,cu); T=T[a[T]>=0]; best=None
    for A,B in PAIRS:
        m=(a[T-A]>=0)&(a[T-B]>=0); TT=T[m]; y=(a[TT]>0).astype(int); same=(a[TT-A]>0)==(a[TT-B]>0)
        g=ll(y[same])+ll(y[~same])-ll(y)
        if best is None or g>best[0]: best=(g,A,B,y[same].mean(),y[~same].mean(),same.sum(),(~same).sum())
    g,A,B,ps,pd_,ns,nd=best
    st=(a[cu-A]>0)==(a[cu-B]>0)
    hi_state = st if ps>pd_ else (not st)
    out.append(dict(type=r.machine_type,sig=r.signal,hi=int(hi_state),lw=np.log(w/(1-w)),
      lhi=np.log(max(ps,pd_)+1e-3)-np.log(1-max(ps,pd_)+1e-3),llo=np.log(min(ps,pd_)+1e-3)-np.log(1-min(ps,pd_)+1e-3),g=g,mid=r.machine_id[-4:],ep=r.episode_id))
d=pd.DataFrame(out)
print(d.groupby(['type','hi','sig']).lw.agg(['mean','std','size']).round(2))
