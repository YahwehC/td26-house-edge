import numpy as np, pandas as pd, collections, itertools
exec(open('eda/eda7.py').read().split('res=[]')[0])
feats={'win':lambda v:v>0,'ge2':lambda v:v>=2,'ge5':lambda v:v>=5}
out=[]
for _,r in q.iterrows():
    if c[r.query_id]!='stable': continue
    a=S[(r.episode_id,r.machine_id)][:r.cutoff]
    L0,L1=np.log(SZ[r.machine_type][0]),np.log(SZ[r.machine_type][1])
    T=np.where(a>0)[0]; T=T[T>=8]
    ok=np.ones(len(T),bool)
    for L in range(1,9): ok&=a[T-L]>=0
    T=T[ok]; llr=np.array([L1[IDX[a[k]]]-L0[IDX[a[k]]] for k in T])
    base=abs(llr.sum()); best=(0,None)
    for fn,f in feats.items():
        for L in range(1,9):
            x=f(a[T-L]); g=abs(llr[x].sum())+abs(llr[~x].sum())-base
            if g>best[0]: best=(g,(fn,L))
        for A,B in itertools.combinations(range(1,9),2):
            x=f(a[T-A])==f(a[T-B]); g=abs(llr[x].sum())+abs(llr[~x].sum())-base
            if g>best[0]: best=(g,(fn,A,B))
    out.append((r.machine_type,len(T),round(base,1),round(best[0],1),best[1]))
d=pd.DataFrame(out,columns=['type','nwin','base','gain','feat'])
print(d.head(30).to_string()); print(collections.Counter(d.feat).most_common(12)); print(d.gain.describe())
