import numpy as np, pandas as pd, itertools, collections
exec(open('eda/eda3.py').read().split('q,S=seqs')[0])
for split in ['train','validation']:
    q,S=seqs(split); cond=pd.read_csv(P+split+'/conditions.csv').set_index('query_id').condition
    cnt=collections.Counter(); cells=collections.defaultdict(list)
    for _,r in q.iterrows():
        if cond[r.query_id]!='stable': continue
        a=S[(r.episode_id,r.machine_id)][:r.cutoff]
        T=np.arange(8,r.cutoff); ok=(a[T]>=0)
        for L in range(1,9): ok&=a[T-L]>=0
        T=T[ok]; y=a[T]
        best=None
        for A,B in itertools.combinations(range(1,9),2):
            xa=a[T-A]>0; xb=a[T-B]>0; g=0
            for u in (0,1):
                for v in (0,1): g+=ll((y[(xa==u)&(xb==v)]>0).astype(int))
            if best is None or g>best[0]: best=(g,A,B)
        _,A,B=best; cnt[(A,B)]+=1
        xa=a[T-A]>0; xb=a[T-B]>0
        row=[]
        for u in (0,1):
            for v in (0,1):
                m=(xa==u)&(xb==v); row.append(round((y[m]>0).mean(),2))
        if split=='train' and len(cells[(A,B)])<6: cells[(A,B)].append((r.machine_type,row))
    print(split,'stable best pairs',cnt.most_common())
    if split=='train':
        for k,v in cells.items(): print(k,v)
