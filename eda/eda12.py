import numpy as np, pandas as pd
exec(open('eda/eda3.py').read().split('q,S=seqs')[0])
PAIRS=[(1,3),(2,5),(4,7)]
q,S=seqs('train'); c=pd.read_csv(P+'train/conditions.csv').set_index('query_id').condition
for cond in ['drift','shared_change']:
    eps=q[q.query_id.map(c)==cond].episode_id.unique()[:3]
    for e in eps:
        print('==',cond,e)
        for _,r in q[q.episode_id==e].iterrows():
            a=S[(e,r.machine_id)][:r.cutoff]; line=[]
            for A,B in PAIRS:
                s=''
                for w0 in range(8,r.cutoff-31,32):
                    T=np.arange(w0,w0+32); T=T[(a[T]>=0)&(a[T-A]>=0)&(a[T-B]>=0)]
                    y=a[T]>0; same=(a[T-A]>0)==(a[T-B]>0)
                    dlt=(y[same].mean() if same.any() else .5)-(y[~same].mean() if (~same).any() else .5)
                    s+='%+.1f '%dlt
                line.append(f'{A}{B}: '+s)
            wr=' '.join('%.2f'%((a[w0:w0+32]>0).mean()) for w0 in range(8,r.cutoff-31,32))
            print(r.machine_type[:4],r.signal,' | '.join(line),'| win',wr)
