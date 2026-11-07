import numpy as np, pandas as pd, itertools, collections
P='kit/public/'
def seqs(split):
    h=pd.read_csv(P+split+'/history.csv'); q=pd.read_csv(P+split+'/queries.csv')
    out={}
    for (e,m),g in h.groupby(['episode_id','machine_id']):
        a=np.full(260,-1,int); a[g.tick.values]=g.payout.values; out[(e,m)]=a
    return q,out
def ll(y):  # bernoulli log-lik of counts
    n=len(y); k=y.sum()
    if k==0 or k==n: return 0.0
    p=k/n; return k*np.log(p)+(n-k)*np.log(1-p)
q,S=seqs('train')
cond=pd.read_csv(P+'train/conditions.csv').set_index('query_id').condition
res=[]
for _,r in q.iterrows():
    if cond[r.query_id]=='short_history': continue
    a=S[(r.episode_id,r.machine_id)][:r.cutoff]
    T=np.arange(8,r.cutoff); ok=(a[T]>=0)
    for L in range(1,9): ok&=a[T-L]>=0
    T=T[ok]; y=(a[T]>0).astype(int); base=ll(y)
    gains={}
    for L in range(1,9):
        x=a[T-L]>0; gains[(L,)]=ll(y[x])+ll(y[~x])-base
    for A,B in itertools.combinations(range(1,9),2):
        xa=a[T-A]>0; xb=a[T-B]>0; g=-base
        for u in (0,1):
            for v in (0,1):
                g+=ll(y[(xa==u)&(xb==v)])
        gains[(A,B)]=g
    best=max(gains,key=gains.get); s=sorted(gains.values())
    best1=max([k for k in gains if len(k)==1],key=gains.get)
    res.append((r.machine_type,cond[r.query_id],len(T),best,round(gains[best],1),best1,round(gains[best1],1),round(s[-2],1)))
df=pd.DataFrame(res,columns=['type','cond','n','best','gain','best1','gain1','second'])
print(df.head(40).to_string())
print(df.groupby('type')[['gain','gain1','second']].median())
print(collections.Counter(len(b) for b in df.best))
