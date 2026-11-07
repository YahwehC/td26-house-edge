import numpy as np, pandas as pd, collections
exec(open('eda/eda7.py').read().split('res=[]')[0])
cat=lambda v: -1 if v<0 else (0 if v==0 else (1 if v==1 else 2))
F=collections.defaultdict(list); Y=[]; CND=[]
eps={e:g for e,g in q.groupby('episode_id')}
for _,r in q.iterrows():
    a=S[(r.episode_id,r.machine_id)]; tt=t.loc[r.query_id]; w=1-tt.p_0; cu=r.cutoff
    Y.append(int(abs(tt.p_1/w-SZ[r.machine_type][1][0])<abs(tt.p_1/w-SZ[r.machine_type][0][0]))); CND.append(c[r.query_id])
    F['signal'].append(r.signal)
    for L in range(1,9):
        F[f'own_cat_L{L}'].append(cat(a[cu-L])); F[f'own_big_L{L}'].append(-1 if a[cu-L]<0 else int(a[cu-L]>=5))
    # episode-level: count of other machines won at lag L
    others=[S[(r.episode_id,m)] for m in eps[r.episode_id].machine_id if m!=r.machine_id]
    for L in range(1,9):
        vals=[o[cu-L] for o in others if o[cu-L]>=0]
        F[f'oth_win_L{L}'].append(round(np.mean([v>0 for v in vals])*4) if vals else -1)
        F[f'oth_bigsum_L{L}'].append(min(3,sum(v>=5 for v in vals)))
    F['mid'].append(int(r.machine_id[-1]) )
Y=np.array(Y); CND=np.array(CND)
def score(x,m):
    x=np.array(x)[m]; y=Y[m]; acc=0
    for v in np.unique(x):
        k=x==v; acc+=max(y[k].sum(),(1-y[k]).sum())
    return acc/m.sum(), len(np.unique(x))
for cond in ['stable','drift']:
    m=CND==cond
    s=sorted(((score(v,m),k) for k,v in F.items()),reverse=True)[:10]
    print(cond,'base',round(max(Y[m].mean(),1-Y[m].mean()),3),[(k,round(a,3),n) for (a,n),k in s])
