import numpy as np, pandas as pd, itertools, collections
exec(open('eda/eda3.py').read().split('q,S=seqs')[0])
PAIRS=[(1,3),(2,5),(4,7)]
q,S=seqs('validation'); cond=pd.read_csv(P+'validation/conditions.csv').set_index('query_id').condition
tg=pd.read_csv(P+'validation/targets.csv').set_index('query_id')
rows=[]
for _,r in q.iterrows():
    a=S[(r.episode_id,r.machine_id)][:r.cutoff]; c=r.cutoff
    T=np.arange(8,c); T=T[a[T]>=0]
    best=None
    for A,B in PAIRS:
        m=(a[T-A]>=0)&(a[T-B]>=0); TT=T[m]; y=(a[TT]>0).astype(int); same=(a[TT-A]>0)==(a[TT-B]>0)
        g=ll(y[same])+ll(y[~same])-ll(y)
        if best is None or g>best[0]: best=(g,A,B,TT,y,same)
    g,A,B,TT,y,same=best
    if a[c-A]<0 or a[c-B]<0: st=None
    else: st=(a[c-A]>0)==(a[c-B]>0)
    t=tg.loc[r.query_id]; tw=1-t.p_0
    ew=y[same==st].mean() if st is not None else np.nan
    # size dist among wins in history (all) vs target
    wins=a[T][a[T]>0]; emp=[(wins==v).mean() for v in (1,2,5,20)]
    tsz=[t[k]/tw if tw>1e-9 else np.nan for k in ['p_1','p_2','p_5','p_20']]
    # size dist among wins in same state
    ws=a[TT][(same==st)&(y==1)] if st is not None else np.array([])
    emps=[(ws==v).mean() if len(ws) else np.nan for v in (1,2,5,20)]
    rows.append(dict(cond=cond[r.query_id],type=r.machine_type,sig=r.signal,gain=g,true_w=tw,emp_w=ew,
        sz_err_all=np.nansum((np.array(emp)-np.array(tsz))**2), sz_err_state=np.nansum((np.array(emps)-np.array(tsz))**2),
        t1=tsz[0],t2=tsz[1],t5=tsz[2],t20=tsz[3]))
d=pd.DataFrame(rows); d['werr']=(d.true_w-d.emp_w)**2
print(d.groupby('cond')[['werr','sz_err_all','sz_err_state','gain']].mean())
print(d[d.cond=='stable'][['type','sig','true_w','emp_w','t1','t2','t5','t20']].head(25).round(3).to_string())
print(d[d.true_w>0.05].groupby(['type','sig'])[['t1','t2','t5','t20']].agg(['mean','std']).round(3))
