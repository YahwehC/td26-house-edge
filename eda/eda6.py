import numpy as np, pandas as pd, collections
P='kit/public/validation/'
q=pd.read_csv(P+'queries.csv'); t=pd.read_csv(P+'targets.csv'); c=pd.read_csv(P+'conditions.csv')
d=q.merge(t,on=['query_id','episode_id']).merge(c,on='query_id')
d['w']=1-d.p_0
for k in ['p_1','p_2','p_5','p_20']: d['s'+k[2:]]=d[k]/d.w
d['sz']=d.apply(lambda r:(round(r.s1,3),round(r.s2,3),round(r.s5,3),round(r.s20,3)),axis=1)
for ty,g in d.groupby('machine_type'):
    print(ty, collections.Counter(g.sz).most_common(6))
# win-prob values
print(np.sort(d.w.round(3).values)[::12])
# logit of w
lw=np.log(d.w/(1-d.w)); print('logit w quantiles',np.percentile(lw,[0,5,10,25,50,75,90,95,100]).round(2))
# per condition, fraction of episodes with mixed size regimes
d['reg']=d.groupby('machine_type').sz.transform(lambda s: s.map({v:i for i,v in enumerate(collections.Counter(s).most_common())}) )
