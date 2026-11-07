import sys, math
sys.path.insert(0,'.')
from he.model import *
qv,ev=load_split('kit/public','validation')
vecs=fit_size_vectors(qv,ev['targets'])
pred=predict(qv,vecs,{})
cond={r['query_id']:r['condition'] for r in ev['conditions']}
tg={r['query_id']:[float(r[k]) for k in ('p_0','p_1','p_2','p_5','p_20')] for r in ev['targets']}
qm={q['query_id']:q for q in qv}
rows=[]
for qid,p in pred.items():
    t=tg[qid]; e=sum((a-b)**2 for a,b in zip(p,t)); rows.append((e,qid))
rows.sort(reverse=True)
for e,qid in [r for r in rows if cond[r[1]]=='stable'][:12]:
    q=qm[qid]; s=q['seq']; c=q['cutoff']
    print(round(e,3),qid[-12:],q['machine_type'],'pred',[round(x,2) for x in pred[qid]],'true',[round(x,2) for x in tg[qid]],'last8',[s.get(c-L,-1) for L in range(1,9)])
import collections
print(collections.Counter((cond[q],round(e,1)) for e,q in rows if e>0.1))
