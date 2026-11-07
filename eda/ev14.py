import sys, math, json, collections
sys.path.insert(0,'.')
from hx.api import *
from hx.model import official_score
p=fit_params('kit/public',fit_splits=('train',))
over=json.loads(sys.argv[1]) if len(sys.argv)>1 else {}
p['cfg'].update(over)
qv,ev,pred=predict_split('kit/public','validation',p)
print(official_score(pred,ev,qv))
cond={r['query_id']:r['condition'] for r in ev['conditions']}
tg={r['query_id']:[float(r[k]) for k in ('p_0','p_1','p_2','p_5','p_20')] for r in ev['targets']}
def mk(w,sz): return [1-w]+[w*s for s in sz]
acc=collections.defaultdict(lambda:[0,0,0,0])
for qid,pp in pred.items():
    t=tg[qid]; w=1-pp[0]; tw=1-t[0]
    sz=[x/w for x in pp[1:]]; tsz=[x/tw for x in t[1:]] if tw>1e-9 else sz
    e=lambda a:sum((x-y)**2 for x,y in zip(a,t))
    a=acc[cond[qid]]; a[0]+=e(pp); a[1]+=e(mk(tw,sz)); a[2]+=e(mk(w,tsz)); a[3]+=1
for c,a in sorted(acc.items()): print(c,'full %.5f  trueW(size err) %.5f  trueSize(win err) %.5f'%(a[0]/a[3],a[1]/a[3],a[2]/a[3]))
