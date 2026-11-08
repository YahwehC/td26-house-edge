"""Correct ev85 after fixing offsets to be local to (episode, machine).
Cross-fit one convex weight by ORIGINAL episode; no validation targets used.
Old pseudo caches had merged offsets across resampled episodes: do not reuse them.
"""
import sys,json,pickle,math,time
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,'.')
from he.api import load_params,predict_queries
from he.fallback import predict_fallback
qs,ys=pickle.load(open('eda/_pshort.pkl','rb'));p=load_params('runs/main_v11/params.json');p['blend']['w_main']=1.
start=time.perf_counter();main=predict_queries(qs,p);print('main done',time.perf_counter()-start,flush=True)
fallback=predict_fallback(qs,p['fallback']);print('fallback done',time.perf_counter()-start,flush=True)
pickle.dump({'version':2,'main':main,'fallback':fallback},open('eda/_v12_pshort_v2.pkl','wb'))
by=defaultdict(list)
for q in qs:
    k=q['query_id'];ep=q['episode_id'].rsplit('-r',1)[0]
    a,b=main[k],fallback[k];d=[x-y for x,y in zip(a,b)]
    by[ep].append((sum(x*x for x in d),sum(x*((j==ys[k])-b[j]) for j,x in enumerate(d)),k))
eps=sorted(by);weights=[];losses=defaultdict(dict)
for fold in range(3):
    train=[row for i,ep in enumerate(eps) if i%3!=fold for row in by[ep]]
    weight=max(0.,min(1.,sum(r[1] for r in train)/sum(r[0] for r in train)));weights.append(weight)
    for i,ep in enumerate(eps):
        if i%3!=fold:continue
        for name,w in [('baseline',.9),('pure',1.),('crossfit',weight)]:
            vals=[]
            for _,__,k in by[ep]:
                pr=[w*a+(1-w)*b for a,b in zip(main[k],fallback[k])]
                vals.append(sum((v-(j==ys[k]))**2 for j,v in enumerate(pr)))
            losses[name][ep]=sum(vals)/len(vals)
report={'fold_weights':weights,'full_weight':max(0.,min(1.,sum(r[1] for group in by.values() for r in group)/sum(r[0] for group in by.values() for r in group))),'comparisons':{},'losses':losses}
for name in ('pure','crossfit'):
    d=[losses[name][ep]-losses['baseline'][ep] for ep in eps];m=sum(d)/len(d)
    report['comparisons'][name]={'delta':m,'cluster_se':math.sqrt(sum((v-m)**2 for v in d)/(len(d)-1)/len(d)),'improved':sum(v<0 for v in d)}
print({k:v for k,v in report.items() if k!='losses'},flush=True)
report['audit']='Offsets isolated by (episode,machine); supersedes ev85.'
Path('eda/ev94_blend_isolated.json').write_text(json.dumps(report,indent=2))
