"""Frozen missing-peer correction on training pseudo-short outcomes."""
import sys,json,pickle,copy,math,time
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,'.')
from he.api import load_params,predict_queries
qs,ys=pickle.load(open('eda/_pshort.pkl','rb'));p=load_params('runs/main_v11/params.json');means={}
for name in ('baseline','missing_peer'):
    candidate=copy.deepcopy(p)
    if name=='missing_peer':candidate['cfg']['marginalize_missing_peers']=True
    start=time.perf_counter();pr=predict_queries(qs,candidate);loss=defaultdict(list)
    for q in qs:
        key=q['query_id'];ep=q['episode_id'].rsplit('-r',1)[0]
        loss[ep].append(sum((x-(i==ys[key]))**2 for i,x in enumerate(pr[key])))
    means[name]={ep:sum(v)/len(v) for ep,v in loss.items()};print(name,sum(means[name].values())/len(loss),time.perf_counter()-start,flush=True)
d=[means['missing_peer'][e]-means['baseline'][e] for e in means['baseline']];m=sum(d)/len(d)
report={'delta':m,'cluster_se':math.sqrt(sum((v-m)**2 for v in d)/(len(d)-1)/len(d)),'improved':sum(v<0 for v in d),'episodes':len(d),'means':means}
print(report,flush=True);Path('eda/ev83_missing_peers.json').write_text(json.dumps(report,indent=2))
