"""Exact missing payout-category recursion after episode isolation and peer fixes.
Research NumPy acceleration is checked against stdlib; shipping still uses stdlib.
"""
import sys,json,pickle,math,time
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,'.')
from he.api import load_params,predict_queries
import he.sparse_glm
from sparse_glm_numpy import predict as accelerated
he.sparse_glm.predict=accelerated
qs,ys=pickle.load(open('eda/_pshort.pkl','rb'));cache=pickle.load(open('eda/_v12_pshort_v2.pkl','rb'));assert cache['version']==2;baseline=cache['main']
p=load_params('runs/main_v12/params.json');p['cfg'].update(exact_sparse_glm=True,marginalize_missing_peers=True)
originals=sorted({q['episode_id'].rsplit('-r',1)[0] for q in qs});loss=defaultdict(list);base_loss=defaultdict(list);start=time.perf_counter()
for begin in range(0,len(originals),5):
    selected=set(originals[begin:begin+5]);batch=[q for q in qs if q['episode_id'].rsplit('-r',1)[0] in selected];pred=predict_queries(batch,p)
    for q in batch:
        k=q['query_id'];ep=q['episode_id'].rsplit('-r',1)[0]
        loss[ep].append(sum((x-(i==ys[k]))**2 for i,x in enumerate(pred[k])))
        base_loss[ep].append(sum((x-(i==ys[k]))**2 for i,x in enumerate(baseline[k])))
    print('episodes',len(loss),'seconds',time.perf_counter()-start,flush=True)
means={ep:sum(v)/len(v) for ep,v in loss.items()};base={ep:sum(v)/len(v) for ep,v in base_loss.items()};d=[means[e]-base[e] for e in originals];m=sum(d)/len(d)
report={'delta':m,'cluster_se':math.sqrt(sum((v-m)**2 for v in d)/(len(d)-1)/len(d)),'improved':sum(v<0 for v in d),'episodes':len(d),'means':means,'baseline':base}
print({k:v for k,v in report.items() if k not in ('means','baseline')},flush=True);Path('eda/ev96_exact_sparse.json').write_text(json.dumps(report,indent=2))
