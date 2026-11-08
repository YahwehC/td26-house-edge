"""Training-only exact sparse inference with machine-offset posterior integration.
Fixed existing variance .04, five normal quadrature nodes. Compare all 6,000 cases
against the isolated exact-sparse predictions already evaluated in ev96.
"""
import sys,json,pickle,math,time
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,'.')
from he.api import load_params,predict_queries
import he.sparse_glm
from sparse_glm_numpy import predict as accelerated
he.sparse_glm.predict=accelerated
qs,ys=pickle.load(open('eda/_pshort.pkl','rb'));base=json.load(open('eda/ev96_exact_sparse.json'))['means'];p=load_params('runs/main_v13/params.json');p['global_state']['sparse_offset_quadrature']=True
originals=sorted(base);means={};start=time.perf_counter()
for ep in originals:
    batch=[q for q in qs if q['episode_id'].rsplit('-r',1)[0]==ep];pred=predict_queries(batch,p);loss=[]
    for q in batch:
        k=q['query_id'];loss.append(sum((x-(i==ys[k]))**2 for i,x in enumerate(pred[k])))
    means[ep]=sum(loss)/len(loss);print('episodes',len(means),'of',len(originals),'seconds',time.perf_counter()-start,flush=True)
d=[means[e]-base[e] for e in originals];m=sum(d)/len(d)
report={'delta':m,'cluster_se':math.sqrt(sum((v-m)**2 for v in d)/(len(d)-1)/len(d)),'improved':sum(v<0 for v in d),'episodes':len(d),'means':means,'baseline':base}
print({k:v for k,v in report.items() if k not in ('means','baseline')},flush=True);Path('eda/ev101_integrated_offset.json').write_text(json.dumps(report,indent=2))
