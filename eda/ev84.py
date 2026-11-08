"""Frozen peer-only weather smoothing, paired training pseudo-short comparison.
Baseline aggregate predictions are exactly the main_v11 run from ev83.
"""
import sys,json,pickle,math,time
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,'.')
from he.api import load_params,predict_queries
qs,ys=pickle.load(open('eda/_pshort.pkl','rb'))
p=load_params('runs/main_v11/params.json');p['global_state']['history_weather']=True
baseline=json.load(open('eda/ev83_missing_peers.json'))['means']['baseline']
loss=defaultdict(list);start=time.perf_counter()
# Batches keep original episodes together and provide observable progress.
originals=sorted(baseline)
for begin in range(0,len(originals),5):
    selected=set(originals[begin:begin+5])
    batch=[q for q in qs if q['episode_id'].rsplit('-r',1)[0] in selected]
    pred=predict_queries(batch,p)
    for q in batch:
        key=q['query_id'];ep=q['episode_id'].rsplit('-r',1)[0]
        loss[ep].append(sum((x-(i==ys[key]))**2 for i,x in enumerate(pred[key])))
    print('episodes',len(loss),'seconds',time.perf_counter()-start,flush=True)
means={ep:sum(v)/len(v) for ep,v in loss.items()}
d=[means[e]-baseline[e] for e in originals];m=sum(d)/len(d)
report={'delta':m,'cluster_se':math.sqrt(sum((v-m)**2 for v in d)/(len(d)-1)/len(d)),'improved':sum(v<0 for v in d),'episodes':len(d),'means':means}
print({k:v for k,v in report.items() if k!='means'},flush=True)
Path('eda/ev84_history_weather.json').write_text(json.dumps(report,indent=2))
