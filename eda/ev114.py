"""Matched-mask train test of jointly filtering missing own history and weather.
No new scalar parameters: fixed peer emissions, existing scale/hazard/offset.
"""
import sys,json,pickle,math,time,hashlib
from pathlib import Path
sys.path.insert(0,'.')
from he.api import load_params
from sparse_weather_numpy import predict
file=Path('eda/_pshort_matched.pkl');data=pickle.loads(file.read_bytes());cache=pickle.load(open('eda/_pshort_matched_predictions.pkl','rb'))
assert data['schema']==3 and cache['schema']==3 and cache['input_sha256']==hashlib.sha256(file.read_bytes()).hexdigest()
qs,ys=data['queries'],data['targets'];baseline=cache['predictions']['v13'];p=load_params('runs/main_v13/params.json');eps=sorted({q['episode_id'].rsplit('-r',1)[0] for q in qs});means={};base={};start=time.perf_counter()
for ep in eps:
    batch=[q for q in qs if q['episode_id'].rsplit('-r',1)[0]==ep];pred=predict(batch,p)
    means[ep]=sum(sum((v-(i==ys[q['query_id']]))**2 for i,v in enumerate(pred[q['query_id']])) for q in batch)/len(batch)
    base[ep]=sum(sum((v-(i==ys[q['query_id']]))**2 for i,v in enumerate(baseline[q['query_id']])) for q in batch)/len(batch)
    print('episodes',len(means),'of',len(eps),'seconds',time.perf_counter()-start,flush=True)
d=[means[e]-base[e] for e in eps];m=sum(d)/len(d)
report={'benchmark':data['name'],'delta':m,'cluster_se':math.sqrt(sum((v-m)**2 for v in d)/(len(d)-1)/len(d)),'improved':sum(v<0 for v in d),'episodes':len(d),'means':means,'baseline':base}
print({k:v for k,v in report.items() if k not in ('means','baseline')},flush=True);Path('eda/ev114_joint_sparse_weather.json').write_text(json.dumps(report,indent=2))
