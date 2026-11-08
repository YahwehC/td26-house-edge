"""Train-only weather cavity messages under exact sparse machine hypotheses.
Compare to isolated main_v13 exact-recursion outcomes from ev96; no validation.
"""
import sys,json,pickle,math,time
from pathlib import Path
sys.path.insert(0,'.')
from he.api import load_params
from weather_cavity_numpy import predict
qs,ys=pickle.load(open('eda/_pshort.pkl','rb'));base=json.load(open('eda/ev96_exact_sparse.json'))['means'];p=load_params('runs/main_v13/params.json');originals=sorted(base);means={};start=time.perf_counter()
for ep in originals:
    batch=[q for q in qs if q['episode_id'].rsplit('-r',1)[0]==ep];pred=predict(batch,p)
    means[ep]=sum(sum((v-(i==ys[q['query_id']]))**2 for i,v in enumerate(pred[q['query_id']])) for q in batch)/len(batch)
    print('episodes',len(means),'of',len(originals),'seconds',time.perf_counter()-start,flush=True)
d=[means[e]-base[e] for e in originals];m=sum(d)/len(d)
report={'delta':m,'cluster_se':math.sqrt(sum((v-m)**2 for v in d)/(len(d)-1)/len(d)),'improved':sum(v<0 for v in d),'episodes':len(d),'means':means,'baseline':base}
print({k:v for k,v in report.items() if k not in ('means','baseline')},flush=True);Path('eda/ev102_cavity_weather.json').write_text(json.dumps(report,indent=2))
