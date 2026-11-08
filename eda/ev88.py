"""Train pseudo-short test of joint weather / machine-offset marginalization.
Fixed variance .04 and weather parameters; no new validation-fitted coefficients.
"""
import sys,json,pickle,math,time
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,'.')
from he.api import load_params,predict_queries
qs,ys=pickle.load(open('eda/_pshort.pkl','rb'));baseline,_=pickle.load(open('eda/_v11_pshort.pkl','rb'))
p=load_params('runs/main_v12/params.json');p['global_state']['short_joint_weather']=True
originals=sorted({q['episode_id'].rsplit('-r',1)[0] for q in qs});loss=defaultdict(list);base_loss=defaultdict(list);start=time.perf_counter()
for begin in range(0,len(originals),5):
    selected=set(originals[begin:begin+5]);batch=[q for q in qs if q['episode_id'].rsplit('-r',1)[0] in selected]
    pred=predict_queries(batch,p)
    for q in batch:
        k=q['query_id'];ep=q['episode_id'].rsplit('-r',1)[0]
        loss[ep].append(sum((x-(i==ys[k]))**2 for i,x in enumerate(pred[k])))
        base_loss[ep].append(sum((x-(i==ys[k]))**2 for i,x in enumerate(baseline[k])))
    print('episodes',len(loss),'seconds',time.perf_counter()-start,flush=True)
means={ep:sum(v)/len(v) for ep,v in loss.items()};base={ep:sum(v)/len(v) for ep,v in base_loss.items()}
d=[means[e]-base[e] for e in originals];m=sum(d)/len(d)
report={'delta':m,'cluster_se':math.sqrt(sum((v-m)**2 for v in d)/(len(d)-1)/len(d)),'improved':sum(v<0 for v in d),'episodes':len(d),'means':means,'baseline':base}
print({k:v for k,v in report.items() if k not in ('means','baseline')},flush=True)
Path('eda/ev88_joint_weather.json').write_text(json.dumps(report,indent=2))
