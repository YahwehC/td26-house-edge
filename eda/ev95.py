"""Rerun rejected sparse-history mechanisms with episode-isolated offsets.
Supersedes contaminated pseudo results ev83, ev84, ev88; baseline is pure main_v12.
"""
import sys,json,pickle,math,time,copy
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,'.')
from he.api import load_params,predict_queries
qs,ys=pickle.load(open('eda/_pshort.pkl','rb'));cache=pickle.load(open('eda/_v12_pshort_v2.pkl','rb'));assert cache['version']==2;baseline=cache['main']
p=load_params('runs/main_v12/params.json');originals=sorted({q['episode_id'].rsplit('-r',1)[0] for q in qs});base_loss=defaultdict(list)
for q in qs:
    k=q['query_id'];ep=q['episode_id'].rsplit('-r',1)[0]
    base_loss[ep].append(sum((x-(i==ys[k]))**2 for i,x in enumerate(baseline[k])))
base={ep:sum(v)/len(v) for ep,v in base_loss.items()};report={}
for variant in ('missing_peer','history_weather','short_joint_weather'):
    candidate=copy.deepcopy(p)
    if variant=='missing_peer':candidate['cfg']['marginalize_missing_peers']=True
    else:candidate['global_state'][variant]=True
    loss=defaultdict(list);start=time.perf_counter()
    for begin in range(0,len(originals),5):
        selected=set(originals[begin:begin+5]);batch=[q for q in qs if q['episode_id'].rsplit('-r',1)[0] in selected];pred=predict_queries(batch,candidate)
        for q in batch:
            k=q['query_id'];ep=q['episode_id'].rsplit('-r',1)[0]
            loss[ep].append(sum((x-(i==ys[k]))**2 for i,x in enumerate(pred[k])))
        print(variant,'episodes',len(loss),'seconds',time.perf_counter()-start,flush=True)
    means={ep:sum(v)/len(v) for ep,v in loss.items()};d=[means[e]-base[e] for e in originals];m=sum(d)/len(d)
    report[variant]={'delta':m,'cluster_se':math.sqrt(sum((v-m)**2 for v in d)/(len(d)-1)/len(d)),'improved':sum(v<0 for v in d),'episodes':len(d),'means':means,'baseline':base}
    print(variant,{k:v for k,v in report[variant].items() if k not in ('means','baseline')},flush=True)
    Path('eda/ev95_isolated_mechanisms.json').write_text(json.dumps(report,indent=2))
