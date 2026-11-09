"""Cross-fit convex v12/v13 combination on independent original train episodes."""
import json,pickle,hashlib,math
from pathlib import Path
from collections import defaultdict
report={}
for variant in ('matched','independent'):
    path=Path(f'eda/_pshort_{variant}.pkl');data=pickle.loads(path.read_bytes());cache=pickle.load(open(f'eda/_pshort_{variant}_predictions.pkl','rb'));assert cache['input_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    queries,ys=data['queries'],data['targets'];a,b=cache['predictions']['v13'],cache['predictions']['v12'];moments=defaultdict(lambda:[0.,0.]);by=defaultdict(list)
    for q in queries:
        key=q['query_id'];ep=q['episode_id'].rsplit('-r',1)[0];by[ep].append(key);d=[x-y for x,y in zip(a[key],b[key])]
        moments[ep][0]+=sum(v*v for v in d);moments[ep][1]+=sum(v*((j==ys[key])-b[key][j]) for j,v in enumerate(d))
    eps=sorted(by);weights=[];losses={};baseline={}
    for fold in range(3):
        train=[ep for i,ep in enumerate(eps) if i%3!=fold];weight=max(0.,min(1.,sum(moments[e][1] for e in train)/sum(moments[e][0] for e in train)));weights.append(weight)
        for i,ep in enumerate(eps):
            if i%3!=fold:continue
            values=[];pure=[]
            for key in by[ep]:
                prediction=[weight*x+(1-weight)*y for x,y in zip(a[key],b[key])];values.append(sum((x-(j==ys[key]))**2 for j,x in enumerate(prediction)));pure.append(sum((x-(j==ys[key]))**2 for j,x in enumerate(a[key])))
            losses[ep]=sum(values)/len(values);baseline[ep]=sum(pure)/len(pure)
    d=[losses[e]-baseline[e] for e in eps];mean=sum(d)/len(d)
    report[variant]={'v13_fold_weights':weights,'full_weight':max(0.,min(1.,sum(v[1] for v in moments.values())/sum(v[0] for v in moments.values()))),'delta_vs_v13':mean,'cluster_se':math.sqrt(sum((x-mean)**2 for x in d)/(len(d)-1)/len(d)),'improved':sum(x<0 for x in d),'episodes':len(d)}
    print(variant,report[variant],flush=True)
Path('eda/ev112_structural_blend.json').write_text(json.dumps(report,indent=2))
