"""Independent-mask benchmarks: compare v12/v13 and refit fallback weights.
Default: matched cutoffs. Pass 'independent' to keep the old cutoff distribution.
Report by original stable train episode; no official validation targets are used.
"""
import sys,json,pickle,math,time,hashlib
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,'.')
from he.api import load_params,predict_queries
from he.fallback import predict_fallback
import he.sparse_glm
from sparse_glm_numpy import predict as accelerated
he.sparse_glm.predict=accelerated
variant='independent' if len(sys.argv)>1 and sys.argv[1]=='independent' else 'matched'
file=Path(f'eda/_pshort_{variant}.pkl');data=pickle.loads(file.read_bytes());assert data['schema']==3;qs,ys=data['queries'],data['targets']
p12=load_params('runs/main_v12/params.json');p13=load_params('runs/main_v13/params.json');fallback=load_params('runs/main_v11/params.json')['fallback'];eps=sorted({q['episode_id'].rsplit('-r',1)[0] for q in qs});by=defaultdict(list)
for q in qs:by[q['episode_id'].rsplit('-r',1)[0]].append(q)
predictions={name:{} for name in ('v12','v13','fallback')};start=time.perf_counter()
for ep in eps:
    batch=by[ep];predictions['v12'].update(predict_queries(batch,p12));predictions['v13'].update(predict_queries(batch,p13));predictions['fallback'].update(predict_fallback(batch,fallback))
    print('episodes',eps.index(ep)+1,'of',len(eps),'seconds',time.perf_counter()-start,flush=True)
pickle.dump({'schema':3,'input_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'predictions':predictions},open(f'eda/_pshort_{variant}_predictions.pkl','wb'))
loss={}
for name,pred in predictions.items():
    loss[name]={ep:sum(sum((v-(i==ys[q['query_id']]))**2 for i,v in enumerate(pred[q['query_id']])) for q in by[ep])/len(by[ep]) for ep in eps}
weights={}
for name in ('v12','v13'):
    moments={};fb=predictions['fallback'];pr=predictions[name]
    for ep in eps:
        a=b=0.
        for q in by[ep]:
            k=q['query_id'];d=[x-y for x,y in zip(pr[k],fb[k])];a+=sum(x*x for x in d);b+=sum(x*((j==ys[k])-fb[k][j]) for j,x in enumerate(d))
        moments[ep]=(a,b)
    weights[name]=[];held_loss={}
    for fold in range(3):
        train=[ep for i,ep in enumerate(eps) if i%3!=fold];w=max(0.,min(1.,sum(moments[e][1] for e in train)/sum(moments[e][0] for e in train)));weights[name].append(w)
        for i,ep in enumerate(eps):
            if i%3!=fold:continue
            values=[]
            for q in by[ep]:
                k=q['query_id'];prediction=[w*a+(1-w)*b for a,b in zip(pr[k],fb[k])];values.append(sum((v-(j==ys[k]))**2 for j,v in enumerate(prediction)))
            held_loss[ep]=sum(values)/len(values)
    loss[name+'_blend_cv']=held_loss
report={'benchmark':data['name'],'source_episodes':len(eps),'queries':len(qs),'weights':weights,'comparisons':{},'losses':loss}
for name,means in loss.items():
    d=[means[e]-loss['v12'][e] for e in eps];m=sum(d)/len(d);report['comparisons'][name]={'brier':sum(means.values())/len(eps),'delta_vs_v12':m,'cluster_se':math.sqrt(sum((v-m)**2 for v in d)/(len(d)-1)/len(d)),'improved':sum(v<0 for v in d)}
output='eda/ev109_independent_benchmark.json' if variant=='independent' else 'eda/ev108_matched_benchmark.json'
print({k:v for k,v in report.items() if k!='losses'},flush=True);Path(output).write_text(json.dumps(report,indent=2))
