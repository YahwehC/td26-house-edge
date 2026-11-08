"""Adaptation training pseudo-short benchmark with independent old masks.
Select one convex generic-fallback weight by original-episode folds.
Use fixed variance .04 selected by r2_offset_train; global logits from TRAIN only.
"""
import sys,json,random,math,time,copy
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,'.')
from he.api import load_params,predict_queries
from he.model import load_split
from he.fallback import predict_fallback
p=load_params('runs/r2_v1/params.json');p['global_state']['machine_variance']=.04;p['blend']['w_main']=1.
qs,e=load_split('kit2_x/public','train');cm={r['query_id']:r['condition'] for r in e['conditions']};by=defaultdict(list)
for q in qs:
    if cm[q['query_id']]=='stable':by[q['episode_id']].append(q)
rng=random.Random(20261108);PAY={0:0,1:1,2:2,5:3,20:4};moments={};losses={};start=time.perf_counter()
for ep,ms in sorted(by.items()):
    upper=min(q['cutoff'] for q in ms)-1
    if upper<192:continue
    pseudo=[];ys={}
    for repeat in range(20):
        c=rng.randint(192,min(255,upper))
        for source in ms:
            q=dict(source,episode_id=f'{ep}-r{repeat}',query_id=f'{source["query_id"]}-r{repeat}',cutoff=c);keep=set(range(c-40,c))|set(rng.sample(range(c-40),24));q['seq']={t:v for t,v in source['seq'].items() if t in keep};pseudo.append(q);ys[q['query_id']]=PAY[source['seq'][c]]
    a=predict_queries(pseudo,p);b=predict_fallback(pseudo,p['fallback']);aa=ab=0.;full=[]
    for q in pseudo:
        k=q['query_id'];d=[x-y for x,y in zip(a[k],b[k])];aa+=sum(x*x for x in d);ab+=sum(x*((j==ys[k])-b[k][j]) for j,x in enumerate(d));full.append((a[k],b[k],ys[k]))
    moments[ep]=(aa,ab);losses[ep]=full;print('episodes',len(moments),'seconds',time.perf_counter()-start,flush=True)
eps=sorted(moments);weights=[];comparisons=defaultdict(dict)
for fold in range(3):
    train=[ep for i,ep in enumerate(eps) if i%3!=fold];w=max(0.,min(1.,sum(moments[e][1] for e in train)/sum(moments[e][0] for e in train)));weights.append(w)
    for i,ep in enumerate(eps):
        if i%3!=fold:continue
        for name,weight in [('baseline',.9),('crossfit',w)]:
            values=[sum((weight*x+(1-weight)*z-(j==y))**2 for j,(x,z) in enumerate(zip(a,b))) for a,b,y in losses[ep]];comparisons[name][ep]=sum(values)/len(values)
d=[comparisons['crossfit'][ep]-comparisons['baseline'][ep] for ep in eps];m=sum(d)/len(d)
report={'fold_weights':weights,'full_weight':max(0.,min(1.,sum(v[1] for v in moments.values())/sum(v[0] for v in moments.values()))),'delta':m,'cluster_se':math.sqrt(sum((x-m)**2 for x in d)/(len(d)-1)/len(d)),'improved':sum(x<0 for x in d),'source_episodes':len(eps),'means':comparisons}
print({k:v for k,v in report.items() if k!='means'},flush=True);Path('eda/r2_blend_train.json').write_text(json.dumps(report,indent=2))
