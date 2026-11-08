"""Episode-disjoint 3-fold test of history-fitted size hyperparameters."""
import sys,pickle,copy,json,math
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,'.')
from he.api import load_params,predict_queries
from he.model import load_split
from size_hyper import fit
qs,ex=load_split('kit/public','train');cm={r['query_id']:r['condition'] for r in ex['conditions']}
pseudo,ys=pickle.load(open('eda/_pshort.pkl','rb'))
folds={e:i%3 for i,e in enumerate(sorted({q['episode_id'] for q in qs}))}
p=load_params('runs/main_v6/params.json');loss={n:defaultdict(list) for n in ('base','fitted')}
for fold in range(3):
    candidate=copy.deepcopy(p)
    candidate['cfg']=fit([([q for q in qs if folds[q['episode_id']]!=fold],cm)],p)
    held=[q for q in pseudo if folds[q['episode_id'].rsplit('-r',1)[0]]==fold]
    for name,params in [('base',p),('fitted',candidate)]:
        pred=predict_queries(held,params)
        for q in held:
            k=q['query_id'];e=q['episode_id'].rsplit('-r',1)[0]
            loss[name][e].append(sum((v-(j==ys[k]))**2 for j,v in enumerate(pred[k])))
        print(fold,name,'done',flush=True)
means={n:{e:sum(v)/len(v) for e,v in data.items()} for n,data in loss.items()}
d=[means['fitted'][e]-means['base'][e] for e in means['base']];m=sum(d)/len(d)
report={'brier':{n:sum(data.values())/len(data) for n,data in means.items()},'delta':m,'cluster_se':math.sqrt(sum((x-m)**2 for x in d)/(len(d)-1)/len(d)), 'improved':sum(x<0 for x in d),'episodes':len(d),'means':means}
print(report,flush=True);Path('eda/ev57_train.json').write_text(json.dumps(report,indent=2))
