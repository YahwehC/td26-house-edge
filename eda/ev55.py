"""Three-fold episode-disjoint cell-prior comparison on training pseudo-short outcomes."""
import sys,copy,pickle,math,json,time
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,'.')
from he.api import load_params,predict_queries
from he.model import load_split,DEFAULT_CFG
from he.fit import fit_cell_prior
from eb_prior import fit
qs, ex=load_split('kit/public','train')
cm={r['query_id']:r['condition'] for r in ex['conditions']}
pseudo, ys=pickle.load(open('eda/_pshort.pkl','rb'))
episodes=sorted({q['episode_id'] for q in qs})
folds={e:i%3 for i,e in enumerate(episodes)}
p=load_params('runs/main_v6/params.json')
# Only cell priors differ. Shared structure/size vectors are frozen throughout.
losses={name:defaultdict(list) for name in ('moments','marginal')}
for fold in range(3):
    training=[q for q in qs if folds[q['episode_id']] != fold]
    held=[q for q in pseudo if folds[q['episode_id'].rsplit('-r',1)[0]] == fold]
    baseline=copy.deepcopy(p)
    cfg=dict(DEFAULT_CFG,**p['cfg'])
    baseline['cfg']['cell_prior']=fit_cell_prior([(training,cm)],cfg)
    candidate=copy.deepcopy(baseline)
    candidate['cfg']['cell_prior']=fit([(training,cm)],dict(cfg,cell_prior=baseline['cfg']['cell_prior']))
    for name,params in [('moments',baseline),('marginal',candidate)]:
        pr=predict_queries(held,params)
        for q in held:
            k=q['query_id']; e=q['episode_id'].rsplit('-r',1)[0]
            losses[name][e].append(sum((v-(j==ys[k]))**2 for j,v in enumerate(pr[k])))
        print('fold',fold,name,'done',flush=True)
means={name:{e:sum(v)/len(v) for e,v in data.items()} for name,data in losses.items()}
diffs=[means['marginal'][e]-means['moments'][e] for e in means['moments']]
m=sum(diffs)/len(diffs); se=math.sqrt(sum((d-m)**2 for d in diffs)/(len(diffs)-1)/len(diffs))
report={'mean_brier':{name:sum(data.values())/len(data) for name,data in means.items()},'delta':m,'cluster_se':se,'improved':sum(d<0 for d in diffs),'n_episodes':len(diffs),'per_episode':means}
print(report,flush=True)
Path('eda/ev55_train.json').write_text(json.dumps(report,indent=2))
