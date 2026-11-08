"""Full pseudo-history forecast check of one train-fitted size-noise rate.
Each source episode uses the rate fitted without its original episode fold.
"""
import sys,json,pickle,math,time,copy
from pathlib import Path
sys.path.insert(0,'.')
from he.api import load_params,predict_queries
from he.model import load_split
import he.sparse_glm
from sparse_glm_numpy import predict as accelerated
he.sparse_glm.predict=accelerated
qs,ys=pickle.load(open('eda/_pshort.pkl','rb'));base=json.load(open('eda/ev96_exact_sparse.json'))['means'];fit=json.load(open('eda/ev103_regime_noise.json'))['size_only'];p=load_params('runs/main_v13/params.json');training,_=load_split('kit/public','train');folds={ep:i%3 for i,ep in enumerate(sorted({q['episode_id'] for q in training}))}
originals=sorted(base);means={};start=time.perf_counter()
for ep in originals:
    candidate=copy.deepcopy(p);candidate['cfg']['size_eps']=fit['fold_rates'][folds[ep]]
    batch=[q for q in qs if q['episode_id'].rsplit('-r',1)[0]==ep];pred=predict_queries(batch,candidate)
    means[ep]=sum(sum((v-(i==ys[q['query_id']]))**2 for i,v in enumerate(pred[q['query_id']])) for q in batch)/len(batch)
    print('episodes',len(means),'of',len(originals),'seconds',time.perf_counter()-start,flush=True)
d=[means[e]-base[e] for e in originals];m=sum(d)/len(d)
report={'delta':m,'cluster_se':math.sqrt(sum((v-m)**2 for v in d)/(len(d)-1)/len(d)),'improved':sum(v<0 for v in d),'episodes':len(d),'fold_rates':fit['fold_rates'],'full_rate':fit['full_rate'],'means':means,'baseline':base}
print({k:v for k,v in report.items() if k not in ('means','baseline')},flush=True);Path('eda/ev104_size_noise.json').write_text(json.dumps(report,indent=2))
