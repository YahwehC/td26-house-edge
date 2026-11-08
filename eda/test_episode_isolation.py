"""A forecast cannot depend on another episode with the same local machine IDs."""
import sys,copy,csv,json
sys.path.insert(0,'.')
from he.api import load_params,predict_queries
from he.model import load_split
p=load_params('runs/main_v12/params.json');qs,_=load_split('kit/public','validation');ep=qs[0]['episode_id'];first=[q for q in qs if q['episode_id']==ep]
second=[]
for q in first:
    clone=dict(q,episode_id='independent-copy',query_id=q['query_id']+'-copy')
    clone['seq']={t:1 if v==0 else 0 for t,v in q['seq'].items()}
    second.append(clone)
alone=predict_queries(first,p);together=predict_queries(first+second,p)
assert all(alone[k]==together[k] for k in alone)
print('changing a separate episode with repeated machine IDs leaves predictions bit-identical',flush=True)
columns=('p_0','p_1','p_2','p_5','p_20')
expected={r['query_id']:[float(r[k]) for k in columns] for r in csv.DictReader(open('runs/main_v12/validation_predictions.csv'))}
configs=json.load(open('eda/ev68_crossfit.json'))['configs'];eps=sorted({q['episode_id'] for q in qs});maximum=0.
for fold,config in enumerate(configs):
    candidate=copy.deepcopy(p);candidate['global_state']=dict(config,machine_variance=.04)
    held=[q for q in qs if eps.index(q['episode_id'])%3==fold];actual=predict_queries(held,candidate)
    maximum=max(maximum,max(abs(a-b) for k,row in actual.items() for a,b in zip(row,expected[k])))
    assert maximum<1e-12,maximum
    print('validation fold',fold,'unchanged; max difference',maximum,flush=True)
