"""Descriptive paired episode bootstrap of r2_v1 -> r2_v2 OOF predictions."""
import sys,json,random
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,'.')
from he.model import load_split,read_csv
qs,e=load_split('kit2_x/public','validation');qm={q['query_id']:q for q in qs};cm={r['query_id']:r['condition'] for r in e['conditions']};cols=('p_0','p_1','p_2','p_5','p_20')
target={r['query_id']:[float(r[c]) for c in cols] for r in e['targets']};refrows={r['query_id']:[float(r[c]) for c in cols] for r in e['reference']};loss={}
for name,path in [('v1','runs/r2_v1/validation_predictions.csv'),('v2','eda/r2_pooled_predictions.csv')]:
    groups=defaultdict(list)
    for row in read_csv(path):
        k=row['query_id'];groups[cm[k],qm[k]['episode_id']].append(sum((float(row[c])-v)**2 for c,v in zip(cols,target[k])))
    loss[name]={k:sum(v)/len(v) for k,v in groups.items()}
groups=defaultdict(list)
for k,y in target.items():groups[cm[k],qm[k]['episode_id']].append(sum((a-b)**2 for a,b in zip(refrows[k],y)))
reference={k:sum(v)/len(v) for k,v in groups.items()};by={c:[k for k in reference if k[0]==c] for c,_ in reference}
def score(name,sample):return 20/(1+sum(sum(loss[name][k] for k in keys)/sum(reference[k] for k in keys) for keys in sample.values())/len(sample))
rng=random.Random(20261108);gains=[]
for _ in range(5000):
    sample={c:rng.choices(keys,k=len(keys)) for c,keys in by.items()};gains.append(score('v2',sample)-score('v1',sample))
gains.sort();eps=sorted({q['episode_id'] for q in qs});fold_gains=[]
for fold in range(3):
    sample={c:[k for k in keys if eps.index(k[1])%3==fold] for c,keys in by.items()};fold_gains.append(score('v2',sample)-score('v1',sample))
report={'gain':score('v2',by)-score('v1',by),'bootstrap_95':[gains[125],gains[4874]],'fold_gains':fold_gains,'improved_episodes':sum(loss['v2'][k]<loss['v1'][k] for k in reference),'episodes':len(reference),'caveat':'Public development and fixed pre-existing size vectors; descriptive bootstrap is not an independent hidden-test guarantee.'}
print(report,flush=True);Path('eda/r2_v2_gain.json').write_text(json.dumps(report,indent=2))
