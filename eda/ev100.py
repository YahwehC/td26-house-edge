"""Descriptive episode bootstrap for v12 -> v13, not an independent test guarantee."""
import sys,json,random
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,'.')
from he.model import load_split,read_csv
qs,e=load_split('kit/public','validation');qm={q['query_id']:q for q in qs};cm={r['query_id']:r['condition'] for r in e['conditions']};cols=('p_0','p_1','p_2','p_5','p_20')
target={r['query_id']:[float(r[c]) for c in cols] for r in e['targets']};reference={r['query_id']:[float(r[c]) for c in cols] for r in e['reference']};loss={}
for name,path in [('v12','runs/main_v12/validation_predictions.csv'),('v13','eda/ev99_predictions.csv')]:
    grouped=defaultdict(list)
    for row in read_csv(path):
        k=row['query_id'];grouped[cm[k],qm[k]['episode_id']].append(sum((float(row[c])-v)**2 for c,v in zip(cols,target[k])))
    loss[name]={k:sum(v)/len(v) for k,v in grouped.items()}
grouped=defaultdict(list)
for k,y in target.items():grouped[cm[k],qm[k]['episode_id']].append(sum((a-b)**2 for a,b in zip(reference[k],y)))
ref={k:sum(v)/len(v) for k,v in grouped.items()};groups={c:[k for k in ref if k[0]==c] for c,_ in ref}
def score(name,sample):return 60/(1+sum(sum(loss[name][k] for k in keys)/sum(ref[k] for k in keys) for keys in sample.values())/len(sample))
rng=random.Random(20261108);gains=[]
for _ in range(5000):
    sample={c:rng.choices(keys,k=len(keys)) for c,keys in groups.items()};gains.append(score('v13',sample)-score('v12',sample))
gains.sort();report={'gain':score('v13',groups)-score('v12',groups),'bootstrap_95':[gains[125],gains[4874]],'short_episodes_improved':sum(loss['v13'][k]<loss['v12'][k] for k in groups['short_history']),'short_episodes':len(groups['short_history']),'caveat':'Repeated public-set development; descriptive bootstrap is not an independent generalization guarantee.'}
print(report);Path('eda/ev100_gain.json').write_text(json.dumps(report,indent=2))
