"""Paired episode bootstrap for the reported v8 -> v10 validation gain."""
import sys,json,random
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,'.')
from he.model import load_split,read_csv
q,e=load_split('kit/public','validation');cm={r['query_id']:r['condition'] for r in e['conditions']};qm={r['query_id']:r for r in q}
columns=['p_0','p_1','p_2','p_5','p_20'];target={r['query_id']:[float(r[c]) for c in columns] for r in e['targets']};ref={r['query_id']:[float(r[c]) for c in columns] for r in e['reference']}
loss={}
for name,path in [('v8','runs/main_v8/validation_predictions.csv'),('v9','runs/main_v9/validation_predictions.csv'),('v10','eda/ev68_predictions.csv')]:
    groups=defaultdict(list)
    for row in read_csv(path):
        k=row['query_id'];ep=qm[k]['episode_id'];condition=cm[k];truth=target[k];pred=[float(row[c]) for c in columns]
        groups[condition,ep].append(sum((a-b)**2 for a,b in zip(pred,truth)))
    loss[name]={key:sum(v)/len(v) for key,v in groups.items()}
reference=defaultdict(list)
for k,truth in target.items():reference[cm[k],qm[k]['episode_id']].append(sum((a-b)**2 for a,b in zip(ref[k],truth)))
reference={key:sum(v)/len(v) for key,v in reference.items()};groups={condition:[key for key in reference if key[0]==condition] for condition,_ in reference}
def score(name,sample):
    ratios=[sum(loss[name][key] for key in keys)/sum(reference[key] for key in keys) for keys in sample.values()]
    return 60/(1+sum(ratios)/len(ratios))
rng=random.Random(20261107);gains=[]
for _ in range(5000):
    sample={condition:rng.choices(keys,k=len(keys)) for condition,keys in groups.items()}
    gains.append(score('v10',sample)-score('v8',sample))
gains.sort();report={'gain_v10_vs_v8':score('v10',groups)-score('v8',groups),'bootstrap_95_interval':[gains[125],gains[4874]],'caveat':'Repeated public-set development; interval is descriptive, not an independent test guarantee.','fold_gains_vs_v9':[]}
eps=sorted({r['episode_id'] for r in q})
for fold in range(3):
    sample={condition:[key for key in keys if eps.index(key[1])%3==fold] for condition,keys in groups.items()}
    report['fold_gains_vs_v9'].append(score('v10',sample)-score('v9',sample))
print(report);Path('eda/ev69_bootstrap.json').write_text(json.dumps(report,indent=2))
