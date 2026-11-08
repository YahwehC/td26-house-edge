"""Frozen sparse-win candidate, training-outcome comparison only."""
import sys,pickle,json,math,time
from pathlib import Path
from collections import defaultdict
from unittest.mock import patch
sys.path.insert(0,'.')
import he.joint as joint
from he.api import load_params,predict_queries
from hx.quadrature_sparse import QuadratureMachine as SparseMachine
qs,ys=pickle.load(open('eda/_pshort.pkl','rb'))
if '--full' not in sys.argv:
    qs=[q for q in qs if int(q['episode_id'].rsplit('-r',1)[1])<5]
p=load_params('runs/main_v6/params.json')
means={}
for name,cls in [('base',joint.WinMachine),('sparse',SparseMachine)]:
    start=time.perf_counter()
    with patch.object(joint,'WinMachine',cls): pred=predict_queries(qs,p)
    loss=defaultdict(list)
    for q in qs:
        k=q['query_id'];e=q['episode_id'].rsplit('-r',1)[0]
        loss[e].append(sum((v-(j==ys[k]))**2 for j,v in enumerate(pred[k])))
    means[name]={e:sum(v)/len(v) for e,v in loss.items()}
    print(name,sum(means[name].values())/len(loss),time.perf_counter()-start,flush=True)
d=[means['sparse'][e]-means['base'][e] for e in means['base']]; m=sum(d)/len(d)
report={'delta':m,'cluster_se':math.sqrt(sum((x-m)**2 for x in d)/(len(d)-1)/len(d)), 'improved':sum(x<0 for x in d),'episodes':len(d),'queries':len(qs),'means':means}
print(report,flush=True)
Path('eda/ev59_train'+('_full' if '--full' in sys.argv else '')+'.json').write_text(json.dumps(report,indent=2))

