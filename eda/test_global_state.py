"""Check production state inference normalization and future-tick exclusion."""
import sys,tempfile,csv,copy
from pathlib import Path
sys.path.insert(0,'.')
from he.model import load_split
from he.api import load_params,predict_queries
q,e=load_split('kit/public','validation');ep=q[0]['episode_id'];sample=[r for r in q if r['episode_id']==ep]
p=load_params('runs/main_v9/params.json');base=predict_queries(sample,p)
assert all(min(row)>=0 and abs(sum(row)-1)<1e-10 for row in base.values())
with tempfile.TemporaryDirectory() as temporary:
    folder=Path(temporary)/'test';folder.mkdir()
    with (folder/'queries.csv').open('w',newline='') as f:
        cols=['query_id','episode_id','machine_id','machine_type','signal','cutoff'];w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({k:r[k] for k in cols} for r in sample)
    with (folder/'history.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['episode_id','machine_id','tick','payout'])
        for r in sample:
            for t,v in r['seq'].items():w.writerow([ep,r['machine_id'],t,v])
            for t in range(r['cutoff'],r['cutoff']+20):w.writerow([ep,r['machine_id'],t,20])
    loaded,_=load_split(temporary,'test');after=predict_queries(loaded,p)
    assert base==after
print('normalized; adding future payouts leaves predictions bit-identical')
