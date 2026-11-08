"""Train diagnostic: do the initial eight payouts obey the later autoregression?"""
import sys,pickle,json,math
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,'.')
from he.model import load_split
qs,e=load_split('kit/public','train');cm={r['query_id']:r['condition'] for r in e['conditions']};rows=pickle.load(open('eda/_state_rows.pkl','rb'));by=defaultdict(list)
for r in rows:by[r['machine']].append(r)
pairs=((1,3),(2,5),(4,7));stats=defaultdict(lambda:[0,0]);count=0
for q in qs:
    if cm[q['query_id']]!='stable':continue
    matches=[];seq=q['seq']
    for a,b in pairs:
        for orientation in (0,1):
            known=[r for r in by[q['machine_id']] if r['tick']-a in seq and r['tick']-b in seq]
            if known and all((int((seq[r['tick']-a]>0)==(seq[r['tick']-b]>0))^orientation)==r['high'] for r in known):matches.append((a,b,orientation))
    if len(matches)!=1:continue
    a,b,orientation=matches[0];count+=1
    for t in range(b,16):
        if t not in seq:continue
        high=int((seq[t-a]>0)==(seq[t-b]>0))^orientation
        key=(q['machine_type'],'initial' if t<8 else 'later',high)
        stats[key][0]+=seq[t]>0;stats[key][1]+=1
report={'machines':count,'groups':{str(k):{'wins':v[0],'n':v[1],'rate':v[0]/v[1]} for k,v in sorted(stats.items())}}
print(json.dumps(report,indent=2));Path('eda/ev105_initial_history.json').write_text(json.dumps(report,indent=2))
