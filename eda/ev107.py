"""Audit short-history masks and rebuild faithful training pseudo benchmarks.
No validation/test target probabilities are used. Input schemas/masks are inspected.
"""
import sys,pickle,json,random,math
from pathlib import Path
from collections import defaultdict,Counter
sys.path.insert(0,'.')
from he.model import load_split
train,extra=load_split('kit/public','train');qm={q['machine_id']:q for q in train};condition={r['query_id']:r['condition'] for r in extra['conditions']};by=defaultdict(list)
for q in train:by[q['episode_id']].append(q)
def describe(queries):
    episodes=defaultdict(list)
    for q in queries:episodes[q['episode_id']].append(q)
    cuts=[];unions=[];intersections=[];unique=[]
    for ms in episodes.values():
        old=[set(t for t in q['seq'] if t<q['cutoff']-40) for q in ms]
        cuts.append(ms[0]['cutoff']);unions.append(len(set.union(*old)));intersections.append(len(set.intersection(*old)));unique.append(len(set(map(frozenset,old))))
    return {'episodes':len(episodes),'queries':len(queries),'cutoff_range':[min(cuts),max(cuts)],'mean_cutoff':sum(cuts)/len(cuts),'mean_union_old_ticks':sum(unions)/len(unions),'mean_intersection_old_ticks':sum(intersections)/len(intersections),'unique_old_masks_histogram':dict(Counter(unique))}
report={}
for split in ('train','validation','test'):
    qs,_=load_split('kit/public',split);short=[q for q in qs if len(q['seq'])<.6*q['cutoff']];report[split]=describe(short)
old,old_y=pickle.load(open('eda/_pshort.pkl','rb'));report['old_pseudo']=describe(old)
rng=random.Random(20261108);independent=[]
for old_q in old:
    q=dict(old_q);source=qm[q['machine_id']];c=q['cutoff'];keep=set(range(c-40,c))|set(rng.sample(range(c-40),24));q['seq']={t:v for t,v in source['seq'].items() if t in keep};independent.append(q)
pickle.dump({'schema':3,'name':'independent_masks','queries':independent,'targets':old_y},open('eda/_pshort_independent.pkl','wb'));report['independent_masks']=describe(independent)
rng=random.Random(20261109);matched=[];targets={};PAY={0:0,1:1,2:2,5:3,20:4}
for ep,ms in sorted(by.items()):
    if condition[ms[0]['query_id']]!='stable':continue
    upper=min(q['cutoff'] for q in ms)-1
    if upper<192:continue
    for repeat in range(25):
        cutoff=rng.randint(192,min(255,upper))
        for source in ms:
            q=dict(source);q['episode_id']=f'{ep}-r{repeat}';q['query_id']=f'{ep}-r{repeat}-{source["machine_id"]}';q['cutoff']=cutoff
            keep=set(range(cutoff-40,cutoff))|set(rng.sample(range(cutoff-40),24));q['seq']={t:v for t,v in source['seq'].items() if t in keep}
            assert cutoff not in q['seq'] and len(q['seq'])==64
            targets[q['query_id']]=PAY[source['seq'][cutoff]];matched.append(q)
pickle.dump({'schema':3,'name':'matched_short','queries':matched,'targets':targets},open('eda/_pshort_matched.pkl','wb'));report['matched_short']=describe(matched)
report['matched_sources']=len({q['episode_id'].rsplit('-r',1)[0] for q in matched})
report['caveat']='Matched cutoffs are limited by available stable train history; upper tail is underrepresented. Repeated windows cluster by original episode. Frozen priors still use their documented public-data fits.'
print(json.dumps(report,indent=2));Path('eda/ev107_mask_audit.json').write_text(json.dumps(report,indent=2))
