"""Validate real source outcomes, strict time cutoffs and independent old masks."""
import sys
import pickle
from collections import defaultdict
sys.path.insert(0, '.')
from he.model import load_split
sources, extra = load_split('kit/public', 'train')
source = {q['machine_id']:q for q in sources}
conditions = {r['query_id']:r['condition'] for r in extra['conditions']}
payout = {0:0,1:1,2:2,5:3,20:4}
old, old_targets = pickle.load(open('eda/_pshort.pkl', 'rb'))
old_queries = {q['query_id']:q for q in old}
for variant in ('matched','independent'):
    data = pickle.load(open(f'eda/_pshort_{variant}.pkl', 'rb'))
    assert data['schema'] == 3
    episodes = defaultdict(list)
    seen = set()
    for q in data['queries']:
        original = source[q['machine_id']]
        cutoff = q['cutoff']
        assert q['query_id'] not in seen
        seen.add(q['query_id'])
        assert q['episode_id'].rsplit('-r',1)[0] == original['episode_id']
        assert q['signal'] == original['signal'] and q['machine_type'] == original['machine_type']
        assert cutoff < original['cutoff']
        assert len(q['seq']) == 64
        assert set(range(cutoff-40,cutoff)).issubset(q['seq'])
        assert all(t < cutoff and original['seq'][t] == value for t,value in q['seq'].items())
        assert data['targets'][q['query_id']] == payout[original['seq'][cutoff]]
        if variant == 'matched':
            assert 192 <= cutoff <= 255
            assert conditions[original['query_id']] == 'stable'
        else:
            assert cutoff == old_queries[q['query_id']]['cutoff']
            assert data['targets'][q['query_id']] == old_targets[q['query_id']]
        episodes[q['episode_id']].append(q)
    for machines in episodes.values():
        assert len(machines) == 8
        assert len({q['cutoff'] for q in machines}) == 1
        masks = {frozenset(t for t in q['seq'] if t < q['cutoff']-40) for q in machines}
        assert len(masks) == 8
    print(variant, len(seen), 'queries: source outcomes, time isolation and independent masks verified')
