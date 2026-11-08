"""Research implementation equivalence, including nonzero machine offsets."""
import sys
sys.path.insert(0, '.')
from he.api import load_params
from he.model import load_split
from he.peer import peer_winners
from he.sparse_glm import predict as reference
from sparse_glm_numpy import predict
p = load_params('runs/main_v12/params.json')
qs, _ = load_split('kit/public', 'train')
short = [q for q in qs if len(q['seq']) < .6*q['cutoff']]
for index, q in enumerate(short[:6]):
    mates = [m for m in qs if m['episode_id'] == q['episode_id']]
    peers = {t:peer_winners(mates, q, t-3, True) for t in range(q['cutoff']+1)}
    hot = .2 if index%2 else .8
    offset = -.13 if index%2 else .17
    expected = reference(q, peers, p, p['global_state'], hot, offset)
    actual = predict(q, peers, p, p['global_state'], hot, offset)
    maximum = max(abs(a-b) for a, b in zip(expected, actual))
    assert maximum < 1e-11, maximum
    assert min(actual) >= 0 and abs(sum(actual)-1) < 1e-10
    print(q['query_id'], maximum, flush=True)
