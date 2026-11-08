"""Predeclared size-chain experiment; train outcomes first, validation separately.

Run without arguments for pseudo-short training evaluation; --validation for
the single validation check. SE clusters by ORIGINAL episode, not resamples.
"""
import copy
import json
import math
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from he.api import load_params, predict_queries
from he.model import load_split, official_score

p = load_params('runs/main_v6/params.json')
candidate = copy.deepcopy(p)
candidate['cfg']['size_chain_missing'] = True
report = {}
if '--validation' in sys.argv:
    qs, ex = load_split('kit/public', 'validation')
    for name, params in [('baseline', p), ('chain', candidate)]:
        pred = predict_queries(qs, params)
        report[name] = official_score(pred, ex, qs)
        print(name, report[name], flush=True)
    path = 'eda/ev54_validation.json'
else:
    qs, ys = pickle.load(open('eda/_pshort.pkl', 'rb'))
    losses = {}
    for name, params in [('baseline', p), ('chain', candidate)]:
        start = time.perf_counter()
        pred = predict_queries(qs, params)
        groups = defaultdict(list)
        for q in qs:
            key = q['query_id']
            groups[q['episode_id'].rsplit('-r', 1)[0]].append(
                sum((v - (k == ys[key])) ** 2 for k, v in enumerate(pred[key])))
        losses[name] = {e: sum(v) / len(v) for e, v in groups.items()}
        report[name] = {'brier': sum(losses[name].values()) / len(groups),
                        'seconds': time.perf_counter() - start}
        print(name, report[name], flush=True)
    diffs = [losses['chain'][e] - losses['baseline'][e] for e in losses['baseline']]
    mean = sum(diffs) / len(diffs)
    se = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1) / len(diffs))
    report['paired'] = {'difference': mean, 'cluster_se': se, 'episodes': len(diffs),
                        'improved_episodes': sum(d < 0 for d in diffs)}
    print(report['paired'], flush=True)
    path = 'eda/ev54_train.json'
Path(path).write_text(json.dumps(report, indent=2) + '\n')
