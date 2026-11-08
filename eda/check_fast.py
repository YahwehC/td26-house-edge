"""Regression and timing: optimized vs original exhaustive change-point work."""
import json
import sys
import time
import subprocess
import types
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import he.joint as joint
from he.api import load_params, predict_queries
from he.model import load_split, official_score

params = load_params('runs/main_v6/params.json')
qs, ex = load_split('kit/public', 'validation')
old_module = types.ModuleType('original_model')
exec(subprocess.check_output(['git', 'show', 'd783b27:he/model.py'], text=True), old_module.__dict__)
original = old_module.WinMachine
def exhaustive(q, cfg, lp, **kwargs):
    return original(q, cfg, lp)
report = {}
start = time.perf_counter()
with patch.object(joint, 'WinMachine', exhaustive):
    before = predict_queries(qs, params)
report['original_seconds'] = time.perf_counter() - start
start = time.perf_counter()
after = predict_queries(qs, params)
report['optimized_seconds'] = time.perf_counter() - start
report['max_abs_difference'] = max(abs(a-b) for k in before for a,b in zip(before[k], after[k]))
assert before == after
report['score'] = official_score(after, ex, qs)
print(json.dumps(report, indent=2), flush=True)
Path('eda/check_fast_result.json').write_text(json.dumps(report, indent=2) + '\n')
