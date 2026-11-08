"""Freeze the corrected exact sparse model and build its deterministic test CSV."""
import sys
from pathlib import Path
sys.path.insert(0, '.')
from he.api import load_params, save_params, predict_split, write_submission
from check_csv import check
p = load_params('runs/main_v12/params.json')
p['cfg'].update(exact_sparse_glm=True, marginalize_missing_peers=True)
p['global_state_fit']['sparse_recursion'] = 'Episode-isolated train diagnostic ev96; one OOF validation check ev99. Missing peer wins use empirical marginal rates; machine offset included.'
out = Path('runs/main_v13')
out.mkdir(exist_ok=True)
save_params(out/'params.json', p)
(out/'validation_score.json').write_bytes(Path('eda/ev99_validation.json').read_bytes())
(out/'validation_predictions.csv').write_bytes(Path('eda/ev99_predictions.csv').read_bytes())
q, e, prediction = predict_split('kit/public', 'test', p)
write_submission(out/'submission.csv', prediction)
errors = check(out/'submission.csv', 'kit/public/test/queries.csv')
assert not errors, errors
print('main_v13 test checked', len(prediction), flush=True)
