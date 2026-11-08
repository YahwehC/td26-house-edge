"""Round 2: score main_v11-v13 inference flags on top of runs/r2_v1 params (validation, max 20)."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from he.api import load_params, predict_queries  # noqa: E402
from he.model import load_split, official_score  # noqa: E402

data = Path("kit2_x/public")
base = load_params("runs/r2_v1/params.json")
v0 = load_params("runs/r2_v0/params.json")
val_q, val_ex = load_split(data, "validation")


def variant(offset=False, sparse=False, corr=False):
    p = json.loads(json.dumps(base))
    if offset:
        p["global_state"]["machine_variance"] = 0.04
    if sparse:
        p["cfg"]["exact_sparse_glm"] = True
        p["cfg"]["marginalize_missing_peers"] = True
    if corr and v0.get("logit_corr"):
        p["logit_corr"] = v0["logit_corr"]
    return p


res = {}
for name, kw in [("v1", {}), ("offset", dict(offset=True)), ("offset+sparse", dict(offset=True, sparse=True)),
                 ("sparse", dict(sparse=True))]:
    t = time.perf_counter()
    s = official_score(predict_queries(val_q, variant(**kw)), val_ex, val_q, 20)
    res[name] = s
    print(f"{name:15s} {s['points']:.4f} {s['ratios']} {time.perf_counter() - t:.0f}s", flush=True)
Path("eda/r2_variants.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
