"""Stable top-level API: fit all parameters from a public/ folder, predict any split. Pure stdlib."""
import json
import math
from pathlib import Path

from .model import DEFAULT_CFG, load_split, fit_size_vectors, official_score
from .joint import predict_joint as predict
from .fit import fit_cell_prior, fit_structure_priors, fit_lag_rule

PARAMS_VERSION = 1
# Settings chosen on main-round validation (see CLAUDE.md validation log).
SETTINGS = {
    "model_prior": {"stable": 0.4, "drift": 0.3, "shared": 0.3},
    "model_prior_short": {"stable": 1.0},
    "drift_rho": 0.9,
    "shared_rho": 0.9,
    "kappa_scale": 1.0,
}


def _cond(extra):
    return {r["query_id"]: r["condition"] for r in extra.get("conditions", [])}


def fit_params(public, fit_splits=("train", "validation"), size_split="validation"):
    """Fit everything the predictor needs. Deterministic.

    fit_splits: labelled splits whose 'stable' machines are used for structure/cell priors.
    size_split: split with targets.csv used to learn the two size vectors per type."""
    public = Path(public)
    qs, ex = load_split(public, size_split)
    if "targets" not in ex:
        raise SystemExit(f"{size_split} has no targets.csv; cannot learn size vectors")
    vecs = fit_size_vectors(qs, ex["targets"])
    labelled = []
    for sp in fit_splits:
        q, e = load_split(public, sp)
        labelled.append((q, _cond(e)))
    struct = fit_structure_priors(labelled, vecs)
    cfg = dict(SETTINGS)
    cfg.update({k: v for k, v in struct.items() if not k.startswith("_")})
    cfg["cell_prior"] = fit_cell_prior(labelled, dict(DEFAULT_CFG, **cfg))
    cfg["size_lag_given_pair"], lag_rule = fit_lag_rule(labelled, vecs, dict(DEFAULT_CFG, **cfg))
    return {"params_version": PARAMS_VERSION, "fit_splits": list(fit_splits), "size_vectors": vecs,
            "cfg": cfg, "votes": struct["_votes"], "lag_rule": lag_rule}


def predict_split(public, split, params):
    qs, ex = load_split(Path(public), split)
    pred = predict(qs, params["size_vectors"], params["cfg"])
    return qs, ex, pred


def normalize_row(p):
    p = [min(max(float(x), 0.0), 1.0) for x in p]
    s = math.fsum(p)
    return [x / s for x in p]


def write_submission(path, pred):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("query_id,p_0,p_1,p_2,p_5,p_20\n")
        for qid in sorted(pred):
            f.write(qid + "," + ",".join(repr(x) for x in normalize_row(pred[qid])) + "\n")


def save_params(path, params):
    Path(path).write_text(json.dumps(params, indent=1, sort_keys=True), encoding="utf-8")


def load_params(path):
    p = json.loads(Path(path).read_text(encoding="utf-8"))
    if p.get("params_version") != PARAMS_VERSION:
        raise SystemExit("unsupported params file")
    return p


def score_validation(public, params, max_points=60.0):
    qs, ex, pred = predict_split(public, "validation", params)
    return official_score(pred, ex, qs, max_points)
