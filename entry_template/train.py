"""Re-fit params.json from a public/ data folder (train + validation). Pure stdlib, deterministic.

    python train.py --data /path/to/public --out params.json [--w-main W]

W = weight of the main model in the blend with the generic fallback model (he/fallback.py);
default: the weight stored in the bundled params.json, or 1 when no blend is bundled.
"""
import sys
sys.dont_write_bytecode = True

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from he.api import fit_params, load_params, save_params
from he.fallback import fit_fallback
from he.model import load_split


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fit House Edge parameters")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w-main", type=float, default=None)
    a = ap.parse_args(argv)
    w = a.w_main
    if w is None:
        bundled = HERE / "params.json"
        w = float((load_params(bundled).get("blend") or {}).get("w_main", 1.0)) if bundled.exists() else 1.0
    data = Path(a.data)
    if w > 0:
        params = fit_params(data, fit_splits=("train", "validation"))
        if (HERE / "params.json").exists():
            bundled_cfg = load_params(HERE / "params.json").get("cfg", {})
            if bundled_cfg.get("sparse_win"):
                params["cfg"]["sparse_win"] = True
            if bundled_cfg.get("sparse_joint"):
                params["cfg"]["sparse_joint"] = True
            if bundled_cfg.get("exact_sparse_glm"):
                params["cfg"]["exact_sparse_glm"] = True
            if bundled_cfg.get("marginalize_missing_peers"):
                params["cfg"]["marginalize_missing_peers"] = True
            # Preserve the offline-fitted global-state checkpoint. This helper
            # refits the original priors, not the optional EM checkpoint.
            checkpoint = load_params(HERE / "params.json").get("global_state")
            if checkpoint:
                params["global_state"] = checkpoint
                metadata = load_params(HERE / "params.json").get("global_state_fit")
                if metadata:
                    params["global_state_fit"] = metadata
    else:
        params = {"params_version": 1}
    if w < 1:
        params["blend"] = {"w_main": w}
        params["fallback"] = fit_fallback([load_split(data, "train")[0], load_split(data, "validation")[0]])
    bundled_corr = load_params(HERE / "params.json").get("logit_corr") if (HERE / "params.json").exists() else True
    if bundled_corr:
        # cross-machine coupling: logistic regression on train+validation per-tick outcomes (he/coupling.py)
        from he.coupling import fit_coupling, pairs_from_cfg
        tq, tex = load_split(data, "train")
        vq, vex = load_split(data, "validation")
        cm = lambda ex: {r["query_id"]: r["condition"] for r in ex.get("conditions", [])}
        c = fit_coupling([tq, vq], [cm(tex), cm(vex)], log=lambda *x: None, pairs=pairs_from_cfg(params.get("cfg")))
        if c:
            params["logit_corr"] = c
    save_params(Path(a.out), params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
