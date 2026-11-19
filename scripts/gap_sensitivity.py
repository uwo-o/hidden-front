"""Sensitivity of the unsupported fraction of the piecewise reference fronts to the gap threshold.

Usage:
    python scripts/gap_sensitivity.py
Reads results/pw_{sigma,eta}_runs.json and pw_*_long.json; writes results/pw_gap_sensitivity.json.
"""
from __future__ import annotations

import json

import numpy as np

import analyze_experiment as A
from hidden_front.moead import nondominated_mask
from hidden_front.support import support_gap

THRESHOLDS = [1e-3, 3e-3, 5e-3, 1e-2]


def main():
    out = {}
    for form in ["sigma", "eta"]:
        runs = json.loads((A.RES / f"pw_{form}_runs.json").read_text())["runs"]
        lp = A.RES / f"pw_{form}_long.json"
        extra = json.loads(lp.read_text())["runs"] if lp.exists() else []
        F_all = np.vstack([np.array(r["F"]) for r in runs + extra])
        F = F_all[nondominated_mask(F_all)]
        lo, hi = F.min(0), F.max(0)
        Fr = (F - lo) / (hi - lo)
        Fr = Fr[A.thin(Fr)]
        g = support_gap(Fr, Fr)
        uns = g[g > 1e-3]
        out[form] = {"n": int(len(Fr)),
                     "frac_above": {f"{t:g}": float((g > t).mean()) for t in THRESHOLDS},
                     "gap_unsup_median": float(np.median(uns)) if len(uns) else 0.0,
                     "gap_unsup_p90": float(np.percentile(uns, 90)) if len(uns) else 0.0}
        print(form, out[form], flush=True)
    (A.RES / "pw_gap_sensitivity.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
