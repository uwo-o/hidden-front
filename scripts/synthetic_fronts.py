"""Synthetic check of Theorems 1-2 of the paper on product-structured fronts.

Objectives: maximize a (an extensive output), maximize r (a ratio, e.g. efficiency) and
minimize c = a phi(r). Extensive reformulation: minimize b = a (1/r - 1) instead of
maximizing r. Domain: r in [r0, r1], 0 < a <= 4 r (1 - r).

For the first three profiles chi(u) = phi(1 / (1 + u)) is convex and the extensive
formulation should be fully supported; the last profile has a concave chi and should not.

Usage:
    python scripts/synthetic_fronts.py
Writes results/synthetic_fronts.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hidden_front.support import support_gap

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "TUR-like phi=2r/(1-r)": (lambda r: 2 * r / (1 - r), 0.05, 0.95),
    "linear phi=r": (lambda r: r, 0.05, 0.95),
    "exponential phi=exp(2r)": (lambda r: np.exp(2 * r), 0.05, 0.95),
    "concave chi phi=6-1/(2r^2)": (lambda r: 6 - 1 / (2 * r ** 2), 0.30, 0.95),
}


def main(n=40):
    out = {}
    for name, (phi, r0, r1) in CASES.items():
        pts = []
        for r in np.linspace(r0, r1, n):
            amax = 4 * r * (1 - r)
            for a in np.linspace(amax / n, amax, n):
                pts.append((a, r, a * phi(r), a * (1 / r - 1)))
        a, r, c, b = np.array(pts).T
        res = {}
        for label, F in [("ratio (a, r, c)", np.column_stack([-a, -r, c])),
                         ("extensive (a, b, c)", np.column_stack([-a, b, c]))]:
            Fn = (F - F.min(0)) / (F.max(0) - F.min(0))
            g = support_gap(Fn, Fn)
            res[label] = {"unsupported": float((g > 1e-3).mean()), "max_gap": float(g.max())}
        out[name] = res
        print(name, res, flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "synthetic_fronts.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
