"""Robustness of the support diagnostics on synthetic product-structured fronts.

Two studies on the TUR-like front D = {0 < a <= 4 r (1 - r)}, phi(r) = 2 r / (1 - r), extensive cost
b = a (1/r - 1):

1. Resolution and tolerance (exact structure). The unsupported fraction of a fixed set of 400 test
   points is computed against fronts sampled with an increasing number of points, for several
   tolerances. Theorem 1 predicts that every interior point is unsupported, so the fraction should
   approach one as the sample is refined and the tolerance decreases; the residual is a band near the
   boundary of D whose width shrinks with resolution.

2. Breaking the structure. The cost is c = a^(1 + kappa) phi(r): kappa = 0 is the exact product
   structure; kappa < 0 (economies of scale) and kappa > 0 (diseconomies) break the homogeneity in a.
   For each kappa, the unsupported fraction and the relative weighted-sum ceiling gap
   (HV_ref - HV_supported) / HV_ref are reported in both formulations.

Usage:
    python scripts/structure_robustness.py
Writes results/structure_robustness.json and figures/fig17_structure_robustness.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from pymoo.indicators.hv import HV

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hidden_front.moead import nondominated_mask  # noqa: E402
from hidden_front.support import support_gap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOLS = [1e-2, 1e-3, 1e-4, 1e-5]


def phi(r):
    return 2 * r / (1 - r)


def sample_domain(n, rng=None):
    """Points (a, r) of D on a regular grid (rng=None) or uniformly at random."""
    if rng is None:
        k = int(np.sqrt(n))
        r, t = np.meshgrid(np.linspace(0.05, 0.95, k), np.linspace(0.02, 1.0, k))
        r, t = r.ravel(), t.ravel()
    else:
        r, t = rng.uniform(0.05, 0.95, n), rng.uniform(0.02, 1.0, n)
    a = t * 4 * r * (1 - r)
    return a, r, t


def fronts(a, r, kappa=0.0):
    c = a ** (1 + kappa) * phi(r)
    b = a * (1 / r - 1)
    return np.column_stack([-a, -r, c]), np.column_stack([-a, b, c])


def normalize(F, ref=None):
    ref = F if ref is None else ref
    lo, hi = ref.min(0), ref.max(0)
    return (F - lo) / (hi - lo)


def resolution_study(seed=1):
    rng = np.random.default_rng(seed)
    a_t, r_t, t_t = sample_domain(400, rng)
    keep = (t_t > 0.05) & (t_t < 0.98)                  # interior test points
    a_t, r_t, t_t = a_t[keep], r_t[keep], t_t[keep]
    out = {}
    for n in (400, 1600, 6400):
        a, r, _ = sample_domain(n)
        Fr, _ = fronts(a, r)
        Tr, _ = fronts(a_t, r_t)
        ref = np.vstack([Fr, Tr])
        g = support_gap(normalize(Tr, ref), normalize(ref, ref))
        out[n] = {f"{tol:g}": float(np.mean(g > tol)) for tol in TOLS}
        out[n]["median_gap"] = float(np.median(g))
        print("resolution", n, out[n], flush=True)
    return out


def kappa_study(kappas=(-0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0), n=1600):
    a, r, _ = sample_domain(n)
    hv = HV(ref_point=np.full(3, 1.1))
    out = {}
    for k in kappas:
        row = {}
        for name, F in zip(("ratio", "extensive"), fronts(a, r, k)):
            F = F[nondominated_mask(F)]
            G = normalize(F)
            g = support_gap(G, G)
            h_ref, h_sup = hv(G), hv(G[g <= 1e-3])
            row[name] = dict(unsupported=float(np.mean(g > 1e-3)), max_gap=float(g.max()),
                             ceiling_gap=float((h_ref - h_sup) / h_ref), n_front=int(len(G)))
        out[f"{k:g}"] = row
        print("kappa", k, row, flush=True)
    return out


def figure(res, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    ax = axes[0]
    ns = sorted(int(n) for n in res["resolution"])
    for n, mk in zip(ns, "os^"):
        ax.plot(TOLS, [res["resolution"][str(n)][f"{tol:g}"] for tol in TOLS], mk + "-", ms=5,
                label=f"{n} points")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("tolerance on the support gap")
    ax.set_ylabel("unsupported fraction (interior points)")
    ax.set_ylim(0.5, 1.02)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("(a) exact structure: tolerance and resolution", fontsize=10)
    ks = sorted(float(k) for k in res["kappa"])
    pos = np.arange(len(ks))
    for ax, key, title in ((axes[1], "unsupported", "(b) unsupported fraction"),
                           (axes[2], "ceiling_gap", "(c) relative weighted-sum ceiling gap")):
        for name, col in (("ratio", "#e0821e"), ("extensive", "#1f4e9c")):
            ax.plot(pos, [res["kappa"][f"{k:g}"][name][key] for k in ks], "o-", c=col, label=name)
        ax.axvline(ks.index(0.0), c="0.6", lw=0.8)
        ax.set_xticks(pos, [f"{k:g}" for k in ks], fontsize=8)
        ax.set_xlabel(r"$\kappa$ in $c = a^{1+\kappa}\varphi(r)$")
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=["all", "resolution", "kappa", "figure"], default="all")
    ap.add_argument("--kappas", default="-0.5,-0.25,0,0.25,0.5,1,2")
    args = ap.parse_args()
    path = ROOT / "results" / "structure_robustness.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    if args.part in ("all", "resolution"):
        res["resolution"] = {str(k): v for k, v in resolution_study().items()}
    if args.part in ("all", "kappa"):
        res.setdefault("kappa", {}).update(kappa_study(tuple(float(k) for k in args.kappas.split(","))))
    path.write_text(json.dumps(res, indent=1))
    if args.part in ("all", "figure") and "resolution" in res and "kappa" in res:
        figure(res, ROOT / "figures" / "fig17_structure_robustness.png")


if __name__ == "__main__":
    main()
