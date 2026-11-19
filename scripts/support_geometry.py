"""Geometry of a TUR-saturating front in the two formulations (Theorems 1-2 of the paper).

A front that saturates the thermodynamic uncertainty relation,

    dP = 2 T_c eta P / (eta_C - eta)          (<=> xi = 1),

is a SADDLE in the minimization coordinates (-P, -eta, dP) and CONVEX in
(-P, dP, Sigma), where Sigma = P (eta_C - eta) / (eta T_c) and the bound reads
dP = 2 P^2 / Sigma.

1. Symbolic Hessians (sympy).
2. Numerical check: fraction of supported points (reachable by a weighted sum)
   on a synthetic TUR-saturating surface, in both coordinate systems.

Usage:
    python scripts/support_geometry.py
Writes results/geometry.json and figures/fig10_geometry.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import sympy as sp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hidden_front.support import support_gap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GAP_TOL = 1e-3


def symbolic():
    P, eta, S, Tc, etaC = sp.symbols("P eta Sigma T_c eta_C", positive=True)
    g = 2 * Tc * eta * P / (etaC - eta)                  # dP(P, eta) on the bound
    H = sp.hessian(g, (P, eta))
    det_eta = sp.simplify(H.det())
    h = 2 * P ** 2 / S                                    # dP(P, Sigma) on the bound
    H2 = sp.hessian(h, (P, S))
    det_sig = sp.simplify(H2.det())
    tr_sig = sp.simplify(H2.trace())
    return dict(hess_eta=str(H), det_eta=str(det_eta),
                hess_sigma=str(H2), det_sigma=str(det_sig), trace_sigma=str(tr_sig))


def sample_surface(n=60, Tc=0.5, etaC=0.5):
    """TUR-saturating surface over the domain 0 < P <= P_max(eta), with P_max(eta)
    a generic hump (the shape of the Otto engine's front)."""
    etas = np.linspace(0.01, etaC - 0.01, n)
    pts = []
    for e in etas:
        pmax = 4 * e * (etaC - e) / etaC ** 2
        for P in np.linspace(pmax / n, pmax, n):
            dP = 2 * Tc * e * P / (etaC - e)
            S = P * (etaC - e) / (e * Tc)
            pts.append((P, e, dP, S))
    return np.array(pts)


def gaps_of(F):
    Fn = (F - F.min(0)) / (F.max(0) - F.min(0))
    return support_gap(Fn, Fn)


def main():
    out = dict(symbolic=symbolic())
    pts = sample_surface()
    P, e, dP, S = pts.T
    g_eta = gaps_of(np.column_stack([-P, -e, dP]))
    g_sig = gaps_of(np.column_stack([-P, dP, S]))
    out["numeric"] = dict(n=int(len(pts)),
                          eta=dict(frac_unsupported=float((g_eta > GAP_TOL).mean()), max_gap=float(g_eta.max())),
                          sigma=dict(frac_unsupported=float((g_sig > GAP_TOL).mean()), max_gap=float(g_sig.max())))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "geometry.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    fig = plt.figure(figsize=(13, 5.2))
    for i, (F, g, labs, title) in enumerate([
            (np.column_stack([P, e, dP]), g_eta, ("Power P", "Efficiency η", "ΔP"),
             "Coordinates (P, η, ΔP): saddle"),
            (np.column_stack([P, S, dP]), g_sig, ("Power P", "Entropy production Σ", "ΔP"),
             "Coordinates (P, Σ, ΔP): convex")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        sup = g <= GAP_TOL
        ax.scatter(*F[~sup].T, s=4, c="#f4a261", label="Unsupported")
        ax.scatter(*F[sup].T, s=4, c="#264653", label="Supported")
        ax.set_xlabel(labs[0])
        ax.set_ylabel(labs[1])
        ax.set_zlabel(labs[2])
        ax.set_title(f"{title}\n{100 * (~sup).mean():.0f}% unsupported")
        ax.view_init(elev=22, azim=-60)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("The same TUR surface (ξ = 1) in two objective parametrizations")
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "fig10_geometry.png", dpi=180)


if __name__ == "__main__":
    main()
