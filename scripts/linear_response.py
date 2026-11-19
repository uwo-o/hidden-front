"""The degenerate limit: linear-response fronts of Almanza-Marrero, Roldan and Manzano (2026).

Model (arXiv:2603.19885): J = L A with symmetric L, L12 = q sqrt(L11 L22), |q| <= 1
(q = 1: tight coupling, endoreversible machine). Fixed resources: A2 = L22 = T = 1.
Design variables: A1, L11 (and q for non-endoreversible machines).
  P = -A1 J1,  eta_bar = eta / eta_C = -A1 J1 / (A2 J2),  Sigma = A1 J1 + A2 J2,
  sigma_P^2 = 2 A1^2 L11.

With y = sqrt(L11) A1 everything depends on (y, q) only:
  P = -y (y + q),  eta_bar = -y (y + q) / (1 + q y),  Sigma = 1 + 2 q y + y^2,  sigma_P^2 = 2 y^2.
For q = 1 and e = -y in [0, 1]:  P = e (1 - e), eta_bar = e, Sigma = (1 - e)^2, sigma_P^2 = 2 e^2.
-> the front is a CURVE (one effective degree of freedom) lying on the TUR bound
   (sigma_P^2 Sigma = 2 P^2).

The script checks numerically that:
  1. q = 1 dominates every q < 1 (the front is the endoreversible curve);
  2. the sampled front matches the analytic curve;
  3. all of its points are supported (LP test), in the four objectives and in subsets of three;
  4. it saturates the TUR;
  5. the weighted figure of merit Omega (Eq. 6 of the paper) covers e in [0, 1] as the
     weights are swept.

Usage:
    python scripts/linear_response.py
Writes results/linear_response.json and figures/fig14_linear_response.png.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hidden_front.moead import nondominated_mask  # noqa: E402
from hidden_front.problems import OttoProblem  # noqa: E402
from hidden_front.support import support_gap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"


def machine(A1, L11, q):
    L12 = q * np.sqrt(L11)
    J1 = L11 * A1 + L12
    J2 = L12 * A1 + 1.0
    P = -A1 * J1
    return P, P / J2, A1 * J1 + J2, 2 * A1 ** 2 * L11


def main():
    rng = np.random.default_rng(0)
    # sample the design space (A1, L11, q)
    n = 200_000
    L11 = 10 ** rng.uniform(-2, 2, n)
    q = np.where(rng.random(n) < 0.3, 1.0, rng.uniform(0, 1, n))
    y = -rng.uniform(0, 1, n) * q                      # engine regime: 0 < -y < q
    A1 = y / np.sqrt(L11)
    P, eta, Sig, var = machine(A1, L11, q)
    ok = (P > 0) & (eta > 0)
    P, eta, Sig, var, q = P[ok], eta[ok], Sig[ok], var[ok], q[ok]

    F = np.column_stack([-P, -eta, Sig, var])          # minimize: -P, -eta, Sigma, sigma^2
    nd = nondominated_mask(F)
    out = {"n_samples": int(ok.sum()), "n_front": int(nd.sum()),
           "frac_front_with_q1": float((q[nd] > 1 - 1e-12).mean())}

    # distance of the front to the analytic curve, parametrized by e = eta_bar
    e = eta[nd]
    out["max_error_curve"] = float(np.max(np.abs(np.column_stack(
        [P[nd] - e * (1 - e), Sig[nd] - (1 - e) ** 2, var[nd] - 2 * e ** 2]))))
    out["max_error_tur"] = float(np.max(np.abs(var[nd] * Sig[nd] - 2 * P[nd] ** 2)))

    # support test on the curve (uniform sampling in e)
    eg = np.linspace(0.005, 0.995, 400)
    C = np.column_stack([-eg * (1 - eg), -eg, (1 - eg) ** 2, 2 * eg ** 2])
    gaps = {}
    names = ["P", "eta", "Sigma", "sigma^2"]
    for cols in [(0, 1, 2, 3), (0, 1, 3), (0, 3, 2), (0, 1, 2)]:
        Fc = C[:, cols]
        Fn = (Fc - Fc.min(0)) / (Fc.max(0) - Fc.min(0))
        m = nondominated_mask(Fn)
        g = support_gap(Fn[m], Fn[m])
        gaps["(" + ", ".join(names[c] for c in cols) + ")"] = {
            "n_nondominated": int(m.sum()), "frac_unsupported": float((g > 1e-3).mean()),
            "max_gap": float(g.max())}
    out["support"] = gaps

    # weight sweep of Omega = a P_bar + b eta_bar - c Sigma_bar - d sigma_bar^2
    # (normalized by maxima: P_bar = 4 P, Sigma_bar = Sigma, sigma_bar^2 = sigma^2 / 2)
    W = np.array([w for w in itertools.product(np.linspace(0, 1, 41), repeat=3) if sum(w) <= 1 + 1e-12])
    best = []
    for a, b, c in W:
        d = 1 - a - b - c
        Om = a * 4 * eg * (1 - eg) + b * eg - c * (1 - eg) ** 2 - d * eg ** 2
        best.append(eg[np.argmax(Om)])
    best = np.array(best)
    hist, _ = np.histogram(best, bins=20, range=(0, 1))
    out["weight_sweep"] = {"e_min": float(best.min()), "e_max": float(best.max()),
                           "empty_bins_of_20": int((hist == 0).sum())}
    RES.mkdir(exist_ok=True)
    (RES / "linear_response.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    # figure: the linear-response front is a curve; the finite-time Otto front fills a region
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(4.8, 3.8), constrained_layout=True)
    ref_path = RES / "otto_reference.npz"
    if ref_path.exists():
        so = OttoProblem().stats(np.load(ref_path)["X_ref"])
        ax.scatter(so["P"] / so["P"].max(), so["eta"] / 0.5, s=4, color="#9ecae1", linewidths=0,
                   label="Single-qubit Otto (finite time)")
    ec = np.linspace(0.0, 1.0, 400)
    ax.plot(4 * ec * (1 - ec), ec, color="#08306b", lw=2, label="Linear response (Almanza-Marrero et al.)")
    ax.set_xlabel(r"$P/P_{\max}$")
    ax.set_ylabel(r"$\eta/\eta_C$")
    ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 1.12), markerscale=3)
    ax.set_xlim(0, 1.05)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "fig14_linear_response.png", dpi=200)


if __name__ == "__main__":
    main()
