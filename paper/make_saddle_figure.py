"""Figure 0 of the paper: the TUR-saturating front minus its tangent plane.

Left: ratio formulation (P, eta, dP), a saddle (Theorem 1).
Right: extensive formulation (P, dP, Sigma), convex and flat along a ray (Theorem 2, Corollary 1).
Axes are rescaled around the expansion point; the sign of the curvature is invariant.

Usage:
    python paper/make_saddle_figure.py
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

OUT = Path(__file__).resolve().parent / "figs" / "fig0_saddle.pdf"
T_C, ETA_C = 0.5, 0.5


def phi(e):
    return 2 * T_C * e / (ETA_C - e)


def dphi(e):
    return 2 * T_C * ETA_C / (ETA_C - e) ** 2


def main(n=41):
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11})
    u = np.linspace(-1, 1, n)
    U, V = np.meshgrid(u, u)
    cmap = LinearSegmentedColormap.from_list("d", ["#e0821e", "#f4f4f4", "#004B85"])

    # Saddle: g(P, eta) = P phi(eta) minus its tangent plane at (P0, eta0)
    P0, e0 = 0.5, 0.2
    P, E = P0 + 0.15 * U, e0 + 0.03 * V
    Rg = P * phi(E) - (P0 * phi(e0) + phi(e0) * (P - P0) + P0 * dphi(e0) * (E - e0))
    Rg /= np.abs(Rg).max()
    # Convex trough: h(P, Sigma) = 2 P^2 / Sigma minus its tangent plane at (P0, S0)
    S0 = 0.5
    P2, S = P0 + 0.15 * U, S0 + 0.15 * V
    Rh = 2 * P2 ** 2 / S - (2 * P0 ** 2 / S0 + 4 * P0 / S0 * (P2 - P0) - 2 * P0 ** 2 / S0 ** 2 * (S - S0))
    Rh /= np.abs(Rh).max()

    fig = plt.figure(figsize=(9.4, 3.9))
    for k, (R, xl, yl, title, color) in enumerate([
            (Rg, r"$P$", r"$\eta$", "ratio objective $\\eta$: saddle", "#004B85"),
            (Rh, r"$P$", r"$\Sigma$", "extensive objective $\\Sigma$: convex trough", "#b05a00")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.plot_surface(U, V, np.zeros_like(U), color="0.6", alpha=0.18, lw=0)
        ax.plot_surface(U, V, R, facecolors=cmap(TwoSlopeNorm(0, -1, 1)(R)), lw=0.2, edgecolor="0.55",
                        rstride=2, cstride=2, shade=False, alpha=0.97)
        ax.set_zlim(-1, 1)
        ax.set_xlabel(xl, labelpad=-8)
        ax.set_ylabel(yl, labelpad=-8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_title(title, color=color, fontsize=13)
        ax.view_init(24, -58 if k == 0 else -112)
    fig.text(0.5, 0.02, "TUR-saturated front minus its tangent plane (gray); blue above, orange below",
             ha="center", fontsize=9.5, color="0.35")
    fig.subplots_adjust(left=0, right=1, top=0.9, bottom=0.08, wspace=0.0)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.03)


if __name__ == "__main__":
    main()
