"""Saddle vs. convex-trough figure of the talk (Spanish labels).

Usage:
    python talks/2026-thermo-course/fig_silla.py
"""
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11})
Tc, etaC = 0.5, 0.5
n = 41
u = np.linspace(-1, 1, n)
U, V = np.meshgrid(u, u)
cmap = LinearSegmentedColormap.from_list("d", ["#e0821e", "#f4f4f4", "#004B85"])

# saddle: g(P, eta) around (P0, eta0)
P0, e0, dP_, de_ = 0.5, 0.2, 0.15, 0.03
P, E = P0 + dP_ * U, e0 + de_ * V
phi = lambda e: 2 * Tc * e / (etaC - e)
dphi = lambda e: 2 * Tc * etaC / (etaC - e) ** 2
G = P * phi(E)
Rg = G - (P0 * phi(e0) + phi(e0) * (P - P0) + P0 * dphi(e0) * (E - e0))
Rg /= np.abs(Rg).max()
# convex trough: h(P, Sigma) = 2P^2/Sigma around (P0, S0)
S0, dS_ = 0.5, 0.15
P2, S = P0 + 0.15 * U, S0 + dS_ * V
H = 2 * P2 ** 2 / S
Rh = H - (2 * P0 ** 2 / S0 + 4 * P0 / S0 * (P2 - P0) - 2 * P0 ** 2 / S0 ** 2 * (S - S0))
Rh /= np.abs(Rh).max()

fig = plt.figure(figsize=(9.4, 3.9))
for k, (R, xl, yl, tit, col) in enumerate([
        (Rg, r"potencia $P$", r"eficiencia $\eta$", "con eficiencia: silla", "#004B85"),
        (Rh, r"potencia $P$", r"entropía $\Sigma$", "con entropía: canaleta convexa", "#b05a00")]):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    ax.plot_surface(U, V, np.zeros_like(U), color="0.6", alpha=0.18, lw=0)
    ax.plot_surface(U, V, R, facecolors=cmap(TwoSlopeNorm(0, -1, 1)(R)), lw=0.2, edgecolor="0.55",
                    rstride=2, cstride=2, shade=False, alpha=0.97)
    ax.set_zlim(-1, 1)
    ax.set_xlabel(xl, labelpad=-8); ax.set_ylabel(yl, labelpad=-8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(tit, color=col, fontsize=13)
    ax.view_init(24, -58 if k == 0 else -112)
fig.text(0.5, 0.02, "superficie de los mejores ciclos menos su plano tangente (gris)   ·   azul: por encima   ·   naranja: por debajo",
         ha="center", fontsize=9.5, color="0.35")
fig.subplots_adjust(left=0, right=1, top=0.9, bottom=0.08, wspace=0.0)
fig.savefig(Path(__file__).resolve().parent / "images" / "s_silla.pdf", bbox_inches="tight", pad_inches=0.03)
