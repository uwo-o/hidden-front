"""Figures of the Spanish study guide. Labels are intentionally in Spanish.

Usage (from the repository root, with the package installed):
    python docs/course-notes-es/figs_guia.py
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hidden_front.otto import cycle_statistics, excited_population  # noqa: E402

OUT = Path(__file__).parent / "figs"
OUT.mkdir(exist_ok=True)
INK, MUTED = "#222222", "#6b6b6b"
HOT, COLD, BLUE = "#c8553d", "#2f6db5", "#1f4e9c"
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "serif", "mathtext.fontset": "cm",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "legend.frameon": False,
})
WH, WC, TH, TC = 2.0354, 2.0354 * 0.70398, 0.6755, 0.6741   # punto de potencia máxima


def fig_poblacion():
    w = np.linspace(0, 6, 400)
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.plot(w, excited_population(w, 1.0), c=HOT, label=r"baño caliente $T_h=1$")
    ax.plot(w, excited_population(w, 0.5), c=COLD, label=r"baño frío $T_c=0.5$")
    ah, ac = excited_population(WH, 1.0), excited_population(WC, 0.5)
    ax.plot([WH], [ah], "o", c=HOT); ax.plot([WC], [ac], "o", c=COLD)
    ax.annotate(r"$a_h=a(\omega_h,T_h)$", (WH, ah), xytext=(WH + 0.5, ah + 0.12), fontsize=8,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.annotate(r"$a_c=a(\omega_c,T_c)$", (WC, ac), xytext=(WC + 1.7, ac + 0.04), fontsize=8,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_xlabel(r"brecha $\omega$"); ax.set_ylabel(r"población excitada $a$")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(OUT / "poblacion.pdf"); plt.close(fig)


def fig_ciclo():
    s = cycle_statistics(WH, WC, TH, TC)
    p1, p2 = float(s["p1"]), float(s["p2"])
    tq = 0.05
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw=dict(width_ratios=[1, 1.4]))
    ax = axs[0]
    # ciclo en el plano (omega, p): 1 caliente (vertical en wh), 2 quench, 3 frío, 4 quench
    ax.annotate("", (WH, p1), (WH, p2), arrowprops=dict(arrowstyle="-|>", color=HOT, lw=1.6))
    ax.annotate("", (WC, p1), (WH, p1), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    ax.annotate("", (WC, p2), (WC, p1), arrowprops=dict(arrowstyle="-|>", color=COLD, lw=1.6))
    ax.annotate("", (WH, p2), (WC, p2), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    ax.text(WH + 0.03, (p1 + p2) / 2, "1: calor\n$Q_h$", color=HOT, fontsize=8, va="center")
    ax.text((WH + WC) / 2, p1 + 0.0012, "2: trabajo", ha="center", fontsize=8)
    ax.text(WC - 0.03, (p1 + p2) / 2, "3: calor\n$Q_c$", color=COLD, fontsize=8, va="center", ha="right")
    ax.text((WH + WC) / 2, p2 - 0.0028, "4: trabajo", ha="center", fontsize=8)
    ax.set_xlim(WC - 0.45, WH + 0.45); ax.set_ylim(p2 - 0.005, p1 + 0.004)
    ax.set_xlabel(r"brecha $\omega$"); ax.set_ylabel(r"población excitada $p$")
    ax.set_title("Ciclo en el plano $(\\omega, p)$", fontsize=9)
    # p(t) a lo largo de 3 ciclos
    ax = axs[1]
    ah, ac = float(excited_population(WH, 1.0)), float(excited_population(WC, 0.5))
    t, p, T, P = 0.0, p2, [], []
    for _ in range(3):
        tt = np.linspace(0, TH, 60); T += list(t + tt); P += list(ah + (p - ah) * np.exp(-tt)); t += TH; p = P[-1]
        T += [t, t + tq]; P += [p, p]; t += tq
        tt = np.linspace(0, TC, 60); T += list(t + tt); P += list(ac + (p - ac) * np.exp(-tt)); t += TC; p = P[-1]
        T += [t, t + tq]; P += [p, p]; t += tq
    ax.plot(T, P, c=INK, lw=1.2)
    ax.axhline(ah, c=HOT, ls=":", lw=0.9); ax.axhline(ac, c=COLD, ls=":", lw=0.9)
    ax.text(T[-1], ah, r" $a_h$", color=HOT, va="center", fontsize=8)
    ax.text(T[-1], ac, r" $a_c$", color=COLD, va="center", fontsize=8)
    ax.set_xlabel(r"tiempo $t$ (unidades de $1/\gamma$)"); ax.set_ylabel(r"$p(t)$")
    ax.set_title("Ciclo límite: termalización incompleta", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "ciclo.pdf"); plt.close(fig)


def fig_trayectorias():
    rng = np.random.default_rng(3)
    ah, ac = float(excited_population(WH, 1.0)), float(excited_population(WC, 0.5))
    xh, xc, dw = np.exp(-TH), np.exp(-TC), WH - WC
    n, ncyc = 6, 400
    s = (rng.random(n) < 0.08).astype(float)
    Wt = np.zeros((ncyc + 1, n))
    for k in range(ncyc):
        s1 = (rng.random(n) < ah + (s - ah) * xh).astype(float)
        s2 = (rng.random(n) < ac + (s1 - ac) * xc).astype(float)
        Wt[k + 1] = Wt[k] + dw * (s1 - s2); s = s2
    st = cycle_statistics(WH, WC, TH, TC)
    k = np.arange(ncyc + 1)
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    for j in range(n):
        ax.step(k, Wt[:, j], where="post", lw=0.8, color="#9ecae1" if j else BLUE)
    m, v = float(st["W"]), float(st["var_w"])
    ax.plot(k, m * k, c=INK, lw=1.2, label=r"media $\theta'(0)\,n$")
    ax.fill_between(k, m * k - np.sqrt(v * k), m * k + np.sqrt(v * k), color=MUTED, alpha=0.18,
                    label=r"$\pm\sqrt{\theta''(0)\,n}$")
    ax.set_xlabel("número de ciclos $n$"); ax.set_ylabel(r"trabajo acumulado $W_n$")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(OUT / "trayectorias.pdf"); plt.close(fig)


def fig_silla():
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    Tc, etaC = 0.5, 0.5
    fig = plt.figure(figsize=(7.0, 3.0))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    P, E = np.meshgrid(np.linspace(0.05, 1, 30), np.linspace(0.03, 0.42, 30))
    G = 2 * Tc * E * P / (etaC - E)
    ax.plot_surface(P, E, G, cmap="Blues", lw=0, alpha=0.95, rstride=1, cstride=1)
    ax.set_xlabel(r"$P$"); ax.set_ylabel(r"$\eta$"); ax.set_zlabel(r"$\Delta P$")
    ax.set_title(r"$\Delta P = 2T_c\eta P/(\eta_C-\eta)$: silla", fontsize=9)
    ax.view_init(24, -130)
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    P, S = np.meshgrid(np.linspace(0.05, 1, 30), np.linspace(0.15, 1, 30))
    ax.plot_surface(P, S, 2 * P ** 2 / S, cmap="Oranges", lw=0, alpha=0.95)
    ax.set_xlabel(r"$P$"); ax.set_ylabel(r"$\Sigma$"); ax.set_zlabel(r"$\Delta P$")
    ax.set_title(r"$\Delta P = 2P^2/\Sigma$: tazón (convexo)", fontsize=9)
    ax.view_init(24, -130)
    for a in fig.axes:
        a.tick_params(labelsize=6, pad=0)
    fig.savefig(OUT / "silla.pdf"); plt.close(fig)


def fig_legendre():
    th = np.linspace(0, np.pi / 2, 300)
    # frente 2D con una parte no convexa (minimizar f1, f2)
    f1 = np.cos(th); f2 = np.sin(th)
    bump = 0.22 * np.exp(-((th - np.pi / 4) / 0.32) ** 2)
    f1, f2 = 1 - (1 - f1) * 1.0, f2
    f1 = 1 - np.sin(th) + 0 * th
    f1 = np.cos(th) ** 1.0
    F1 = (1 - np.cos(th)); F2 = (1 - np.sin(th))           # frente convexo (cuarto de círculo)
    F1, F2 = F1 + bump * np.cos(th), F2 + bump * np.sin(th)  # abolladura -> parte no convexa
    pts = np.column_stack([F1, F2])
    # puntos soportados: minimizan w·f para algún w
    W = np.linspace(0, 1, 2001)
    sup = np.zeros(len(pts), bool)
    for w in W:
        sup[np.argmin(w * F1 + (1 - w) * F2)] = True
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax in axs:
        ax.plot(F1, F2, c=MUTED, lw=0.8)
        ax.scatter(F1[~sup], F2[~sup], s=6, c="#e0821e", label="no soportado", zorder=3)
        ax.scatter(F1[sup], F2[sup], s=6, c=BLUE, label="soportado", zorder=3)
        ax.set_xlabel(r"$f_1$ (minimizar)"); ax.set_aspect("equal")
    axs[0].set_ylabel(r"$f_2$ (minimizar)")
    # rectas de suma ponderada
    for w in [0.25, 0.5, 0.75]:
        k = np.argmin(w * F1 + (1 - w) * F2); c = w * F1[k] + (1 - w) * F2[k]
        x = np.linspace(-0.1, 1.2, 10)
        axs[0].plot(x, (c - w * x) / (1 - w), c=BLUE, lw=0.7, ls="--")
    axs[0].set_xlim(-0.05, 1.1); axs[0].set_ylim(-0.05, 1.1)
    axs[0].set_title("Suma ponderada: rectas que tocan\nsolo la envolvente convexa", fontsize=9)
    axs[0].legend(loc="upper right", fontsize=7.5)
    # Tchebycheff: esquinas que alcanzan un punto no soportado
    k = np.argmin(np.abs(th - np.pi / 4)); z = np.array([0.0, 0.0])
    p = pts[k]; d = p - z; w = (1 / d) / (1 / d).sum()
    lev = (w * d).max()
    xs = np.array([z[0], lev / w[0], lev / w[0]]); ys = np.array([lev / w[1], lev / w[1], z[1]])
    axs[1].plot([z[0], lev / w[0], lev / w[0]], [lev / w[1], lev / w[1], -0.05], c="#2a7a3b", lw=1.0)
    axs[1].plot([z[0], p[0]], [z[1], p[1]], c="#2a7a3b", lw=0.6, ls=":")
    axs[1].plot(*p, "*", c="#2a7a3b", ms=9)
    axs[1].set_title("Tchebycheff: la esquina del nivel\nalcanza un punto no soportado", fontsize=9)
    axs[1].set_xlim(-0.05, 1.1)
    fig.tight_layout(); fig.savefig(OUT / "legendre.pdf"); plt.close(fig)


if __name__ == "__main__":
    fig_poblacion(); fig_ciclo(); fig_trayectorias(); fig_silla(); fig_legendre()
    print("ok")
