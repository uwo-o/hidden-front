"""Figures for the Spanish-language talk (16:9). Labels are intentionally in Spanish.

Usage (from the repository root, with the package installed):
    python talks/2026-thermo-course/figs_slides.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from hidden_front.otto import cycle_statistics, excited_population  # noqa: E402

OUT = Path(__file__).parent / "images"
OUT.mkdir(exist_ok=True)
DATA = Path(__file__).resolve().parents[2] / "results"
INK, MUTED = "#222222", "#6b6b6b"
HOT, COLD, BLUE, ORANGE = "#D60019", "#004B85", "#004B85", "#e0821e"
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "serif", "mathtext.fontset": "cm",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "legend.frameon": False,
})
WH, WC, TH, TC = 2.0354, 2.0354 * 0.70398, 0.6755, 0.6741


def poblacion():
    w = np.linspace(0, 6, 400)
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    ax.plot(w, excited_population(w, 1.0), c=HOT, lw=2, label=r"baño caliente $T_h=1$")
    ax.plot(w, excited_population(w, 0.5), c=COLD, lw=2, label=r"baño frío $T_c=0.5$")
    ah, ac = excited_population(WH, 1.0), excited_population(WC, 0.5)
    ax.plot([WH], [ah], "o", c=HOT, ms=8); ax.plot([WC], [ac], "o", c=COLD, ms=8)
    ax.annotate(r"$\omega_h$: más arriba", (WH, ah), xytext=(WH + 0.6, ah + 0.1),
                arrowprops=dict(arrowstyle="-", color=MUTED), fontsize=10)
    ax.annotate(r"$\omega_c$: menos arriba", (WC, ac), xytext=(WC + 1.6, ac + 0.03),
                arrowprops=dict(arrowstyle="-", color=MUTED), fontsize=10)
    ax.set_xlabel(r"brecha $\omega$"); ax.set_ylabel(r"prob. de estar arriba")
    ax.legend(loc="upper right")
    fig.savefig(OUT / "s_poblacion.pdf"); plt.close(fig)


def ciclo():
    s = cycle_statistics(WH, WC, TH, TC)
    p1, p2 = float(s["p1"]), float(s["p2"])
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    kw = dict(lw=2.4, mutation_scale=18)
    ax.annotate("", (WH, p1), (WH, p2), arrowprops=dict(arrowstyle="-|>", color=HOT, **kw))
    ax.annotate("", (WC, p1), (WH, p1), arrowprops=dict(arrowstyle="-|>", color=INK, **kw))
    ax.annotate("", (WC, p2), (WC, p1), arrowprops=dict(arrowstyle="-|>", color=COLD, **kw))
    ax.annotate("", (WH, p2), (WC, p2), arrowprops=dict(arrowstyle="-|>", color=INK, **kw))
    ax.text(WH + 0.04, (p1 + p2) / 2, "1: calor\n    $Q_h$", color=HOT, va="center")
    ax.text((WH + WC) / 2, p1 + 0.0013, "2: trabajo", ha="center")
    ax.text(WC - 0.04, (p1 + p2) / 2, "3: calor\n$Q_c$    ", color=COLD, va="center", ha="right")
    ax.text((WH + WC) / 2, p2 - 0.0033, "4: trabajo", ha="center")
    ax.set_xlim(WC - 0.6, WH + 0.6); ax.set_ylim(p2 - 0.006, p1 + 0.005)
    ax.set_xlabel(r"brecha $\omega$  (el “volumen”)"); ax.set_ylabel(r"población excitada $p$")
    fig.savefig(OUT / "s_ciclo.pdf"); plt.close(fig)


def pt():
    ah, ac = float(excited_population(WH, 1.0)), float(excited_population(WC, 0.5))
    s = cycle_statistics(WH, WC, TH, TC)
    t, p, T, P = 0.0, float(s["p2"]), [], []
    for _ in range(3):
        tt = np.linspace(0, TH, 60); T += list(t + tt); P += list(ah + (p - ah) * np.exp(-tt)); t += TH; p = P[-1]
        T += [t, t + 0.05]; P += [p, p]; t += 0.05
        tt = np.linspace(0, TC, 60); T += list(t + tt); P += list(ac + (p - ac) * np.exp(-tt)); t += TC; p = P[-1]
        T += [t, t + 0.05]; P += [p, p]; t += 0.05
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.plot(T, P, c=INK, lw=1.8)
    ax.axhline(ah, c=HOT, ls=":", lw=1.4); ax.axhline(ac, c=COLD, ls=":", lw=1.4)
    ax.text(T[-1] + 0.05, ah, "equilibrio\ncaliente", color=HOT, va="center", fontsize=9)
    ax.text(T[-1] + 0.05, ac, "equilibrio\nfrío", color=COLD, va="center", fontsize=9)
    ax.set_ylim(0.035, 0.125)
    ax.set_xlabel(r"tiempo"); ax.set_ylabel(r"$p(t)$")
    fig.savefig(OUT / "s_pt.pdf"); plt.close(fig)


def trayectorias():
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
    m, v = float(st["W"]), float(st["var_w"])
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for j in range(n):
        ax.step(k, Wt[:, j], where="post", lw=1.0, color="#8fb3d9")
    ax.plot(k, m * k, c=INK, lw=2, label="promedio: potencia")
    ax.fill_between(k, m * k - np.sqrt(v * k), m * k + np.sqrt(v * k), color=MUTED, alpha=0.18,
                    label="ancho: fluctuaciones")
    ax.set_xlabel("número de ciclos"); ax.set_ylabel("trabajo acumulado")
    ax.legend(loc="upper left")
    fig.savefig(OUT / "s_trayectorias.pdf"); plt.close(fig)


def envolvente():
    th = np.linspace(0, np.pi / 2, 300)
    bump = 0.22 * np.exp(-((th - np.pi / 4) / 0.32) ** 2)
    F1 = (1 - np.cos(th)) + bump * np.cos(th); F2 = (1 - np.sin(th)) + bump * np.sin(th)
    sup = np.zeros(len(th), bool)
    for w in np.linspace(0, 1, 2001):
        sup[np.argmin(w * F1 + (1 - w) * F2)] = True
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.plot(F1, F2, c=MUTED, lw=0.8)
    ax.scatter(F1[~sup], F2[~sup], s=10, c=ORANGE, label="invisible para la suma ponderada", zorder=3)
    ax.scatter(F1[sup], F2[sup], s=10, c=BLUE, label="alcanzable", zorder=3)
    for w in [0.25, 0.5, 0.75]:
        k = np.argmin(w * F1 + (1 - w) * F2); c = w * F1[k] + (1 - w) * F2[k]
        x = np.linspace(-0.1, 1.2, 10)
        ax.plot(x, (c - w * x) / (1 - w), c=BLUE, lw=0.9, ls="--")
    ax.set_xlim(-0.05, 1.1); ax.set_ylim(-0.05, 1.15); ax.set_aspect("equal")
    ax.set_xlabel("objetivo 1 (menos es mejor)"); ax.set_ylabel("objetivo 2 (menos es mejor)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=9, markerscale=1.5)
    fig.savefig(OUT / "s_envolvente.pdf"); plt.close(fig)


def silla():
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    Tc, etaC = 0.5, 0.5
    fig = plt.figure(figsize=(9.0, 3.6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    P, E = np.meshgrid(np.linspace(0.05, 1, 30), np.linspace(0.03, 0.40, 30))
    ax.plot_surface(P, E, 2 * Tc * E * P / (etaC - E), cmap="Blues", lw=0, alpha=0.95)
    ax.set_xlabel(r"potencia $P$"); ax.set_ylabel(r"eficiencia $\eta$"); ax.set_zlabel(r"fluct. $\Delta P$")
    ax.set_title("con eficiencia: silla", fontsize=13, color=BLUE)
    ax.view_init(24, -130)
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    P, S = np.meshgrid(np.linspace(0.05, 1, 30), np.linspace(0.15, 1, 30))
    ax.plot_surface(P, S, 2 * P ** 2 / S, cmap="Oranges", lw=0, alpha=0.95)
    ax.set_xlabel(r"potencia $P$"); ax.set_ylabel(r"entropía $\Sigma$"); ax.set_zlabel(r"fluct. $\Delta P$")
    ax.set_title("con entropía producida: tazón", fontsize=13, color="#b05a00")
    ax.view_init(24, -130)
    for a in fig.axes:
        a.tick_params(labelsize=7, pad=0)
        a.set_xticklabels([]); a.set_yticklabels([]); a.set_zticklabels([])
    fig.savefig(OUT / "s_silla.pdf"); plt.close(fig)


def costo():
    d = np.load(DATA / "physical_cost.npz")
    pr, er = d["P_over_Pmax"], d["eta_over_etaC"]
    cmap = LinearSegmentedColormap.from_list("b", plt.cm.Blues(np.linspace(0.18, 1.0, 256)))
    vmax = 100 * np.percentile(d["eps_eta"], 99)
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True, constrained_layout=True)
    for ax, eps, title in [(axs[0], d["eps_eta"], "figura de mérito con eficiencia"),
                           (axs[1], d["eps_sigma"], "figura de mérito con entropía")]:
        o = np.argsort(eps)
        sc = ax.scatter(pr[o], er[o], c=100 * eps[o], cmap=cmap, vmin=0, vmax=vmax, s=6, lw=0)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(r"potencia / potencia máxima")
    axs[0].set_ylabel(r"eficiencia / Carnot")
    cb = fig.colorbar(sc, ax=axs, shrink=0.95, pad=0.02)
    cb.set_label("pérdida (%)"); cb.outline.set_visible(False)
    fig.savefig(OUT / "s_costo.pdf"); plt.close(fig)


ALGS = ["SMS-EMOA", "MOEA/D-AWA", "MOEA/D", "NSGA-III", "RVEA", "NSGA-II", "MOEA/D-WS"]
NAME = {a: a for a in ALGS} | {"MOEA/D-WS": "suma\nponderada"}


def hv(fname, out):
    d = json.loads((DATA / fname).read_text())
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.4), constrained_layout=True)
    for ax, form, lab in [(axs[0], "eta", "con eficiencia"), (axs[1], "sigma", "con entropía")]:
        runs = d[form]["runs"]
        order = sorted(ALGS, key=lambda a: -np.median([r["hv"] for r in runs if r["algorithm"] == a]))
        bp = ax.boxplot([[r["hv"] for r in runs if r["algorithm"] == a] for a in order], widths=0.55,
                        patch_artist=True, medianprops=dict(color=INK), flierprops=dict(ms=2.5))
        for b, a in zip(bp["boxes"], order):
            b.set_facecolor("#c9d6ea" if a != "MOEA/D-WS" else ORANGE); b.set_edgecolor(MUTED)
        ax.axhline(d[form]["hv_reference"], c=INK, ls=":", lw=1, label="mejor frente conocido")
        ax.axhline(d[form]["hv_supported"], c=ORANGE, ls="--", lw=1.2, label="máximo de cualquier suma ponderada")
        ax.set_xticks(range(1, len(order) + 1), [NAME[a] for a in order], rotation=35, ha="right", fontsize=9)
        ax.set_title(lab, fontsize=12)
    axs[0].set_ylabel("hipervolumen")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(OUT / out); plt.close(fig)


def lr():
    d = np.load(DATA / "physical_cost.npz")
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.scatter(d["P_over_Pmax"], d["eta_over_etaC"], s=4, c="#9ecae1", lw=0, label="Otto en tiempo finito: superficie")
    e = np.linspace(0, 1, 400)
    ax.plot(4 * e * (1 - e), e, c=BLUE, lw=2, label="respuesta lineal: curva")
    ax.set_xlabel("potencia / potencia máxima"); ax.set_ylabel("eficiencia / Carnot")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), markerscale=3)
    fig.savefig(OUT / "s_lr.pdf"); plt.close(fig)


if __name__ == "__main__":
    poblacion(); ciclo(); pt(); trayectorias(); envolvente(); silla(); costo(); lr()
    hv("ot_hv_runs.json", "s_hv_otto.pdf"); hv("pw_hv_runs.json", "s_hv_tramos.pdf")
    print("ok")


def entropia():
    p = np.linspace(1e-4, 1 - 1e-4, 400)
    S = -p * np.log(p) - (1 - p) * np.log(1 - p)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(p, S, c=BLUE, lw=2)
    ax.set_xlabel(r"población excitada $p$"); ax.set_ylabel(r"entropía del qubit $S(p)$")
    ax.axvline(0.0948, c=HOT, ls=":", lw=1.2); ax.axvline(0.0747, c=COLD, ls=":", lw=1.2)
    ax.text(0.11, 0.08, r"$p_1$", color=HOT); ax.text(0.02, 0.08, r"$p_2$", color=COLD)
    fig.savefig(OUT / "s_entropia.pdf"); plt.close(fig)
