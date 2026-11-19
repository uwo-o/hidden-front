"""Figures 1-5 of the paper (also used by the physics draft in paper/physics-draft).

Usage (from the repository root):
    python paper/make_figures.py
Reads results/{physical_cost.npz, ot_hv_runs.json, pw_hv_runs.json} and writes paper/figs/*.pdf.
Figure 1 recomputes the support test on a synthetic TUR-saturating surface (about 1 min).
Figure 0 (saddle vs. convex trough) is produced by paper/make_saddle_figure.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
OUT = HERE / "figs"
OUT.mkdir(exist_ok=True)

INK, MUTED = "#222222", "#6b6b6b"
SUP, UNSUP = "#1f4e9c", "#e0821e"          # supported / unsupported (CVD-distinct blue/orange)
plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "font.family": "serif", "mathtext.fontset": "cm",
})
COL1, COL2 = 3.4, 7.0                        # REVTeX column / page widths (in)
ALGS = ["SMS-EMOA", "MOEA/D-AWA", "MOEA/D", "NSGA-III", "RVEA", "NSGA-II", "MOEA/D-WS"]
NAME = {a: a for a in ALGS}


def fig_geometry():
    from hidden_front.support import support_gap
    Tc = etaC = 0.5
    n = 50
    pts = []
    for e in np.linspace(0.01, etaC - 0.01, n):
        pmax = 4 * e * (etaC - e) / etaC ** 2
        for P in np.linspace(pmax / n, pmax, n):
            pts.append((P, e, 2 * Tc * e * P / (etaC - e), P * (etaC - e) / (e * Tc)))
    P, e, dP, S = np.array(pts).T
    res = {}
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.5), sharey=True)
    for ax, F, title in [(axs[0], np.column_stack([-P, -e, dP]), r"objectives $(P,\eta,\Delta P)$"),
                         (axs[1], np.column_stack([-P, dP, S]), r"objectives $(P,\Delta P,\Sigma)$")]:
        Fn = (F - F.min(0)) / (F.max(0) - F.min(0))
        g = support_gap(Fn, Fn)
        sup = g <= 1e-3
        res[title] = float((~sup).mean())
        ax.scatter(P[~sup], e[~sup] / etaC, s=5, c=UNSUP, lw=0, label="unsupported")
        ax.scatter(P[sup], e[sup] / etaC, s=5, c=SUP, lw=0, label="supported")
        ax.set_title(f"{title}: {100 * (~sup).mean():.0f}% unsupported")
        ax.set_xlabel(r"$P/P_{\max}$")
    axs[0].set_ylabel(r"$\eta/\eta_C$")
    axs[0].legend(frameon=False, loc="upper right", markerscale=2)
    fig.savefig(OUT / "fig1_geometry.pdf")
    plt.close(fig)
    return res


def fig_cost():
    d = np.load(DATA / "physical_cost.npz")
    pr, er = d["P_over_Pmax"], d["eta_over_etaC"]
    cmap = LinearSegmentedColormap.from_list("b", plt.cm.Blues(np.linspace(0.18, 1.0, 256)))
    vmax = 100 * np.percentile(d["eps_eta"], 99)
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.6), sharey=True, constrained_layout=True)
    for ax, eps, title in [(axs[0], d["eps_eta"], r"linear merit in $(P,\eta,\Delta P)$"),
                           (axs[1], d["eps_sigma"], r"linear merit in $(P,\Delta P,\Sigma)$")]:
        o = np.argsort(eps)
        sc = ax.scatter(pr[o], er[o], c=100 * eps[o], cmap=cmap, vmin=0, vmax=vmax, s=4, lw=0)
        ax.set_title(title)
        ax.set_xlabel(r"$P/P_{\max}$")
        ax.text(0.97, 0.97, rf"$\varepsilon>5\%$ on {100 * (eps > 0.05).mean():.0f}% of the front",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)
    axs[0].set_ylabel(r"$\eta/\eta_C$")
    cb = fig.colorbar(sc, ax=axs, shrink=0.95, pad=0.02)
    cb.set_label(r"loss $\varepsilon$ (%)")
    cb.outline.set_visible(False)
    fig.savefig(OUT / "fig2_cost.pdf")
    plt.close(fig)


def fig_hv(fname, out, title_fmt):
    d = json.loads((DATA / fname).read_text())
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.5), constrained_layout=True)
    for ax, form, lab in [(axs[0], "eta", r"$(P,\eta,\Delta P)$"), (axs[1], "sigma", r"$(P,\Delta P,\Sigma)$")]:
        runs = d[form]["runs"]
        order = sorted(ALGS, key=lambda a: -np.median([r["hv"] for r in runs if r["algorithm"] == a]))
        data = [[r["hv"] for r in runs if r["algorithm"] == a] for a in order]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=True,
                        medianprops=dict(color=INK, lw=1), flierprops=dict(ms=2.5, mec=MUTED),
                        whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
        for b, a in zip(bp["boxes"], order):
            b.set_facecolor("#c9d6ea" if a != "MOEA/D-WS" else UNSUP)
            b.set_edgecolor(MUTED)
        ax.axhline(d[form]["hv_reference"], c=INK, ls=":", lw=0.9, label="reference front")
        ax.axhline(d[form]["hv_supported"], c=UNSUP, ls="--", lw=0.9, label="any weighted sum (ceiling)")
        ax.set_xticks(range(1, len(order) + 1), [NAME[a] for a in order], rotation=35, ha="right")
        ax.set_title(title_fmt.format(lab))
    axs[0].set_ylabel("hypervolume")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(OUT / out)
    plt.close(fig)


def fig_linear_response():
    d = np.load(DATA / "physical_cost.npz")
    fig, ax = plt.subplots(figsize=(COL1, 2.5), constrained_layout=True)
    ax.scatter(d["P_over_Pmax"], d["eta_over_etaC"], s=3, c="#9ecae1", lw=0,
               label="finite-time Otto (this work)")
    e = np.linspace(0, 1, 400)
    ax.plot(4 * e * (1 - e), e, c="#08306b", lw=1.6, label="linear response [Almanza-Marrero et al.]")
    ax.set_xlabel(r"$P/P_{\max}$")
    ax.set_ylabel(r"$\eta/\eta_C$")
    ax.set_xlim(0, 1.04)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), markerscale=3, ncol=1)
    fig.savefig(OUT / "fig4_linear_response.pdf")
    plt.close(fig)


def fig_hv_extended(prefix, out, labels):
    """Horizontal HV boxplots of all algorithms (main and extra scalarizations) for both formulations."""
    path = DATA / f"{prefix}_extended.json"
    if not path.exists():
        print(f"skip {out}: run scripts/analyze_extended.py first")
        return
    summ = json.loads(path.read_text())
    runs = json.loads((DATA / f"{prefix}_extended_runs.json").read_text())
    fig, axs = plt.subplots(1, len(summ), figsize=(COL2, 3.0), constrained_layout=True)
    for ax, res, rr, lab in zip(np.atleast_1d(axs), summ, runs, labels):
        main = rr["runs"].get("main", []) + rr["runs"].get("scal", [])
        algs = [a for a in res["tables"]["main"]]
        algs = sorted(algs, key=lambda a: -np.median([r["hv"] for r in main if r["algorithm"] == a]))
        data = [[r["hv"] for r in main if r["algorithm"] == a] for a in algs]
        bp = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=True,
                        medianprops=dict(color=INK, lw=1), flierprops=dict(ms=2.5, mec=MUTED),
                        whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
        for b, a in zip(bp["boxes"], algs):
            b.set_facecolor(UNSUP if a == "MOEA/D-WS" else ("#f2d3b0" if a in ("MOEA/D-L2", "MOEA/D-L4", "MOEA/D-LWS", "MOEA/D-PBI") else "#c9d6ea"))
            b.set_edgecolor(MUTED)
        ax.axvline(res["hv_reference"], c=INK, ls=":", lw=0.9, label="reference front")
        ax.axvline(res["hv_ceiling"], c=UNSUP, ls="--", lw=0.9, label="any weighted sum (ceiling)")
        ax.set_yticks(range(1, len(algs) + 1), algs, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("hypervolume")
        ax.set_title(lab)
    h, l = np.atleast_1d(axs)[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(OUT / out)
    plt.close(fig)


def fig_ro_scale():
    """Number of vessels n of the final solutions of MOEA/D-WS and NSGA-II in both RO formulations."""
    path = DATA / "ro_extended_runs.json"
    if not path.exists():
        print("skip fig8_ro_scale: run scripts/analyze_extended.py --model ro first")
        return
    runs = json.loads(path.read_text())
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.3), sharey=True, constrained_layout=True)
    bins = np.linspace(1, 100, 34)
    for ax, rr, lab in zip(axs, runs, ["ratio $(Q, r, W, n)$", "extensive $(Q, B, W, n)$"]):
        for a, col in (("NSGA-II", "#c9d6ea"), ("MOEA/D-WS", UNSUP)):
            n = np.array(rr["scale_samples"].get(a, []))
            if len(n):
                ax.hist(n, bins=bins, density=True, color=col, alpha=0.85 if a == "MOEA/D-WS" else 1.0,
                        label=a, edgecolor="white", lw=0.3)
        ax.set_xlabel("number of vessels $n$")
        ax.set_title(lab)
    axs[0].set_ylabel("density of final solutions")
    axs[0].legend(frameon=False)
    fig.savefig(OUT / "fig8_ro_scale.pdf")
    plt.close(fig)


def fig_robustness():
    path = DATA / "structure_robustness.json"
    if not path.exists():
        print("skip fig6_robustness: run scripts/structure_robustness.py first")
        return
    import importlib.util
    spec = importlib.util.spec_from_file_location("sr", ROOT / "scripts" / "structure_robustness.py")
    sr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sr)
    sr.figure(json.loads(path.read_text()), OUT / "fig6_robustness.pdf")


if __name__ == "__main__":
    print("geometry:", fig_geometry(), flush=True)
    fig_cost()
    fig_hv("ot_hv_runs.json", "fig3_hv_otto.pdf", "Otto engine, objectives {}")
    fig_linear_response()
    fig_hv("pw_hv_runs.json", "fig5_hv_piecewise.pdf", "Piecewise control, objectives {}")
    fig_robustness()
    fig_hv_extended("ot", "fig3b_hv_otto_all.pdf", [r"Otto, ratio $(P,\eta,\Delta P)$", r"Otto, extensive $(P,\Delta P,\Sigma)$"])
    fig_hv_extended("ro", "fig7_hv_ro.pdf", [r"RO, ratio $(Q,r,W,n)$", r"RO, extensive $(Q,B,W,n)$"])
    fig_ro_scale()
    print("ok")
