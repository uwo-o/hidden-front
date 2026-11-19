"""Analysis of the main experiment (run_experiment.py) for one engine model.

For each formulation (sigma: (P, dP, Sigma); eta: (P, eta, dP)):
  * reference front = nondominated union of all runs, plus five long SMS-EMOA runs
    (piecewise; scripts/refine_reference.py) or the dense four-variable sweep
    (otto; scripts/run_otto_baseline.py), thinned on a voxel grid;
  * normalized HV (reference point 1.1) and IGD+; two-sided Mann-Whitney U tests
    against MOEA/D-AWA with Holm correction;
  * unsupported fraction of the reference front (LP support test) and the
    fraction of each run lying in the unsupported region;
  * weighted-sum ceiling: HV of the supported part of the reference front;
  * TUR ratio xi over the reference front.

Usage:
    python scripts/analyze_experiment.py [--model piecewise|otto]
Writes results/{pw,ot}_summary.{json,md}, results/{pw,ot}_hv_runs.json and two figures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from scipy.stats import mannwhitneyu

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hidden_front.moead import nondominated_mask  # noqa: E402
from hidden_front.problems import OttoProblem, PiecewiseQDProblem  # noqa: E402
from hidden_front.support import support_gap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
GAP_TOL = 1e-3
ALGS = ["NSGA-II", "NSGA-III", "SMS-EMOA", "RVEA", "MOEA/D", "MOEA/D-AWA", "MOEA/D-WS"]
LABEL = {"sigma": "(P, ΔP, Σ)", "eta": "(P, η, ΔP)"}
COL = {"NSGA-II": "#1f77b4", "NSGA-III": "#17becf", "SMS-EMOA": "#9467bd", "RVEA": "#8c564b",
       "MOEA/D": "#ff7f0e", "MOEA/D-AWA": "#d62728", "MOEA/D-WS": "#7f7f7f"}
PREFIX = {"piecewise": "pw", "otto": "ot"}
FIGURES = {"piecewise": ("fig11_piecewise_hv.png", "fig12_piecewise_fronts.png",
                         "Piecewise control (8 steps, 17 variables)",
                         "Front of the piecewise-controlled quantum-dot engine"),
           "otto": ("fig15_otto_hv.png", "fig16_otto_fronts.png",
                    "Single-qubit Otto engine (4 variables)", "Front of the single-qubit Otto engine")}


def holm(pvals):
    """Holm step-down adjustment of a vector of p-values."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def thin(F, bins=60):
    """Indices of one point per voxel of a normalized three-objective front."""
    v = np.floor(F * bins).clip(0, bins - 1).astype(np.int64)
    _, u = np.unique((v[:, 0] * bins + v[:, 1]) * bins + v[:, 2], return_index=True)
    return u


def tc_etac(pb):
    if isinstance(pb, OttoProblem):
        return pb.params.T_c, pb.params.eta_carnot
    return 1 / pb.qd.beta_c, pb.qd.eta_carnot


def analyze(form, model="piecewise"):
    pre = PREFIX[model]
    data = json.loads((RES / f"{pre}_{form}_runs.json").read_text())
    runs = data["runs"]
    pb = OttoProblem(objectives=form) if model == "otto" else \
        PiecewiseQDProblem(n_steps=data["steps"], objectives=form)
    extra = []
    long_path = RES / f"{pre}_{form}_long.json"
    if long_path.exists():   # long SMS-EMOA runs: they only refine the reference front
        extra = json.loads(long_path.read_text())["runs"]
    X_all = np.vstack([np.array(r["X"]) for r in runs + extra])
    F_all = np.vstack([np.array(r["F"]) for r in runs + extra])
    if model == "otto":      # dense four-variable sweep (both formulations) as extra reference
        Xd = np.vstack([np.load(RES / f)["X_ref"] for f in ("otto_reference.npz", "otto_sigma_reference.npz")])
        X_all = np.vstack([X_all, Xd])
        F_all = np.vstack([F_all, pb.objectives_from_stats(pb.stats(Xd))])
    nd = nondominated_mask(F_all)
    X_ref, F_ref = X_all[nd], F_all[nd]
    ideal, nadir = F_ref.min(0), F_ref.max(0)

    def norm(F):
        return (np.asarray(F) - ideal) / (nadir - ideal)

    u = thin(norm(F_ref))
    X_ref, F_ref = X_ref[u], F_ref[u]
    Fr = norm(F_ref)
    print(f"[{form}] reference: {nd.sum()} nondominated -> {len(Fr)} after thinning; support LP...", flush=True)
    gaps = support_gap(Fr, Fr)

    hv, igd = HV(ref_point=np.full(3, 1.1)), IGDPlus(Fr)
    for r in runs:
        F = norm(np.array(r["F"]))
        r["hv"], r["igd"] = float(hv(F)), float(igd(F))
        d = ((F[:, None, :] - Fr[None]) ** 2).sum(-1)
        r["unsup"] = float((gaps[d.argmin(1)] > GAP_TOL).mean())
        r["n_distinct"] = int(len(np.unique(np.round(F, 3), axis=0)))

    # physics on the reference front
    s = pb.stats(X_ref)
    P, eta, dP = s["P"], s["eta"], s["dP"]
    Tc, etaC = tc_etac(pb)
    ok = (P > 1e-3 * P.max()) & (eta > 0) & (eta < etaC)
    xi = 2 * Tc * P[ok] * eta[ok] / (dP[ok] * (etaC - eta[ok]))

    awa = [r["hv"] for r in runs if r["algorithm"] == "MOEA/D-AWA"]
    others = [a for a in ALGS if a != "MOEA/D-AWA"]
    raw_p = [mannwhitneyu(awa, [r["hv"] for r in runs if r["algorithm"] == a], alternative="two-sided").pvalue
             for a in others]
    adj = dict(zip(others, holm(np.array(raw_p))))
    table = {}
    for a in ALGS:
        rs = [r for r in runs if r["algorithm"] == a]
        h = np.array([r["hv"] for r in rs])
        g = np.array([r["igd"] for r in rs])
        table[a] = dict(hv_median=float(np.median(h)), hv_iqr=float(np.subtract(*np.percentile(h, [75, 25]))),
                        igd_median=float(np.median(g)), unsup=float(np.mean([r["unsup"] for r in rs])),
                        distinct=float(np.mean([r["n_distinct"] for r in rs])),
                        seconds=float(np.median([r["seconds"] for r in rs])),
                        p_holm_vs_awa=None if a == "MOEA/D-AWA" else float(adj[a]))
    long_hv = [float(hv(norm(np.array(r["F"])))) for r in extra]
    # ceiling of any weighted-sum method: HV of the supported points of the reference front
    hv_sup = float(hv(Fr[gaps <= GAP_TOL]))
    summary = dict(objectives=LABEL[form], n_reference=int(len(Fr)), hv_reference=float(hv(Fr)),
                   hv_supported=hv_sup,
                   hv_long_smsemoa=long_hv,
                   frac_unsupported=float((gaps > GAP_TOL).mean()), max_gap=float(gaps.max()),
                   tur=dict(median_xi=float(np.median(xi)), max_xi=float(xi.max()),
                            frac_violation=float((xi > 1).mean()), n=int(len(xi))),
                   algorithms=table)
    return summary, runs, dict(Fr=Fr, F_ref=F_ref, gaps=gaps, P=P, eta=eta, dP=dP, Sigma=s["Sigma"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=["piecewise", "otto"], default="piecewise")
    model = ap.parse_args().model
    pre = PREFIX[model]
    figs = FIGURES[model]
    FIG.mkdir(exist_ok=True)
    out, allruns, refs = {}, {}, {}
    for form in ["sigma", "eta"]:
        out[form], allruns[form], refs[form] = analyze(form, model)
    (RES / f"{pre}_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    (RES / f"{pre}_hv_runs.json").write_text(json.dumps({form: dict(
        hv_reference=out[form]["hv_reference"], hv_supported=out[form]["hv_supported"],
        runs=[dict(algorithm=r["algorithm"], seed=r["seed"], hv=r["hv"], igd=r["igd"]) for r in allruns[form]])
        for form in out}, indent=1))

    with open(RES / f"{pre}_summary.md", "w") as fh:
        for form, sm in out.items():
            fh.write(f"\n## {sm['objectives']}\n\nReference front: {sm['n_reference']} points, "
                     f"HV = {sm['hv_reference']:.4f}; unsupported: {100 * sm['frac_unsupported']:.0f}% "
                     f"(max gap {sm['max_gap']:.3f}); TUR: median xi = {sm['tur']['median_xi']:.3f}, "
                     f"max = {sm['tur']['max_xi']:.3f}; weighted-sum ceiling (HV of the supported points) "
                     f"= {sm['hv_supported']:.4f}\n\n")
            fh.write("| Algorithm | median HV | IQR | median IGD+ | % in unsupported region | "
                     "distinct solutions | p (Holm) vs AWA |\n|---|---|---|---|---|---|---|\n")
            for a, t in sorted(sm["algorithms"].items(), key=lambda kv: -kv[1]["hv_median"]):
                pv = "-" if t["p_holm_vs_awa"] is None else f"{t['p_holm_vs_awa']:.1e}"
                fh.write(f"| {a} | {t['hv_median']:.4f} | {t['hv_iqr']:.4f} | {t['igd_median']:.4f} | "
                         f"{100 * t['unsup']:.0f}% | {t['distinct']:.0f} | {pv} |\n")
    print((RES / f"{pre}_summary.md").read_text())

    # HV per algorithm in both formulations
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    for ax, form in zip(axes, ["sigma", "eta"]):
        runs = allruns[form]
        order = sorted(ALGS, key=lambda a: -np.median([r["hv"] for r in runs if r["algorithm"] == a]))
        bp = ax.boxplot([[r["hv"] for r in runs if r["algorithm"] == a] for a in order],
                        patch_artist=True, widths=0.6)
        for b, a in zip(bp["boxes"], order):
            b.set_facecolor(COL[a])
            b.set_alpha(0.75)
        ax.axhline(out[form]["hv_reference"], c="k", ls=":", lw=1, label="Reference front")
        ax.axhline(out[form]["hv_supported"], c="0.45", ls="--", lw=1, label="Ceiling of any weighted sum")
        ax.set_xticks(range(1, len(order) + 1), order, rotation=30, ha="right")
        ax.set_title(f"Objectives {LABEL[form]}: hypervolume (↑)")
        ax.legend(fontsize=8)
    nrun = len(allruns["sigma"]) // len(ALGS)
    nev = int(np.median([r["n_eval"] for r in allruns["sigma"]]))
    fig.suptitle(f"{figs[2]}, {nrun} seeds, {nev:,} evaluations")
    fig.tight_layout()
    fig.savefig(FIG / figs[0], dpi=180)
    plt.close(fig)

    # Reference fronts colored by support, both formulations
    fig = plt.figure(figsize=(14, 5.5))
    for i, form in enumerate(["eta", "sigma"]):
        r = refs[form]
        sup = r["gaps"] <= GAP_TOL
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        third = r["eta"] if form == "eta" else r["Sigma"]
        ax.scatter(r["P"][~sup], third[~sup], r["dP"][~sup], s=5, c="#f4a261", label="Unsupported")
        ax.scatter(r["P"][sup], third[sup], r["dP"][sup], s=5, c="#264653", label="Supported")
        ax.set_xlabel("Power P")
        ax.set_ylabel("η" if form == "eta" else "Σ")
        ax.set_zlabel("ΔP")
        ax.set_title(f"{LABEL[form]}: {100 * (~sup).mean():.0f}% unsupported")
        ax.view_init(elev=22, azim=-60)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(figs[3])
    fig.tight_layout()
    fig.savefig(FIG / figs[1], dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
