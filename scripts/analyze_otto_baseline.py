"""Analysis and figures of the baseline Otto experiment (efficiency formulation).

Reads results/otto_runs.json, results/otto_reference.npz and results/benchmarks.json.
Writes figures/fig01-fig07 and results/otto_analysis.json.

Usage:
    python scripts/analyze_otto_baseline.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from hidden_front.moead import nondominated_mask  # noqa: E402
from hidden_front.otto import OttoParams, tur_ratio  # noqa: E402
from hidden_front.problems import OttoProblem, physical  # noqa: E402
from hidden_front.support import support_gap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"
GAP_TOL = 1e-3
ALGS = ["NSGA-II", "NSGA-III", "MOEA/D", "MOEA/D-WT", "MOEA/D-AWA"]
WS = "Weighted sum"
COLORS = {"NSGA-II": "#1f77b4", "NSGA-III": "#17becf", "MOEA/D": "#ff7f0e",
          "MOEA/D-WT": "#bcbd22", "MOEA/D-AWA": "#d62728", WS: "#7f7f7f"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def median_run(runs, alg, key="hv"):
    rs = sorted([r for r in runs if r["algorithm"] == alg], key=lambda r: r[key])
    return rs[len(rs) // 2]


def sigma_space_support(params, bins=60):
    """Repeat the convexity analysis with the entropy production Sigma instead of
    eta (the choice of Erdman et al.)."""
    pb = OttoProblem(params)
    om = np.geomspace(pb.xl[0], pb.xu[0], 50)
    rs = np.linspace(pb.xl[1], 1.0, 41)[1:-1]
    ta = np.linspace(pb.xl[2], pb.xu[2], 30)
    G = np.meshgrid(om, rs, ta, ta, indexing="ij")
    s = pb.stats(np.column_stack([g.ravel() for g in G]))
    P, eta, dP = s["P"], s["eta"], s["dP"]
    Sig = P * (params.eta_carnot - eta) / (eta * params.T_c)
    F = np.column_stack([-P, dP, Sig])
    F = F[nondominated_mask(F)]
    Fn = (F - F.min(0)) / (F.max(0) - F.min(0))
    v = np.floor(Fn * bins).clip(0, bins - 1).astype(np.int64)
    _, u = np.unique((v[:, 0] * bins + v[:, 1]) * bins + v[:, 2], return_index=True)
    return support_gap(Fn[u], Fn[u])


def main():
    FIG.mkdir(exist_ok=True)
    data = json.loads((RES / "otto_runs.json").read_text())
    runs = data["runs"]
    params = OttoParams(**data["params"])
    ref = np.load(RES / "otto_reference.npz")
    F_ref, X_ref, ideal, nadir = ref["F_ref"], ref["X_ref"], ref["ideal"], ref["nadir"]
    F_ws = ref["F_ws"]

    def norm(F):
        return (np.asarray(F) - ideal) / (nadir - ideal)

    # ------------------------------------------------ front convexity
    print("Support test on the reference front...", flush=True)
    gap_ref = support_gap(norm(F_ref), norm(F_ref))
    sup = gap_ref <= GAP_TOL
    gap_ws = support_gap(norm(F_ws), norm(F_ref))
    print("Repeating in the (P, dP, Sigma) space...", flush=True)
    gap_sigma = sigma_space_support(params)

    # fraction of each algorithm's solutions in unsupported regions
    tree_ref = norm(F_ref)

    def frac_unsupported(F):
        Fn = norm(F)
        d = ((Fn[:, None, :] - tree_ref[None]) ** 2).sum(-1)
        return float((gap_ref[d.argmin(1)] > GAP_TOL).mean())

    for r in runs:
        r["frac_unsupported"] = frac_unsupported(np.array(r["F"]))

    # ------------------------------------------------ physics
    P, eta, dP = physical(F_ref)
    i_max = np.argmax(P)
    xi = tur_ratio(P, dP, eta, params)
    ok = np.isfinite(xi) & (P > 1e-4 * P.max())

    summary = dict(
        n_reference=int(len(F_ref)),
        frac_unsupported_reference=float((~sup).mean()),
        max_gap_reference=float(gap_ref.max()),
        weighted_sum_max_gap=float(gap_ws.max()),
        sigma_space=dict(n=int(len(gap_sigma)), frac_unsupported=float((gap_sigma > GAP_TOL).mean()),
                         max_gap=float(gap_sigma.max())),
        max_power=dict(P=float(P[i_max]), eta=float(eta[i_max]), dP=float(dP[i_max]),
                       omega_h=float(X_ref[i_max, 0]), r=float(X_ref[i_max, 1]),
                       tau_h=float(10 ** X_ref[i_max, 2]), tau_c=float(10 ** X_ref[i_max, 3])),
        eta_carnot=params.eta_carnot, eta_curzon_ahlborn=float(params.eta_ca),
        tur=dict(max_xi=float(xi[ok].max()), median_xi=float(np.median(xi[ok])),
                 frac_violation=float((xi[ok] > 1).mean())),
        algorithms={a: dict(
            hv_mean=float(np.mean([r["hv"] for r in runs if r["algorithm"] == a])),
            hv_std=float(np.std([r["hv"] for r in runs if r["algorithm"] == a])),
            igd_mean=float(np.mean([r["igd_plus"] for r in runs if r["algorithm"] == a])),
            igd_std=float(np.std([r["igd_plus"] for r in runs if r["algorithm"] == a])),
            frac_unsupported_mean=float(np.mean([r["frac_unsupported"] for r in runs if r["algorithm"] == a])),
            front_size_mean=float(np.mean([len(r["F"]) for r in runs if r["algorithm"] == a])),
        ) for a in ALGS},
        weighted_sum=data["weighted_sum"],
        reference_hv=data["reference"]["hv"],
    )
    (RES / "otto_analysis.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # ================================================= FIGURES
    # 1. 3D front
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(P[~sup], eta[~sup], dP[~sup], s=4, c="#f4a261", alpha=0.6, label="Unreachable by any weighted sum")
    ax.scatter(P[sup], eta[sup], dP[sup], s=7, c="#264653", label="Reachable by a weighted sum")
    ax.set_xlabel("Power P")
    ax.set_ylabel("Efficiency η")
    ax.set_zlabel("Fluctuations ΔP")
    ax.view_init(elev=22, azim=-60)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Pareto front of the single-qubit Otto engine")
    fig.tight_layout()
    fig.savefig(FIG / "fig01_front_3d.png", dpi=200)
    plt.close(fig)

    # 2. P vs eta colored by dP
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sc = ax.scatter(eta, P, c=np.log10(np.maximum(dP, 1e-12)), s=6, cmap="viridis")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("log₁₀ ΔP")
    ax.axvline(params.eta_ca, ls="--", c="k", lw=1)
    ax.text(params.eta_ca, P.max() * 1.03, f" Curzon-Ahlborn\n η = {params.eta_ca:.3f}", fontsize=9, va="bottom")
    ax.axvline(params.eta_carnot, ls=":", c="k", lw=1)
    ax.text(params.eta_carnot, P.max() * 0.55, " Carnot", fontsize=9, ha="left")
    ax.plot(eta[i_max], P[i_max], "r*", ms=14, label=f"Maximum power: η = {eta[i_max]:.3f}")
    ax.set_xlabel("Efficiency η")
    ax.set_ylabel("Power P")
    ax.set_ylim(0, P.max() * 1.18)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Power-efficiency trade-off (color: fluctuations)")
    fig.tight_layout()
    fig.savefig(FIG / "fig02_power_efficiency.png", dpi=200)
    plt.close(fig)

    # 3. Algorithms (median-HV run) in two projections
    names = ALGS + [WS]
    fig, axes = plt.subplots(2, len(names), figsize=(3.1 * len(names), 6.2), sharex="row", sharey="row")
    for j, a in enumerate(names):
        F = F_ws[nondominated_mask(F_ws)] if a == WS else np.array(median_run(runs, a)["F"])
        Pa, ea, da = physical(F)
        for i, (xv, yv, xa, ya) in enumerate([(eta, P, ea, Pa), (P, dP, Pa, da)]):
            ax = axes[i, j]
            ax.scatter(xv, yv, s=2, c="0.85")
            ax.scatter(xa, ya, s=10, c=COLORS[a])
            if i == 0:
                ax.set_title(a, fontsize=11)
        axes[0, j].set_xlabel("η")
        axes[1, j].set_xlabel("P")
    axes[0, 0].set_ylabel("P")
    axes[1, 0].set_ylabel("ΔP")
    fig.suptitle("Final solutions (median-hypervolume run; gray = reference front)")
    fig.tight_layout()
    fig.savefig(FIG / "fig03_algorithms.png", dpi=200)
    plt.close(fig)

    # 4. Metrics
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, key, title in [(axes[0], "hv", "Hypervolume (↑)"), (axes[1], "igd_plus", "IGD+ (↓)"),
                           (axes[2], "frac_unsupported", "Fraction in the unsupported region")]:
        vals = [[r[key] for r in runs if r["algorithm"] == a] for a in ALGS]
        bp = ax.boxplot(vals, patch_artist=True, widths=0.6)
        for patch, a in zip(bp["boxes"], ALGS):
            patch.set_facecolor(COLORS[a])
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(ALGS) + 1), ALGS, rotation=30, ha="right")
        ax.set_title(title)
        if key == "hv":
            ax.axhline(data["weighted_sum"]["hv"], c=COLORS[WS], ls="--", label=WS)
            ax.axhline(data["reference"]["hv"], c="k", ls=":", label="Reference front")
            ax.legend(fontsize=8)
        if key == "igd_plus":
            ax.axhline(data["weighted_sum"]["igd_plus"], c=COLORS[WS], ls="--", label=WS)
            ax.legend(fontsize=8)
        if key == "frac_unsupported":
            ax.axhline((~sup).mean(), c="k", ls=":", label="Reference front")
            ax.legend(fontsize=8)
    fig.suptitle(f"{data['seeds']} seeds, {data['gens']} generations, N = 105")
    fig.tight_layout()
    fig.savefig(FIG / "fig04_metrics.png", dpi=200)
    plt.close(fig)

    # 5. Why the front is not convex: slices at constant eta
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    eta_grid, counts = np.unique(np.round(eta, 9), return_counts=True)
    eta_grid = eta_grid[counts >= 20]                 # r values of the sweep
    for e, c in zip([0.1, 0.2, 0.3, 0.4], ["#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]):
        e0 = eta_grid[np.argmin(np.abs(eta_grid - e))]
        m = np.abs(eta - e0) < 1e-8
        o = np.argsort(P[m])
        ax.plot(P[m][o], dP[m][o], "o", c=c, ms=3, label=f"η = {e0:.3f}")
        pp = np.linspace(0, P[m].max(), 50)
        ax.plot(pp, 2 * params.T_c * e0 / (params.eta_carnot - e0) * pp, "--", c="k", lw=0.9, zorder=3)
    ax.plot([], [], "k--", lw=1, label="TUR bound: ΔP = 2T_c ηP/(η_C − η)")
    ax.set_xlabel("Power P")
    ax.set_ylabel("Fluctuations ΔP")
    ax.set_title("At fixed η the front is a straight line: the TUR bound")
    ax.legend(fontsize=9)
    ax = axes[1]
    ax.scatter(P[~sup], dP[~sup], s=4, c="#f4a261", label="Unsupported")
    ax.scatter(P[sup], dP[sup], s=8, c="#264653", label="Supported")
    Pw, ew, dw_ = physical(F_ws)
    ax.scatter(Pw, dw_, s=40, facecolors="none", edgecolors="k", label="Weighted sum (L-BFGS-B)")
    ax.set_xlabel("Power P")
    ax.set_ylabel("Fluctuations ΔP")
    ax.set_title("Ruled saddle surface: nonconvex interior")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig05_nonconvexity.png", dpi=200)
    plt.close(fig)

    # 6. TUR ratio
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1.4, 1]))
    ax = axes[0]
    sc = ax.scatter(eta[ok], P[ok], c=xi[ok], s=6, cmap="coolwarm",
                    norm=TwoSlopeNorm(vcenter=1.0, vmin=min(0.9, xi[ok].min()), vmax=max(1.01, xi[ok].max())))
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("ξ = 2T_c Pη / [ΔP(η_C − η)]")
    ax.set_xlabel("Efficiency η")
    ax.set_ylabel("Power P")
    ax.set_title("TUR ratio over the front (steady-state TUR: ξ ≤ 1)")
    ax = axes[1]
    ax.hist(xi[ok], bins=60, color="#457b9d")
    ax.axvline(1.0, c="k", ls="--", lw=1)
    ax.set_xlabel("ξ")
    ax.set_ylabel("Front points")
    ax.set_title(f"Median ξ = {np.median(xi[ok]):.3f};  max = {xi[ok].max():.4f}")
    fig.tight_layout()
    fig.savefig(FIG / "fig06_tur.png", dpi=200)
    plt.close(fig)

    # 7. Benchmarks
    bpath = RES / "benchmarks.json"
    if bpath.exists():
        bench = json.loads(bpath.read_text())
        variants = list(dict.fromkeys(r["variant"] for r in bench))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, p in zip(axes[:2], ["DTLZ2", "DTLZ7"]):
            vals = [[r["igd_plus"] for r in bench if r["problem"] == p and r["variant"] == v] for v in variants]
            ax.boxplot(vals, widths=0.6)
            ax.set_xticks(range(1, len(variants) + 1), variants, rotation=25, ha="right", fontsize=9)
            ax.set_title(f"{p}: IGD+ (↓)")
            ax.set_yscale("log")
        ax = axes[2]
        ax.remove()
        ax = fig.add_subplot(1, 3, 3, projection="3d")
        for v, c in [("MOEA/D", "#ff7f0e"), ("MOEA/D-AWA (wag=20)", "#d62728")]:
            rs = sorted([r for r in bench if r["problem"] == "DTLZ7" and r["variant"] == v],
                        key=lambda r: r["igd_plus"])
            F = np.array(rs[len(rs) // 2]["F"])
            ax.scatter(*F.T, s=8, c=c, label=v)
        ax.set_title("DTLZ7 (median run)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "fig07_benchmarks.png", dpi=200)
        plt.close(fig)
    print("Figures written to", FIG)


if __name__ == "__main__":
    main()
