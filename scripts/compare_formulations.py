"""Compare the ratio (P, eta, dP) and extensive (P, dP, Sigma) formulations of the baseline Otto runs.

Requires:
    python scripts/run_otto_baseline.py --objectives eta
    python scripts/run_otto_baseline.py --objectives sigma

Writes results/formulation_comparison.json and figures/fig08-fig09.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hidden_front.moead import nondominated_mask  # noqa: E402
from hidden_front.support import support_gap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
GAP_TOL = 1e-3
ALGS = ["NSGA-II", "NSGA-III", "MOEA/D", "MOEA/D-WT", "MOEA/D-AWA"]
LABEL = {"eta": "(P, η, ΔP)", "sigma": "(P, ΔP, Σ)"}
COLORS = {"eta": "#e76f51", "sigma": "#2a9d8f"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def load(tag):
    runs = json.loads((RES / f"{tag}_runs.json").read_text())
    ref = dict(np.load(RES / f"{tag}_reference.npz"))
    return runs, ref


def analyze(form):
    tag = "otto" if form == "eta" else "otto_sigma"
    data, ref = load(tag)
    ideal, nadir = ref["ideal"], ref["nadir"]

    def norm(F):
        return (np.asarray(F) - ideal) / (nadir - ideal)

    Fr = norm(ref["F_ref"])
    gaps_path = RES / f"{tag}_support_gaps.npy"
    if gaps_path.exists() and len(np.load(gaps_path)) == len(Fr):
        gaps = np.load(gaps_path)
    else:
        print(f"[{form}] support LP over {len(Fr)} points...", flush=True)
        gaps = support_gap(Fr, Fr)
        np.save(gaps_path, gaps)

    def frac_unsup(F):
        d = ((norm(F)[:, None, :] - Fr[None]) ** 2).sum(-1)
        return float((gaps[d.argmin(1)] > GAP_TOL).mean())

    runs = data["runs"]
    algs = {}
    for a in ALGS:
        rs = [r for r in runs if r["algorithm"] == a]
        algs[a] = dict(hv=[r["hv"] for r in rs], igd=[r["igd_plus"] for r in rs],
                       unsup=[frac_unsup(np.array(r["F"])) for r in rs])
    F_ws = ref["F_ws"]
    ws_gap = support_gap(norm(F_ws), Fr)
    return dict(form=form, gaps=gaps, Fr=Fr, F_ref=ref["F_ref"], algs=algs,
                ws=data["weighted_sum"], ref_hv=data["reference"]["hv"], F_ws=F_ws,
                ws_max_gap=float(ws_gap.max()), ws_unsup=frac_unsup(F_ws[nondominated_mask(F_ws)]))


def main():
    out = {f: analyze(f) for f in ["eta", "sigma"]}

    summary = {}
    for f, o in out.items():
        awa_hv = np.mean(o["algs"]["MOEA/D-AWA"]["hv"])
        best = max(ALGS, key=lambda a: np.mean(o["algs"][a]["hv"]))
        summary[f] = dict(
            objectives=LABEL[f],
            n_reference=int(len(o["gaps"])),
            frac_unsupported=float((o["gaps"] > GAP_TOL).mean()),
            max_gap=float(o["gaps"].max()),
            hv_reference=o["ref_hv"],
            weighted_sum=dict(hv=o["ws"]["hv"], igd_plus=o["ws"]["igd_plus"],
                              distinct_solutions=o["ws"]["n_unique"]),
            weighted_sum_hv_loss_vs_awa=float((awa_hv - o["ws"]["hv"]) / awa_hv),
            best_by_hv=best,
            algorithms={a: dict(hv=f"{np.mean(v['hv']):.4f} +- {np.std(v['hv']):.4f}",
                                igd_plus=f"{np.mean(v['igd']):.4f} +- {np.std(v['igd']):.4f}",
                                frac_in_unsupported_region=float(np.mean(v["unsup"])))
                        for a, v in o["algs"].items()},
        )
    (RES / "formulation_comparison.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # ---------------- Figure 8: metrics side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    x = np.arange(len(ALGS))
    for k, (key, title) in enumerate([("hv", "Hypervolume (↑)"), ("igd", "IGD+ (↓)")]):
        ax = axes[k]
        for j, f in enumerate(["eta", "sigma"]):
            o = out[f]
            vals = [o["algs"][a][key] for a in ALGS]
            bp = ax.boxplot(vals, positions=x + (j - 0.5) * 0.36, widths=0.3, patch_artist=True,
                            manage_ticks=False)
            for b in bp["boxes"]:
                b.set_facecolor(COLORS[f])
                b.set_alpha(0.75)
            ws_val = o["ws"]["hv"] if key == "hv" else o["ws"]["igd_plus"]
            ax.axhline(ws_val, c=COLORS[f], ls="--", lw=1.2)
        ax.set_xticks(x, ALGS, rotation=25, ha="right")
        ax.set_title(title)
    for f in ["eta", "sigma"]:
        axes[0].plot([], [], c=COLORS[f], lw=6, alpha=0.75, label=LABEL[f])
    axes[0].plot([], [], c="k", ls="--", label="Weighted sum")
    axes[0].legend(fontsize=9)

    ax = axes[2]
    fr = [summary[f]["frac_unsupported"] * 100 for f in ["eta", "sigma"]]
    gm = [summary[f]["max_gap"] for f in ["eta", "sigma"]]
    bars = ax.bar([LABEL["eta"], LABEL["sigma"]], fr, color=[COLORS["eta"], COLORS["sigma"]], alpha=0.85)
    for b, v, g in zip(bars, fr, gm):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%\nmax gap {g:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_ylabel("% of the front unreachable by any weighted sum")
    ax.set_title("How much of the front does linear scalarization miss?")
    fig.suptitle("Same physics, two formulations (10 seeds, 400 gen., N = 105; "
                 "metrics normalized per formulation)")
    fig.tight_layout()
    fig.savefig(FIG / "fig08_eta_vs_sigma.png", dpi=200)
    plt.close(fig)

    # ---------------- Figure 9: front in (P, dP, Sigma)
    o = out["sigma"]
    F = o["F_ref"]
    P, dP, S = -F[:, 0], F[:, 1], F[:, 2]
    sup = o["gaps"] <= GAP_TOL
    fig = plt.figure(figsize=(13, 5.5))
    ax = fig.add_subplot(121, projection="3d")
    ax.scatter(P[~sup], S[~sup], dP[~sup], s=4, c="#f4a261", alpha=0.6, label="Unreachable by any weighted sum")
    ax.scatter(P[sup], S[sup], dP[sup], s=5, c="#264653", alpha=0.6, label="Reachable by a weighted sum")
    ax.set_xlabel("Power P")
    ax.set_ylabel("Entropy production Σ")
    ax.set_zlabel("Fluctuations ΔP")
    ax.view_init(elev=22, azim=-60)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Front in (P, ΔP, Σ)")
    ax = fig.add_subplot(122)
    m = (S > 1e-6) & (P > 1e-3 * P.max())
    ax.loglog(2 * P[m] ** 2 / S[m], dP[m], ".", ms=2, c="#2a9d8f")
    lim = [max(1e-9, dP[m].min()), dP[m].max()]
    ax.loglog(lim, lim, "k--", lw=1, label="ΔP = 2P²/Σ (TUR bound)")
    ax.set_xlabel("2P²/Σ")
    ax.set_ylabel("ΔP")
    ax.set_title("The front lies on the TUR bound")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig09_front_sigma.png", dpi=200)
    plt.close(fig)
    print("Figures written to", FIG)


if __name__ == "__main__":
    main()
