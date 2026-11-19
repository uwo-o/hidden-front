"""Physical cost of designing the Otto engine with a linear figure of merit.

Question: if a designer selects the cycle by maximizing a linear combination of the
objectives (a weighted sum, like Omega of Almanza-Marrero et al. 2026 or F of Erdman
et al. 2023), which part of the (P, eta, dP) Pareto front can they reach, and how much
do they lose elsewhere?

Two linear figures of merit over the same physical cycles:
  - linear in (P, eta, dP):  max a P + b eta - c dP
  - linear in (P, dP, Sigma): max a P - b dP - c Sigma   (the one of Erdman et al.)
The set reachable by each one is the supported part of its front (LP support test).

Loss of a point p of the (P, eta, dP) front with respect to a reachable set S:
  eps(p) = min_{s in S} max(P_p / P_s, eta_p / eta_s, dP_s / dP_p) - 1,
i.e. the best reachable alternative is at least eps worse (relatively) in some objective.
This is the multiplicative epsilon-indicator of S with respect to {p}.

Usage:
    python scripts/physical_cost.py        (reads results/otto*_reference.npz)
Writes results/physical_cost.{json,npz} and figures/fig13_physical_cost.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from hidden_front.moead import nondominated_mask  # noqa: E402
from hidden_front.problems import OttoProblem  # noqa: E402
from hidden_front.support import support_gap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
GAP_TOL = 1e-3
BINS = 45


def thin(F, bins=BINS):
    v = np.floor(F * bins).clip(0, bins - 1).astype(np.int64)
    _, u = np.unique((v[:, 0] * bins + v[:, 1]) * bins + v[:, 2], return_index=True)
    return np.sort(u)


def front(F):
    """Nondominated, normalized and thinned front; returns indices and normalized coordinates."""
    idx = np.flatnonzero(nondominated_mask(F))
    Fn = (F[idx] - F[idx].min(0)) / (F[idx].max(0) - F[idx].min(0))
    keep = thin(Fn)
    return idx[keep], Fn[keep]


def eps_loss(Pp, Sp):
    """Multiplicative eps of each row of Pp = (P, eta, dP) with respect to the set Sp."""
    out = np.empty(len(Pp))
    for k, (P, e, d) in enumerate(Pp):
        r = np.maximum.reduce([P / Sp[:, 0], e / Sp[:, 1], Sp[:, 2] / d])
        out[k] = max(r.min() - 1.0, 0.0)
    return out


def main():
    pb = OttoProblem()
    X = np.vstack([np.load(RES / "otto_reference.npz")["X_ref"],
                   np.load(RES / "otto_sigma_reference.npz")["X_ref"]])
    s = pb.stats(X)
    P, eta, dP, Sig = s["P"], s["eta"], s["dP"], s["Sigma"]
    ok = (P > 1e-2 * P.max()) & (eta > 1e-3) & (dP > 0) & np.isfinite(Sig)
    X, P, eta, dP, Sig = X[ok], P[ok], eta[ok], dP[ok], Sig[ok]
    phys = np.column_stack([P, eta, dP])

    i_e, Fe = front(np.column_stack([-P, -eta, dP]))
    i_s, Fs = front(np.column_stack([-P, dP, Sig]))
    print(f"eta front: {len(i_e)} points, Sigma front: {len(i_s)} points; support LP...", flush=True)
    g_e = support_gap(Fe, Fe)
    g_s = support_gap(Fs, Fs)
    S_e = phys[i_e[g_e <= GAP_TOL]]          # reachable with a merit linear in (P, eta, dP)
    S_s = phys[i_s[g_s <= GAP_TOL]]          # reachable with a merit linear in (P, dP, Sigma)

    target = phys[i_e]                       # the (P, eta, dP) front to be covered
    eps_e = eps_loss(target, S_e)
    eps_s = eps_loss(target, S_s)

    Pmax, etaC = P.max(), pb.params.eta_carnot
    pr, er = target[:, 0] / Pmax, target[:, 1] / etaC

    def summarize(eps):
        return {"median_pct": 100 * float(np.median(eps)), "p90_pct": 100 * float(np.percentile(eps, 90)),
                "max_pct": 100 * float(eps.max()),
                "frac_eps_gt_1pct": float((eps > 0.01).mean()), "frac_eps_gt_5pct": float((eps > 0.05).mean()),
                "frac_eps_gt_10pct": float((eps > 0.10).mean())}

    # regime: where the large losses of the eta-linear merit are
    big = eps_e > 0.05
    regime = {
        "median_P_over_Pmax_loss_gt_5pct": float(np.median(pr[big])) if big.any() else None,
        "median_eta_over_etaC_loss_gt_5pct": float(np.median(er[big])) if big.any() else None,
        "median_P_over_Pmax_reachable": float(np.median(pr[eps_e <= 1e-9])),
        "median_eta_over_etaC_reachable": float(np.median(er[eps_e <= 1e-9])),
    }
    # concrete example: front point with the largest loss and its best reachable alternative
    k = int(np.argmax(eps_e))
    r = np.maximum.reduce([target[k, 0] / S_e[:, 0], target[k, 1] / S_e[:, 1], S_e[:, 2] / target[k, 2]])
    alt = S_e[int(np.argmin(r))]
    example = {"target": {"P_over_Pmax": float(target[k, 0] / Pmax), "eta_over_etaC": float(target[k, 1] / etaC),
                          "dP": float(target[k, 2])},
               "best_eta_linear": {"P_over_Pmax": float(alt[0] / Pmax), "eta_over_etaC": float(alt[1] / etaC),
                                   "dP": float(alt[2])}}

    out = {"n_front_eta": int(len(i_e)), "n_front_sigma": int(len(i_s)),
           "frac_supported_eta": float((g_e <= GAP_TOL).mean()),
           "frac_supported_sigma": float((g_s <= GAP_TOL).mean()),
           "loss_linear_merit_eta": summarize(eps_e), "loss_linear_merit_sigma": summarize(eps_s),
           "regime": regime, "worst_case_example": example}
    np.savez(RES / "physical_cost.npz", P_over_Pmax=pr, eta_over_etaC=er, eps_eta=eps_e, eps_sigma=eps_s,
             gap_eta=g_e, dP=target[:, 2])
    (RES / "physical_cost.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True, constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list("blues_t", plt.cm.Blues(np.linspace(0.18, 1.0, 256)))
    vmax = max(np.percentile(eps_e, 99), 1e-3) * 100
    for ax, eps, title in [(axs[0], eps_e, "Linear merit in (P, η, ΔP)"),
                           (axs[1], eps_s, "Linear merit in (P, ΔP, Σ)")]:
        order = np.argsort(eps)
        sc = ax.scatter(pr[order], er[order], c=100 * eps[order], cmap=cmap, vmin=0, vmax=vmax,
                        s=9, linewidths=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$P/P_{\max}$")
        ax.text(0.97, 0.97, f"ε > 5%: {100 * (eps > 0.05).mean():.0f}% of the front",
                transform=ax.transAxes, fontsize=9, color="0.25", ha="right", va="top")
    axs[0].set_ylabel(r"$\eta/\eta_C$")
    cb = fig.colorbar(sc, ax=axs, shrink=0.9)
    cb.set_label("loss ε of the best reachable alternative (%)")
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "fig13_physical_cost.png", dpi=200)


if __name__ == "__main__":
    main()
