"""Extended analysis: all decomposition baselines, effect sizes and the weighted-sum ceiling.

For one model (otto or ro) and each formulation:
  * reference front = nondominated union of every run file of the model (main experiment, extra
    scalarizations, budget/population sensitivity) plus the dense reference sweep, thinned on a
    voxel grid of the normalized objective space;
  * HV (reference point 1.1) and IGD+ in normalized coordinates;
  * two-sided Mann-Whitney U tests against MOEA/D-AWA with Holm correction, and the Vargha-Delaney
    effect size A12 = P(HV_alg > HV_AWA) + P(=)/2;
  * support test of the reference front, the weighted-sum ceiling and the HV-loss decomposition;
  * fraction of each run in the unsupported region and number of distinct solutions;
  * for the reverse-osmosis model, the fraction of solutions at the bounds of the scale variable.

Usage:
    python scripts/analyze_extended.py --model otto
    python scripts/analyze_extended.py --model ro
Writes results/{ot,ro}_extended.{json,md} and figures/fig18_{ot,ro}_scalarizations.png.
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
from hidden_front.problems import DesalinationProblem, OttoProblem  # noqa: E402
from hidden_front.support import support_gap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
GAP_TOL = 1e-3
ORDER = ["SMS-EMOA", "MOEA/D-AWA", "MOEA/D", "NSGA-III", "RVEA", "NSGA-II",
         "MOEA/D-PBI", "MOEA/D-LWS", "MOEA/D-LWS0.25x", "MOEA/D-LWS0.5x", "MOEA/D-LWS2x", "MOEA/D-LWS4x", "MOEA/D-L4", "MOEA/D-L2", "MOEA/D-WS"]
SETTINGS = {"otto": dict(prefix="ot", forms=("eta", "sigma"), bins=60,
                         label={"eta": "ratio (P, eta, dP)", "sigma": "extensive (P, dP, Sigma)"}),
            "ro": dict(prefix="ro", forms=("recovery", "brine"), bins=16,
                       label={"recovery": "ratio (Q, r, W, n)", "brine": "extensive (Q, B, W, n)"})}


def holm(p):
    order = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for k, i in enumerate(order):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def a12(x, y):
    """Vargha-Delaney A12: probability that a value of x exceeds a value of y (ties count 1/2)."""
    x, y = np.asarray(x)[:, None], np.asarray(y)[None, :]
    return float(np.mean((x > y) + 0.5 * (x == y)))


def thin(G, bins):
    v = np.floor(G * bins).clip(0, bins - 1).astype(np.int64)
    key = np.zeros(len(G), dtype=np.int64)
    for j in range(G.shape[1]):
        key = key * bins + v[:, j]
    _, u = np.unique(key, return_index=True)
    return np.sort(u)


def problem(model, form):
    return OttoProblem(objectives=form) if model == "otto" else DesalinationProblem(objectives=form)


def load_runs(model, form):
    pre = SETTINGS[model]["prefix"]
    groups = {}
    for path in sorted(RES.glob(f"{pre}_{form}*_runs.json")):
        tag = path.stem[len(f"{pre}_{form}"):-len("_runs")].lstrip("_") or "main"
        groups[tag] = json.loads(path.read_text())["runs"]
    return groups


def dense_reference(model, form, pb):
    if model == "otto":
        Xd = np.vstack([np.load(RES / f)["X_ref"] for f in ("otto_reference.npz", "otto_sigma_reference.npz")])
    else:
        d = np.load(RES / "ro_reference.npz")
        Xd = np.vstack([d["recovery_X"], d["brine_X"]])
    return Xd, pb.evaluate(Xd, return_values_of=["F"])


def analyze(model, form):
    cfg = SETTINGS[model]
    pb = problem(model, form)
    groups = load_runs(model, form)
    all_runs = [r for g in groups.values() for r in g]
    Xd, Fd = dense_reference(model, form, pb)
    X_all = np.vstack([Xd] + [np.array(r["X"]) for r in all_runs])
    F_all = np.vstack([Fd] + [np.array(r["F"]) for r in all_runs])
    nd = nondominated_mask(F_all)
    X_ref, F_ref = X_all[nd], F_all[nd]
    ideal, nadir = F_ref.min(0), F_ref.max(0)

    def norm(F):
        return (np.asarray(F) - ideal) / (nadir - ideal)

    u = thin(norm(F_ref), cfg["bins"])
    X_ref, F_ref = X_ref[u], F_ref[u]
    Fr = norm(F_ref)
    print(f"[{model} {form}] reference {nd.sum()} -> {len(Fr)}; support LP...", flush=True)
    gaps = support_gap(Fr, Fr)
    m = Fr.shape[1]
    hv, igd = HV(ref_point=np.full(m, 1.1)), IGDPlus(Fr)
    hv_ref, hv_sup = float(hv(Fr)), float(hv(Fr[gaps <= GAP_TOL]))

    for r in all_runs:
        F = norm(np.array(r["F"]))
        r["hv"], r["igd"] = float(hv(F)), float(igd(F))
        d = ((F[:, None, :] - Fr[None]) ** 2).sum(-1)
        r["unsup"] = float((gaps[d.argmin(1)] > GAP_TOL).mean())
        r["n_distinct"] = int(len(np.unique(np.round(F, 3), axis=0)))
        if model == "ro":
            n = np.array(r["X"])[:, 0]
            r["at_scale_bounds"] = float(np.mean((n < pb.xl[0] + 0.5) | (n > pb.xu[0] - 0.5)))

    def table(runs):
        algs = [a for a in ORDER if any(r["algorithm"] == a for r in runs)]
        base = [r["hv"] for r in runs if r["algorithm"] == "MOEA/D-AWA"]
        rows, raw = {}, {}
        for a in algs:
            rs = [r for r in runs if r["algorithm"] == a]
            h = np.array([r["hv"] for r in rs])
            row = dict(n_runs=len(rs), hv_median=float(np.median(h)),
                       hv_iqr=float(np.subtract(*np.percentile(h, [75, 25]))),
                       igd_median=float(np.median([r["igd"] for r in rs])),
                       unsup=float(np.mean([r["unsup"] for r in rs])),
                       distinct=float(np.mean([r["n_distinct"] for r in rs])),
                       above_ceiling=float(np.mean(h > hv_sup)))
            if model == "ro":
                row["at_scale_bounds"] = float(np.mean([r["at_scale_bounds"] for r in rs]))
            if base and a != "MOEA/D-AWA":
                raw[a] = mannwhitneyu(h, base, alternative="two-sided").pvalue
                row["a12_vs_awa"] = a12(h, base)
            rows[a] = row
        if raw:
            for a, p in zip(raw, holm(np.array(list(raw.values())))):
                rows[a]["p_holm_vs_awa"] = float(p)
        return rows

    tables = {}
    main = groups.get("main", []) + groups.get("scal", [])
    tables["main"] = table(main)
    for tag, runs in groups.items():
        if tag not in ("main", "scal"):
            tables[tag] = table(runs)
    ws = tables["main"].get("MOEA/D-WS")
    out = dict(model=model, formulation=form, label=cfg["label"][form], n_reference=int(len(Fr)),
               unsupported=float(np.mean(gaps > GAP_TOL)), max_gap=float(gaps.max()),
               gap_quantiles={q: float(np.quantile(gaps, q)) for q in (0.25, 0.5, 0.75, 0.9)},
               unsupported_by_tol={f"{t:g}": float(np.mean(gaps > t)) for t in (1e-2, 1e-3, 1e-4, 1e-5)},
               hv_reference=hv_ref, hv_ceiling=hv_sup, tables=tables)
    if ws:
        out["ws_loss"] = dict(reachability=hv_ref - hv_sup, distribution=hv_sup - ws["hv_median"])
    if model == "ro":
        n_ref = X_ref[:, 0]
        out["reference_at_scale_bounds"] = float(np.mean((n_ref < pb.xl[0] + 0.5) | (n_ref > pb.xu[0] - 0.5)))
        out["supported_at_scale_bounds"] = float(np.mean(
            ((n_ref < pb.xl[0] + 0.5) | (n_ref > pb.xu[0] - 0.5))[gaps <= GAP_TOL])) if (gaps <= GAP_TOL).any() else None
        out["unsupported_interior_scale"] = float(np.mean(
            gaps[(n_ref >= pb.xl[0] + 0.5) & (n_ref <= pb.xu[0] - 0.5)] > GAP_TOL))
    np.save(RES / f"{cfg['prefix']}_{form}_gaps.npy", gaps)
    keys = ("algorithm", "seed", "hv", "igd", "unsup", "n_distinct", "at_scale_bounds")
    out["runs"] = {tag: [{k: r[k] for k in keys if k in r} for r in runs] for tag, runs in groups.items()}
    if model == "ro":   # scale of the final solutions of each MOEA/D-WS / MOEA/D run, for the figure
        out["scale_samples"] = {a: np.concatenate([np.array(r["X"])[:, 0] for r in main if r["algorithm"] == a]).tolist()
                                for a in ("MOEA/D-WS", "MOEA/D", "MOEA/D-AWA", "NSGA-II")
                                if any(r["algorithm"] == a for r in main)}
    return out, main


def markdown(results):
    lines = []
    for res in results:
        lines.append(f"## {res['model']}: {res['label']}\n")
        lines.append(f"Reference front: {res['n_reference']} points, HV = {res['hv_reference']:.4f}; "
                     f"unsupported {100 * res['unsupported']:.0f}% (max gap {res['max_gap']:.3f}); "
                     f"weighted-sum ceiling {res['hv_ceiling']:.4f}.")
        if "ws_loss" in res:
            lines.append(f"MOEA/D-WS loss: reachability {res['ws_loss']['reachability']:.4f} + "
                         f"distribution {res['ws_loss']['distribution']:.4f}.")
        lines.append("")
        for tag, rows in res["tables"].items():
            lines.append(f"### runs: {tag}\n")
            extra = " | at scale bounds" if res["model"] == "ro" else ""
            lines.append(f"| Algorithm | median HV | IQR | median IGD+ | unsupported region | distinct | runs above ceiling | A12 vs AWA | p (Holm){extra} |")
            lines.append("|---|---|---|---|---|---|---|---|---|" + ("---|" if extra else ""))
            for a, r in rows.items():
                s = (f"| {a} | {r['hv_median']:.4f} | {r['hv_iqr']:.4f} | {r['igd_median']:.4f} | "
                     f"{100 * r['unsup']:.0f}% | {r['distinct']:.0f} | {100 * r['above_ceiling']:.0f}% | "
                     f"{r.get('a12_vs_awa', float('nan')):.2f} | {r.get('p_holm_vs_awa', float('nan')):.1e} |")
                if extra:
                    s += f" {100 * r['at_scale_bounds']:.0f}% |"
                lines.append(s)
            lines.append("")
    return "\n".join(lines)


def figure(model, results, mains, path):
    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 3.8))
    for ax, res, runs in zip(np.atleast_1d(axes), results, mains):
        algs = [a for a in ORDER if any(r["algorithm"] == a for r in runs)]
        data = [[r["hv"] for r in runs if r["algorithm"] == a] for a in algs]
        bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.6)
        for a, b in zip(algs, bp["boxes"]):
            b.set_facecolor("#e0821e" if a in ("MOEA/D-WS",) else ("#f2c79a" if a.startswith("MOEA/D-L") else "#c9d6ea"))
        ax.set_yticks(range(1, len(algs) + 1), algs, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(res["hv_reference"], c="0.3", ls=":", lw=1, label="reference front")
        ax.axvline(res["hv_ceiling"], c="#e0821e", ls="--", lw=1.2, label="weighted-sum ceiling")
        ax.set_xlabel("HV (normalized, reference point 1.1)")
        ax.set_title(res["label"], fontsize=10)
        ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=list(SETTINGS), default="otto")
    args = ap.parse_args()
    cfg = SETTINGS[args.model]
    results, mains = [], []
    for form in cfg["forms"]:
        res, runs = analyze(args.model, form)
        results.append(res)
        mains.append(runs)
    (RES / f"{cfg['prefix']}_extended_runs.json").write_text(json.dumps(
        [{k: res.pop(k) for k in ("runs", "scale_samples") if k in res} | {"formulation": res["formulation"]}
         for res in results]))
    (RES / f"{cfg['prefix']}_extended.json").write_text(json.dumps(results, indent=1))
    (RES / f"{cfg['prefix']}_extended.md").write_text(markdown(results))
    figure(args.model, results, mains, FIG / f"fig18_{cfg['prefix']}_scalarizations.png")
    print(markdown(results))


if __name__ == "__main__":
    main()
