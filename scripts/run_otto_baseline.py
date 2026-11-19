"""Baseline experiment: Pareto front of the single-qubit Otto engine (10 seeds).

1. Reference front from a dense sweep of the four decision variables.
2. NSGA-II, NSGA-III, MOEA/D, MOEA/D-WT (WS-transformed weights) and MOEA/D-AWA,
   several seeds, same population size and number of evaluations.
3. Linear-scalarization baseline: one multi-start L-BFGS-B weighted-sum optimization
   per weight vector, the analogue of the figure of merit of Erdman et al. (2023).
4. HV and IGD+ in the objective space normalized by the reference front.

The dense reference front written here (results/otto[_sigma]_reference.npz) is also
used by analyze_experiment.py --model otto and physical_cost.py.

Usage:
    python scripts/run_otto_baseline.py [--seeds 10] [--gens 400] [--jobs 2] [--objectives eta|sigma]
Writes results/otto[_sigma]_runs.json and results/otto[_sigma]_reference.npz.
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from scipy.optimize import minimize as sp_minimize

from hidden_front.moead import MOEAD, nondominated_mask
from hidden_front.otto import OttoParams
from hidden_front.problems import OttoProblem

ROOT = Path(__file__).resolve().parents[1]
N_PARTITIONS = 13          # 105 subproblems / individuals
ALGORITHMS = ["NSGA-II", "NSGA-III", "MOEA/D", "MOEA/D-WT", "MOEA/D-AWA"]
MOEAD_VARIANTS = {"MOEA/D": dict(ws=False, awa=False), "MOEA/D-WT": dict(ws=True, awa=False),
                  "MOEA/D-AWA": dict(ws=True, awa=True, wag=10)}


def reference_front(problem: OttoProblem, n_omega=90, n_r=61, n_tau=45):
    """Dense sweep. Since eta depends on r only, each slice r = const is first
    filtered in 2D (max P, min dP) and the union is then filtered in 3D.

    In the Sigma formulation that pre-filter is invalid (at fixed r, Sigma is
    proportional to P, so almost nothing is dominated within a slice), so the
    full grid is filtered instead."""
    if problem.objectives == "sigma":
        return reference_front_full(problem)
    xl, xu = problem.xl, problem.xu
    omegas = np.geomspace(xl[0], xu[0], n_omega)
    rs = np.linspace(xl[1], xu[1], n_r)[:-1]             # r = 1 -> P = 0
    taus = np.linspace(xl[2], xu[2], n_tau)
    Wg, Th, Tc = np.meshgrid(omegas, taus, taus, indexing="ij")
    Wg, Th, Tc = Wg.ravel(), Th.ravel(), Tc.ravel()
    keep_X, keep_F = [], []
    for r in rs:
        X = np.column_stack([Wg, np.full_like(Wg, r), Th, Tc])
        s = problem.stats(X)
        P, dP = s["P"], s["dP"]
        order = np.lexsort((dP, -P))                     # P descending, dP ascending
        d = dP[order]
        prev_min = np.concatenate([[np.inf], np.minimum.accumulate(d)[:-1]])
        sel = order[d < prev_min]
        keep_X.append(X[sel])
        keep_F.append(np.column_stack([-P[sel], -s["eta"][sel], dP[sel]]))
    X, F = np.vstack(keep_X), np.vstack(keep_F)
    nd = nondominated_mask(F)
    return X[nd], F[nd]


def reference_front_full(problem, n_omega=60, n_r=49, n_tau=34):
    xl, xu = problem.xl, problem.xu
    G = np.meshgrid(np.geomspace(xl[0], xu[0], n_omega), np.linspace(xl[1], xu[1], n_r)[:-1],
                    np.linspace(xl[2], xu[2], n_tau), np.linspace(xl[3], xu[3], n_tau), indexing="ij")
    X = np.column_stack([g.ravel() for g in G])
    F = problem.evaluate(X)
    nd = nondominated_mask(F)
    return X[nd], F[nd]


def thin(X, F, ideal, nadir, bins=80):
    """Keep one point per voxel of the normalized objective space. The sweep is
    very unevenly dense (many points with P ~ 0); thinning leaves an approximately
    uniform sample of the front for IGD+ and the analysis."""
    Fn = (F - ideal) / (nadir - ideal)
    v = np.floor(Fn * bins).clip(0, bins - 1).astype(np.int64)
    key = (v[:, 0] * bins + v[:, 1]) * bins + v[:, 2]
    _, u = np.unique(key, return_index=True)
    return X[u], F[u]


def run_algorithm(name, seed, gens, objectives="eta"):
    problem = OttoProblem(objectives=objectives)
    t0 = time.time()
    if name.startswith("MOEA/D"):
        alg = MOEAD(problem, n_partitions=N_PARTITIONS, max_gen=gens, seed=seed,
                    **MOEAD_VARIANTS[name]).run()
        nd = nondominated_mask(alg.F)
        X, F, n_eval = alg.X[nd], alg.F[nd], alg.n_eval
    else:
        ref = get_reference_directions("das-dennis", 3, n_partitions=N_PARTITIONS)
        ops = dict(crossover=SBX(prob=1.0, eta=20), mutation=PM(eta=20))
        algo = NSGA2(pop_size=len(ref), **ops) if name == "NSGA-II" else \
            NSGA3(ref_dirs=ref, pop_size=len(ref), **ops)
        res = minimize(problem, algo, ("n_gen", gens), seed=seed, verbose=False)
        pop_F, pop_X = res.pop.get("F"), res.pop.get("X")
        nd = nondominated_mask(pop_F)
        X, F = pop_X[nd], pop_F[nd]
        n_eval = res.algorithm.evaluator.n_eval
    _, u = np.unique(np.round(F, 14), axis=0, return_index=True)
    X, F = X[u], F[u]
    return dict(algorithm=name, seed=seed, X=np.asarray(X).tolist(), F=np.asarray(F).tolist(),
                n_eval=int(n_eval), seconds=time.time() - t0)


def _job(args):
    return run_algorithm(*args)


def weighted_sum_baseline(problem, ideal, nadir, n_partitions=N_PARTITIONS, n_starts=8, seed=0):
    """For each weight vector w (uniform simplex grid) minimize sum_i w_i f_hat_i
    with multi-start L-BFGS-B. Returns the weights, solutions and objectives."""
    rng = np.random.default_rng(seed)
    W = get_reference_directions("das-dennis", 3, n_partitions=n_partitions)
    scale = nadir - ideal
    bounds = list(zip(problem.xl, problem.xu))
    sols = []
    for w in W:
        def f(x, w=w):
            return float(w @ ((problem.evaluate(x[None])[0] - ideal) / scale))
        best = None
        for _ in range(n_starts):
            x0 = problem.xl + rng.random(4) * (problem.xu - problem.xl)
            r = sp_minimize(f, x0, method="L-BFGS-B", bounds=bounds)
            if best is None or r.fun < best.fun:
                best = r
        sols.append(best.x)
    X = np.array(sols)
    return W, X, problem.evaluate(X)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--gens", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--objectives", choices=["eta", "sigma"], default="eta")
    args = ap.parse_args()
    tag = "otto" if args.objectives == "eta" else "otto_sigma"
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    problem = OttoProblem(objectives=args.objectives)
    print(f"Formulation: {args.objectives}", flush=True)

    print("Reference front (dense sweep)...", flush=True)
    t0 = time.time()
    Xg, Fg = reference_front(problem)
    print(f"  {len(Fg)} nondominated points ({time.time() - t0:.0f}s)", flush=True)

    tasks = [(a, s, args.gens, args.objectives) for a in ALGORITHMS for s in range(args.seeds)]
    runs = []
    with Pool(args.jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, tasks)):
            runs.append(r)
            print(f"[{i + 1}/{len(tasks)}] {r['algorithm']:11s} seed={r['seed']} "
                  f"|front|={len(r['F'])} evals={r['n_eval']} ({r['seconds']:.0f}s)", flush=True)
    (out_dir / f"{tag}_runs_raw.json").write_text(json.dumps(runs))

    # Final reference front = ND(sweep U all runs)
    X_all = np.vstack([Xg] + [np.array(r["X"]) for r in runs])
    F_all = np.vstack([Fg] + [np.array(r["F"]) for r in runs])
    nd = nondominated_mask(F_all)
    X_ref, F_ref = X_all[nd], F_all[nd]
    ideal, nadir = F_ref.min(0), F_ref.max(0)
    n_full = len(F_ref)
    X_ref, F_ref = thin(X_ref, F_ref, ideal, nadir)
    print(f"Final reference front: {n_full} nondominated points -> {len(F_ref)} after voxel thinning",
          flush=True)

    print("Weighted-sum baseline...", flush=True)
    W_ws, X_ws, F_ws = weighted_sum_baseline(problem, ideal, nadir)

    def norm(F):
        return (np.asarray(F) - ideal) / (nadir - ideal)

    hv = HV(ref_point=np.full(3, 1.1))
    igd = IGDPlus(norm(F_ref))
    for r in runs:
        r["hv"] = float(hv(norm(r["F"])))
        r["igd_plus"] = float(igd(norm(r["F"])))
    ws_nd = nondominated_mask(F_ws)
    ws_metrics = dict(hv=float(hv(norm(F_ws[ws_nd]))), igd_plus=float(igd(norm(F_ws[ws_nd]))),
                      n_unique=int(len(np.unique(np.round(norm(F_ws), 4), axis=0))))
    ref_metrics = dict(hv=float(hv(norm(F_ref))))

    (out_dir / f"{tag}_runs.json").write_text(json.dumps(dict(
        objectives=args.objectives, runs=runs, weighted_sum=ws_metrics, reference=ref_metrics,
        params=OttoParams().__dict__, gens=args.gens, seeds=args.seeds)))
    np.savez(out_dir / f"{tag}_reference.npz", X_ref=X_ref, F_ref=F_ref, ideal=ideal,
             nadir=nadir, X_ws=X_ws, F_ws=F_ws, W_ws=W_ws)

    print("\nSummary (normalized by the reference front; HV reference point = 1.1)")
    print(f"  {'Reference':12s} HV={ref_metrics['hv']:.4f}")
    for a in ALGORITHMS:
        h = [r["hv"] for r in runs if r["algorithm"] == a]
        g = [r["igd_plus"] for r in runs if r["algorithm"] == a]
        print(f"  {a:12s} HV={np.mean(h):.4f}+-{np.std(h):.4f}  IGD+={np.mean(g):.4f}+-{np.std(g):.4f}")
    print(f"  {'Weighted sum':12s} HV={ws_metrics['hv']:.4f}  IGD+={ws_metrics['igd_plus']:.4f}  "
          f"distinct solutions={ws_metrics['n_unique']}/{len(W_ws)}")


if __name__ == "__main__":
    main()
