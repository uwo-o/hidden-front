"""Main experiment: seven MOEAs on an engine model in both formulations.

Models:
    piecewise  quantum-dot engine of Erdman et al. (2023), N = 8 control steps (17 variables)
    otto       single-qubit Otto engine (4 variables)

Algorithms (same budget, population N = 105):
    NSGA-II, NSGA-III, SMS-EMOA, RVEA (pymoo); MOEA/D and MOEA/D-AWA (normalized
    Tchebycheff); and MOEA/D-WS, MOEA/D with a linear weighted sum, the evolutionary
    counterpart of sweeping the weights of a linear figure of merit.

Usage (settings of the paper):
    python scripts/run_experiment.py --model piecewise --seeds 30 --gens 3000 --jobs 8
    python scripts/run_experiment.py --model otto      --seeds 30 --gens 400  --jobs 8
Writes results/pw_{sigma,eta}_runs.json (piecewise) or results/ot_{sigma,eta}_runs.json (otto).
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
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from hidden_front.moead import MOEAD, nondominated_mask
from hidden_front.problems import OttoProblem, PiecewiseQDProblem

ROOT = Path(__file__).resolve().parents[1]
ALGS = ["NSGA-II", "NSGA-III", "SMS-EMOA", "RVEA", "MOEA/D", "MOEA/D-AWA", "MOEA/D-WS"]
MOEAD_VARIANTS = {"MOEA/D": dict(ws=False, awa=False),
                  "MOEA/D-AWA": dict(ws=True, awa=True, wag=10),
                  "MOEA/D-WS": dict(scalarization="ws")}
H = 13                                    # Das-Dennis divisions -> 105 weight vectors
PREFIX = {"piecewise": "pw", "otto": "ot"}


def make_problem(model, form, steps=8):
    """model='piecewise': piecewise quantum dot (17 variables); model='otto': single-qubit Otto (4)."""
    if model == "otto":
        return OttoProblem(objectives=form)
    return PiecewiseQDProblem(n_steps=steps, objectives=form)


def run(args):
    """Run one algorithm once. args = (algorithm, formulation, seed, generations, steps[, model])."""
    alg, form, seed, gens, steps = args[:5]
    model = args[5] if len(args) > 5 else "piecewise"
    pb = make_problem(model, form, steps)
    t0 = time.time()
    if alg.startswith("MOEA/D"):
        a = MOEAD(pb, n_partitions=H, max_gen=gens, seed=seed, **MOEAD_VARIANTS[alg]).run()
        X, F, n_eval = a.X, a.F, a.n_eval
    else:
        ref = get_reference_directions("das-dennis", 3, n_partitions=H)
        ops = dict(crossover=SBX(prob=1.0, eta=20), mutation=PM(eta=20))
        algo = {"NSGA-II": lambda: NSGA2(pop_size=len(ref), **ops),
                "NSGA-III": lambda: NSGA3(ref_dirs=ref, pop_size=len(ref), **ops),
                "SMS-EMOA": lambda: SMSEMOA(pop_size=len(ref), **ops),
                "RVEA": lambda: RVEA(ref_dirs=ref, pop_size=len(ref), **ops)}[alg]()
        res = minimize(pb, algo, ("n_gen", gens), seed=seed, verbose=False)
        X, F = res.pop.get("X"), res.pop.get("F")
        n_eval = res.algorithm.evaluator.n_eval
    nd = nondominated_mask(F)
    X, F = X[nd], F[nd]
    _, u = np.unique(np.round(F, 14), axis=0, return_index=True)
    return dict(algorithm=alg, objectives=form, seed=seed, X=X[u].tolist(), F=F[u].tolist(),
                n_eval=int(n_eval), seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--model", choices=["piecewise", "otto"], default="piecewise")
    ap.add_argument("--forms", default="sigma,eta", help="comma-separated formulations to run")
    args = ap.parse_args()
    (ROOT / "results").mkdir(exist_ok=True)
    for form in args.forms.split(","):
        tasks = [(a, form, s, args.gens, args.steps, args.model) for s in range(args.seeds) for a in ALGS]
        out = []
        with Pool(args.jobs) as pool:
            for i, r in enumerate(pool.imap_unordered(run, tasks)):
                out.append(r)
                print(f"[{form} {i + 1}/{len(tasks)}] {r['algorithm']:13s} seed={r['seed']} "
                      f"|front|={len(r['F'])} evals={r['n_eval']} ({r['seconds']:.0f}s)", flush=True)
        (ROOT / "results" / f"{PREFIX[args.model]}_{form}_runs.json").write_text(json.dumps(dict(
            runs=out, steps=args.steps, gens=args.gens, seeds=args.seeds, model=args.model)))


if __name__ == "__main__":
    main()
