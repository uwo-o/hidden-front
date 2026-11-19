"""Main experiment: seven MOEAs on an engine model in both formulations.

Models:
    piecewise  quantum-dot engine of Erdman et al. (2023), N = 8 control steps (17 variables)
    otto       single-qubit Otto engine (4 variables)
    ro         reverse-osmosis desalination plant sizing (3 variables, 4 objectives)

Algorithms (same budget, population N = 105):
    NSGA-II, NSGA-III, SMS-EMOA, RVEA (pymoo); MOEA/D and MOEA/D-AWA (normalized
    Tchebycheff); and MOEA/D-WS, MOEA/D with a linear weighted sum, the evolutionary
    counterpart of sweeping the weights of a linear figure of merit.

Further decomposition baselines (--algs): MOEA/D-PBI, MOEA/D-LWS, MOEA/D-L2, MOEA/D-L4.

Usage (settings of the paper):
    python scripts/run_experiment.py --model piecewise --seeds 30 --gens 3000 --jobs 8
    python scripts/run_experiment.py --model otto      --seeds 30 --gens 400  --jobs 8
    python scripts/run_experiment.py --model otto --algs MOEA/D-PBI,MOEA/D-LWS,MOEA/D-L2,MOEA/D-L4 --tag scal
    python scripts/run_experiment.py --model ro --seeds 30 --gens 400 --jobs 8
Writes results/{pw,ot,ro}_{form}[_{tag}]_runs.json.
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
EXTRA_ALGS = ["MOEA/D-PBI", "MOEA/D-LWS", "MOEA/D-L2", "MOEA/D-L4"]
MOEAD_VARIANTS = {"MOEA/D": dict(ws=False, awa=False),
                  "MOEA/D-AWA": dict(ws=True, awa=True, wag=10),
                  "MOEA/D-WS": dict(scalarization="ws"),
                  "MOEA/D-PBI": dict(scalarization="pbi", theta=5.0),
                  "MOEA/D-LWS": dict(scalarization="lws"),
                  "MOEA/D-LWS0.5x": dict(scalarization="lws", cone_factor=0.5),
                  "MOEA/D-LWS0.25x": dict(scalarization="lws", cone_factor=0.25),
                  "MOEA/D-LWS2x": dict(scalarization="lws", cone_factor=2.0),
                  "MOEA/D-LWS4x": dict(scalarization="lws", cone_factor=4.0),
                  "MOEA/D-L2": dict(scalarization="lp", p=2.0),
                  "MOEA/D-L4": dict(scalarization="lp", p=4.0)}
PARTITIONS = {3: 13, 4: 7}                # Das-Dennis divisions -> 105 (m = 3) or 120 (m = 4) weights
PREFIX = {"piecewise": "pw", "otto": "ot", "ro": "ro"}
FORMS = {"piecewise": "sigma,eta", "otto": "sigma,eta", "ro": "brine,recovery"}


def make_problem(model, form, steps=8):
    """Problem instance for a model and a formulation."""
    if model == "otto":
        return OttoProblem(objectives=form)
    if model == "ro":
        from hidden_front.problems import DesalinationProblem
        return DesalinationProblem(objectives=form)
    return PiecewiseQDProblem(n_steps=steps, objectives=form)


def run(args):
    """Run one algorithm once. args = (algorithm, formulation, seed, generations, steps[, model])."""
    alg, form, seed, gens, steps = args[:5]
    model = args[5] if len(args) > 5 else "piecewise"
    scale = args[6] if len(args) > 6 else 1
    pb = make_problem(model, form, steps)
    H = PARTITIONS[pb.n_obj] * scale if scale > 0 else -scale   # negative value: explicit divisions
    t0 = time.time()
    if alg.startswith("MOEA/D"):
        a = MOEAD(pb, n_partitions=H, max_gen=gens, seed=seed, **MOEAD_VARIANTS[alg]).run()
        X, F, n_eval = a.X, a.F, a.n_eval
    else:
        ref = get_reference_directions("das-dennis", pb.n_obj, n_partitions=H)
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
    ap.add_argument("--model", choices=["piecewise", "otto", "ro"], default="piecewise")
    ap.add_argument("--forms", default=None, help="comma-separated formulations (default: both)")
    ap.add_argument("--algs", default=",".join(ALGS), help="comma-separated algorithm labels")
    ap.add_argument("--pop-scale", type=int, default=1,
                    help="multiply the Das-Dennis divisions (1: N = 105 for m = 3; 2: N = 406)")
    ap.add_argument("--partitions", type=int, default=None,
                    help="explicit number of Das-Dennis divisions (overrides --pop-scale; 20 -> N = 231)")
    ap.add_argument("--tag", default="", help="suffix of the output file")
    args = ap.parse_args()
    (ROOT / "results").mkdir(exist_ok=True)
    algs = args.algs.split(",")
    tag = f"_{args.tag}" if args.tag else ""
    for form in (args.forms or FORMS[args.model]).split(","):
        scale = -args.partitions if args.partitions else args.pop_scale
        tasks = [(a, form, s, args.gens, args.steps, args.model, scale)
                 for s in range(args.seeds) for a in algs]
        out = []
        with Pool(args.jobs) as pool:
            for i, r in enumerate(pool.imap_unordered(run, tasks)):
                out.append(r)
                print(f"[{form} {i + 1}/{len(tasks)}] {r['algorithm']:13s} seed={r['seed']} "
                      f"|front|={len(r['F'])} evals={r['n_eval']} ({r['seconds']:.0f}s)", flush=True)
        (ROOT / "results" / f"{PREFIX[args.model]}_{form}{tag}_runs.json").write_text(json.dumps(dict(
            runs=out, steps=args.steps, gens=args.gens, seeds=args.seeds, model=args.model,
            pop_scale=args.pop_scale)))


if __name__ == "__main__":
    main()
