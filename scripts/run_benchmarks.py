"""Validation of the MOEA/D implementation on DTLZ2 (regular front) and DTLZ7 (disconnected front).

All variants use normalized Tchebycheff scalarization:
    MOEA/D                 uniform weights
    MOEA/D-WT              WS-transformed weights (Qi et al. 2014)
    MOEA/D-AWA (wag=20)    adaptive weight adjustment every 20 generations
    MOEA/D-AWA (wag=10)    adaptive weight adjustment every 10 generations

Usage:
    python scripts/run_benchmarks.py [--seeds 10] [--gens 300] [--jobs 2]
Writes results/benchmarks.json.
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions

from hidden_front.moead import MOEAD

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "MOEA/D": dict(ws=False, awa=False),
    "MOEA/D-WT": dict(ws=True, awa=False),
    "MOEA/D-AWA (wag=20)": dict(ws=True, awa=True, wag=20),
    "MOEA/D-AWA (wag=10)": dict(ws=True, awa=True, wag=10),
}
PROBLEMS = {"DTLZ2": dict(name="dtlz2", n_var=12), "DTLZ7": dict(name="dtlz7", n_var=22)}


def make_problem(key):
    cfg = PROBLEMS[key]
    pb = get_problem(cfg["name"], n_var=cfg["n_var"], n_obj=3)
    if key == "DTLZ2":
        pf = pb.pareto_front(get_reference_directions("das-dennis", 3, n_partitions=60))
    else:
        pf = pb.pareto_front()
    return pb, pf


def job(args):
    key, variant, seed, gens = args
    pb, pf = make_problem(key)
    t0 = time.time()
    alg = MOEAD(pb, n_partitions=13, max_gen=gens, seed=seed, **VARIANTS[variant]).run()
    F = alg.result_F
    return dict(problem=key, variant=variant, seed=seed, igd_plus=float(IGDPlus(pf)(F)),
                n_front=int(len(F)), n_eval=alg.n_eval, seconds=time.time() - t0,
                F=F.tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    tasks = [(p, v, s, args.gens) for p in PROBLEMS for v in VARIANTS for s in range(args.seeds)]
    with Pool(args.jobs) as pool:
        out = []
        for i, r in enumerate(pool.imap_unordered(job, tasks)):
            out.append(r)
            print(f"[{i + 1}/{len(tasks)}] {r['problem']:6s} {r['variant']:22s} seed={r['seed']} "
                  f"IGD+={r['igd_plus']:.4f} ({r['seconds']:.0f}s)", flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "benchmarks.json").write_text(json.dumps(out))

    print("\nIGD+ summary (mean +- std)")
    for p in PROBLEMS:
        for v in VARIANTS:
            vals = [r["igd_plus"] for r in out if r["problem"] == p and r["variant"] == v]
            print(f"  {p:6s} {v:22s} {np.mean(vals):.4f} +- {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
