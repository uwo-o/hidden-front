"""Refine the reference fronts of the piecewise experiment with long SMS-EMOA runs.

Five runs of 3000 generations per formulation, used only to build the reference front
(they do not enter the algorithm statistics). They help separate genuine nonconvexity
from convergence noise.

Usage:
    python scripts/refine_reference.py [--jobs 2]
Writes results/pw_{sigma,eta}_long.json.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

from run_experiment import run

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    for form in ["sigma", "eta"]:
        with Pool(args.jobs) as pool:
            out = pool.map(run, [("SMS-EMOA", form, 100 + s, 3000, 8) for s in range(5)])
        (ROOT / "results" / f"pw_{form}_long.json").write_text(json.dumps(dict(runs=out)))
        print(form, [len(r["F"]) for r in out], [round(r["seconds"]) for r in out], flush=True)


if __name__ == "__main__":
    main()
