"""Dense reference fronts of the reverse-osmosis sizing problem in both formulations.

The decision space x = [n, log10 q_f, p] is sampled with a scrambled Sobol sequence; the
nondominated set is found in chunks, merged, and thinned on a voxel grid of the normalized
four-objective space. The support test is then applied to the thinned front.

Usage:
    python scripts/ro_reference.py [--samples 131072] [--bins 12]
Writes results/ro_reference.npz with X, F and support gaps for each formulation.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from hidden_front.moead import nondominated_mask
from hidden_front.problems import DesalinationProblem
from hidden_front.support import support_gap

ROOT = Path(__file__).resolve().parents[1]


def nondominated_chunked(F, chunk=16384):
    """Indices of the nondominated rows of F, computed chunk by chunk and merged."""
    keep = []
    for i in range(0, len(F), chunk):
        idx = np.arange(i, min(i + chunk, len(F)))
        keep.append(idx[nondominated_mask(F[idx])])
    idx = np.concatenate(keep)
    return idx[nondominated_mask(F[idx])]


def thin(G, bins):
    """One point per voxel of a normalized front (any number of objectives)."""
    v = np.floor(G * bins).clip(0, bins - 1).astype(np.int64)
    key = np.zeros(len(G), dtype=np.int64)
    for j in range(G.shape[1]):
        key = key * bins + v[:, j]
    _, u = np.unique(key, return_index=True)
    return np.sort(u)


def build(form, samples, bins, seed=0, extra_X=None):
    pb = DesalinationProblem(objectives=form)
    X = qmc.Sobol(3, scramble=True, seed=seed).random(samples)
    X = pb.xl + X * (pb.xu - pb.xl)
    if extra_X is not None:
        X = np.vstack([X, extra_X])
    F = pb.evaluate(X, return_values_of=["F"])
    nd = nondominated_chunked(F)
    X, F = X[nd], F[nd]
    ideal, nadir = F.min(0), F.max(0)
    G = (F - ideal) / (nadir - ideal)
    u = thin(G, bins)
    return X[u], F[u], G[u], ideal, nadir, len(nd)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", type=int, default=2 ** 17)
    ap.add_argument("--bins", type=int, default=12)
    args = ap.parse_args()
    out = {}
    for form in ("recovery", "brine"):
        t0 = time.time()
        X, F, G, ideal, nadir, n_nd = build(form, args.samples, args.bins)
        print(f"[{form}] {n_nd} nondominated -> {len(G)} after thinning; support LP...", flush=True)
        gaps = support_gap(G, G)
        print(f"[{form}] unsupported {np.mean(gaps > 1e-3):.3f}, max gap {gaps.max():.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        out.update({f"{form}_X": X, f"{form}_F": F, f"{form}_gaps": gaps,
                    f"{form}_ideal": ideal, f"{form}_nadir": nadir})
    np.savez_compressed(ROOT / "results" / "ro_reference.npz", **out)


if __name__ == "__main__":
    main()
