"""Supported points of a Pareto front (reachable by a weighted sum)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def support_gap(points, front):
    """Support gap of each point p (minimization, normalized objective space).

    Solves, for every p, the linear program

        s*(p) = min_{w >= 0, sum w = 1}  max_{q in front U {p}}  w . (p - q)

    s* = 0  <=>  some weighted sum w . f is minimized by p over the front
                 (p is *supported*: reachable by linear scalarization).
    s* > 0  <=>  no weighted sum reaches p; s* measures how far it is from being
                 reachable, in normalized units.
    """
    points, front = np.atleast_2d(points), np.atleast_2d(front)
    m = front.shape[1]
    Q = np.vstack([front])
    gaps = np.empty(len(points))
    c = np.r_[np.zeros(m), 1.0]
    A_eq = np.r_[np.ones(m), 0.0][None]
    bounds = [(0, None)] * m + [(None, None)]
    for k, p in enumerate(points):
        Qk = np.vstack([Q, p])
        A = np.hstack([p - Qk, -np.ones((len(Qk), 1))])
        r = linprog(c, A_ub=A, b_ub=np.zeros(len(Qk)), A_eq=A_eq, b_eq=[1.0],
                    bounds=bounds, method="highs")
        gaps[k] = max(r.fun, 0.0)
    return gaps
