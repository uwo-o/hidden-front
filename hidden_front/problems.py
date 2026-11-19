"""Multi-objective engine problems (pymoo format, minimization).

Two engines, each in two formulations:
    objectives="eta":    f = [-P, -eta, dP]    (ratio formulation: efficiency)
    objectives="sigma":  f = [-P, dP, Sigma]   (extensive formulation: entropy production,
                                                as in Erdman et al. 2023)
"""
from __future__ import annotations

import numpy as np
from pymoo.core.problem import Problem

from .otto import OttoParams, cycle_statistics
from .piecewise import QDParams, cycle_stats

FORMULATIONS = ("eta", "sigma")


def _objectives(s, objectives):
    if objectives == "eta":
        return np.column_stack([-s["P"], -s["eta"], s["dP"]])
    return np.column_stack([-s["P"], s["dP"], s["Sigma"]])


class OttoProblem(Problem):
    """Single-qubit Otto engine with four decision variables.

    x = [omega_h, r, log10 tau_h, log10 tau_c] with omega_c = r omega_h.
    r in [T_c / T_h, 1] guarantees engine operation (P >= 0) and eta = 1 - r in [0, eta_C].
    """

    def __init__(self, params: OttoParams = OttoParams(), omega_bounds=(0.1, 10.0),
                 log_tau_bounds=(-2.0, 1.0), objectives="eta"):
        if objectives not in FORMULATIONS:
            raise ValueError("objectives must be 'eta' or 'sigma'")
        self.params = params
        self.objectives = objectives
        r_min = params.T_c / params.T_h
        xl = np.array([omega_bounds[0], r_min, log_tau_bounds[0], log_tau_bounds[0]])
        xu = np.array([omega_bounds[1], 1.0, log_tau_bounds[1], log_tau_bounds[1]])
        super().__init__(n_var=4, n_obj=3, xl=xl, xu=xu)

    def stats(self, X):
        X = np.atleast_2d(X)
        wh, r, lth, ltc = X.T
        return cycle_statistics(wh, r * wh, 10.0 ** lth, 10.0 ** ltc, self.params)

    def objectives_from_stats(self, s):
        return _objectives(s, self.objectives)

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = self.objectives_from_stats(self.stats(X))


def physical(F):
    """Convert objectives of the 'eta' formulation back to (P, eta, dP)."""
    F = np.atleast_2d(F)
    return -F[:, 0], -F[:, 1], F[:, 2]


class PiecewiseQDProblem(Problem):
    """Quantum-dot engine (Erdman et al. 2023) with N-step piecewise control.

    x = [u_1..u_N, v_1..v_N, log10 tau]; step k uses the hot bath if v_k > 0.5.
    """

    def __init__(self, n_steps=8, params=None, objectives="sigma", log_tau_bounds=(-1.5, 1.5)):
        if objectives not in FORMULATIONS:
            raise ValueError("objectives must be 'eta' or 'sigma'")
        self.qd = params or QDParams()
        self.N = n_steps
        self.objectives = objectives
        xl = np.r_[np.full(n_steps, self.qd.u_min), np.zeros(n_steps), log_tau_bounds[0]]
        xu = np.r_[np.full(n_steps, self.qd.u_max), np.ones(n_steps), log_tau_bounds[1]]
        super().__init__(n_var=2 * n_steps + 1, n_obj=3, xl=xl, xu=xu)

    def stats(self, X):
        X = np.atleast_2d(X)
        N = self.N
        return cycle_stats(X[:, :N], X[:, N:2 * N] > 0.5, 10.0 ** X[:, -1], self.qd)

    def objectives_from_stats(self, s):
        return _objectives(s, self.objectives)

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = self.objectives_from_stats(self.stats(X))


RO_FORMULATIONS = ("recovery", "brine")


class DesalinationProblem(Problem):
    """Reverse-osmosis plant sizing with four objectives (see hidden_front.desalination).

    x = [n, log10 q_f, p]: number of parallel pressure vessels (the free scale), feed flow per vessel
    and applied pressure (both in the dimensionless units of the model).

    objectives="recovery": f = [-Q, -r, W, n]   (ratio formulation)
    objectives="brine":    f = [-Q, B, W, n]    (extensive formulation: brine discharge)
    """

    def __init__(self, params=None, n_bounds=(1.0, 100.0), log_qf_bounds=(-1.5, 1.0),
                 p_bounds=(1.02, 3.0), objectives="recovery"):
        from .desalination import ROParams
        if objectives not in RO_FORMULATIONS:
            raise ValueError("objectives must be 'recovery' or 'brine'")
        self.params = params or ROParams()
        self.objectives = objectives
        xl = np.array([n_bounds[0], log_qf_bounds[0], p_bounds[0]])
        xu = np.array([n_bounds[1], log_qf_bounds[1], min(p_bounds[1], self.params.p_max)])
        super().__init__(n_var=3, n_obj=4, xl=xl, xu=xu)

    def stats(self, X):
        from .desalination import plant
        X = np.atleast_2d(X)
        return plant(X[:, 0], 10.0 ** X[:, 1], X[:, 2], self.params)

    def objectives_from_stats(self, s):
        if self.objectives == "recovery":
            return np.column_stack([-s["Q"], -s["r"], s["W"], s["n"]])
        return np.column_stack([-s["Q"], s["B"], s["W"], s["n"]])

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = self.objectives_from_stats(self.stats(X))
