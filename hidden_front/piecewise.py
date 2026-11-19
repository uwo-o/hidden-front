"""Quantum-dot engine of Erdman et al. (PRR 5, L022017, 2023) with piecewise-constant control.

H(t) = u(t) E0 sigma_z / 2  ->  gap eps(t) = E0 u(t). Excited occupation p:
    dp/dt = -gamma (p - f(beta(t) eps(t))),   f(x) = 1 / (1 + e^x)
The bath alternates between beta_H and beta_C (a binary choice per step, as in
the original paper).

A cycle of duration tau is split into N equal steps of length dt = tau / N. In
step k the gap is eps_k and the bath is b_k in {H, C}. Between steps the gap
changes instantaneously (no gap-change time, as in Erdman et al.), extracting
the work w = -(eps_k - eps_{k-1}) s, with s in {0, 1} the state of the system
(the ground-level part, +-eps/2, cancels over a cycle).

Exact statistics follow from the tilted matrix of the cycle
    T(chi) = prod_k D_k(chi) M_k,   D_k = diag(1, e^{-chi d_eps_k}),
with M_k the relaxation matrix of step k. det T = prod x_k (because
sum_k d_eps_k = 0), so lambda_max has a closed form and theta'(0), theta''(0)
follow from analytic derivatives of the trace (propagating T, T', T'' with the
product rule).

Physical objectives:  P = theta'(0) / tau,  dP = theta''(0) / tau,
    Sigma = -(1 / tau) sum_k beta_k Q_k   (Q_k = heat absorbed in step k),
    eta = W / Q_H   (Q_H = heat absorbed from the hot bath).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit


@dataclass(frozen=True)
class QDParams:
    beta_h: float = 1.0      # beta_H  (T_h = 1)
    beta_c: float = 2.0      # beta_C  (T_c = 0.5) -> Carnot efficiency 0.5
    E0: float = 2.5
    u_min: float = 0.2
    u_max: float = 1.1
    gamma: float = 1.0

    @property
    def eta_carnot(self) -> float:
        return 1.0 - self.beta_h / self.beta_c


def _mm(A, B):
    return np.einsum("...ij,...jk->...ik", A, B)


def cycle_stats(u, hot, tau, params: QDParams = QDParams()):
    """u: (B, N) control levels u_k; hot: (B, N) booleans (True = hot bath);
    tau: (B,) cycle durations. Returns a dict of (B,) arrays."""
    u = np.atleast_2d(u).astype(float)
    hot = np.atleast_2d(hot).astype(bool)
    tau = np.atleast_1d(tau).astype(float)
    Bn, N = u.shape
    p = params
    eps = p.E0 * u                                        # (B, N)
    beta = np.where(hot, p.beta_h, p.beta_c)
    a = expit(-beta * eps)                                # f(beta eps)
    x = np.exp(-p.gamma * tau / N)[:, None] * np.ones((1, N))
    deps = eps - np.roll(eps, 1, axis=1)                  # d_eps_k (gap change before step k)
    c = -deps                                             # w = c_k * s

    # Relaxation matrix of each step (row-stochastic)
    M = np.empty((Bn, N, 2, 2))
    M[..., 0, 1] = a * (1 - x)
    M[..., 0, 0] = 1 - M[..., 0, 1]
    M[..., 1, 0] = (1 - a) * (1 - x)
    M[..., 1, 1] = 1 - M[..., 1, 0]
    # D_k M_k: the gap change acts on the arriving state, before relaxation
    Z = np.zeros((Bn, N, 2, 2))
    D0 = Z.copy(); D0[..., 0, 0] = 1; D0[..., 1, 1] = 1
    D1 = Z.copy(); D1[..., 1, 1] = c
    D2 = Z.copy(); D2[..., 1, 1] = c ** 2
    A0, A1, A2 = _mm(D0, M), _mm(D1, M), _mm(D2, M)
    T0 = np.broadcast_to(np.eye(2), (Bn, 2, 2)).copy()
    T1 = np.zeros((Bn, 2, 2)); T2 = np.zeros((Bn, 2, 2))
    for k in range(N):
        T0, T1, T2 = (_mm(T0, A0[:, k]),
                      _mm(T1, A0[:, k]) + _mm(T0, A1[:, k]),
                      _mm(T2, A0[:, k]) + 2 * _mm(T1, A1[:, k]) + _mm(T0, A2[:, k]))
    D = np.prod(x, axis=1)
    t1 = np.trace(T1, axis1=1, axis2=2)
    t2 = np.trace(T2, axis1=1, axis2=2)
    omD = 1 - D
    lam1 = t1 / omD
    lam2 = t2 / omD - 2 * D * t1 ** 2 / omD ** 3
    W = lam1
    varw = lam2 - lam1 ** 2

    # Periodic state (at the start of step 0) and heat per step
    q = T0[:, 0, 1] / (T0[:, 0, 1] + T0[:, 1, 0] + 1e-300)   # stationary excited probability
    Q = np.empty((Bn, N))
    pk = q.copy()
    for k in range(N):
        pe = a[:, k] + (pk - a[:, k]) * x[:, k]
        Q[:, k] = eps[:, k] * (pe - pk)
        pk = pe
    Qh = np.where(hot, Q, 0.0).sum(1)
    Sigma = -(beta * Q).sum(1) / tau
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where((W > 0) & (Qh > 0), W / Qh, 0.0)
    return dict(P=W / tau, dP=varw / tau, Sigma=Sigma, eta=eta, W=W, Qsum=Q.sum(1),
                Qh=Qh, var_w=varw)
