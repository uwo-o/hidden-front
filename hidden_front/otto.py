"""Finite-time quantum Otto engine with a single two-level system (qubit).

Units: hbar = k_B = 1. Relaxation rates are given in units of gamma.

Cycle (qubit state s in {0, 1}, energy gap omega):
  1. Contact with the hot bath T_h at gap omega_h for a time tau_h. The excited
     population relaxes exponentially towards equilibrium:
         p -> a_h + (p - a_h) x_h,   a = 1 / (1 + exp(omega / T)),   x = exp(-gamma tau)
  2. Gap change omega_h -> omega_c at fixed population (duration tau_q).
     Extracted work = (omega_h - omega_c) * s1.
  3. Contact with the cold bath T_c at gap omega_c for a time tau_c.
  4. Gap change omega_c -> omega_h (duration tau_q).
     Invested work = (omega_h - omega_c) * s2.

Because H(t) commutes with itself at all times (only the gap changes), no
coherences are generated and the qubit is a classical two-state Markov chain.

Stochastic work extracted per cycle:  w = d_omega * (s1 - s2).

Full counting statistics use the tilted transfer matrix of one cycle

    T(chi)[s, s'] = sum_{s1} M_h[s, s1] e^{chi d_omega s1} M_c[s1, s'] e^{-chi d_omega s'}

and theta(chi) = ln lambda_max(T(chi)) is the scaled cumulant generating
function (per cycle):
    <w>       = theta'(0)
    var rate  = theta''(0)   (asymptotic variance per cycle, correlations included)

Since det T(chi) = x_h x_c does not depend on chi, lambda_max has a closed form
and its derivatives at chi = 0 are analytic (see `cycle_statistics`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit


@dataclass(frozen=True)
class OttoParams:
    T_h: float = 1.0       # hot bath temperature
    T_c: float = 0.5       # cold bath temperature -> Carnot efficiency 0.5
    gamma_h: float = 1.0   # relaxation rate with the hot bath
    gamma_c: float = 1.0   # relaxation rate with the cold bath
    tau_q: float = 0.05    # duration of each gap change (two per cycle)

    @property
    def eta_carnot(self) -> float:
        return 1.0 - self.T_c / self.T_h

    @property
    def eta_ca(self) -> float:
        """Curzon-Ahlborn efficiency."""
        return 1.0 - np.sqrt(self.T_c / self.T_h)


def excited_population(omega, T):
    """Equilibrium population of the excited level: 1 / (1 + e^{omega / T})."""
    return expit(-np.asarray(omega, dtype=float) / T)


def relaxation_matrix(a, x):
    """Row-stochastic transition matrix M[s, s'] of a qubit relaxing towards the
    excited population `a` with factor x = e^{-gamma tau}.

    Returns an array of shape (..., 2, 2)."""
    a = np.asarray(a, dtype=float)
    x = np.asarray(x, dtype=float)
    M = np.empty(np.broadcast(a, x).shape + (2, 2))
    M[..., 0, 1] = a * (1.0 - x)
    M[..., 0, 0] = 1.0 - M[..., 0, 1]
    M[..., 1, 0] = (1.0 - a) * (1.0 - x)
    M[..., 1, 1] = 1.0 - M[..., 1, 0]
    return M


def cycle_statistics(omega_h, omega_c, tau_h, tau_c, params: OttoParams = OttoParams()):
    """Statistics of the limit cycle (vectorized over the arguments).

    Returns a dict with:
        W        mean extracted work per cycle
        Q_h      mean heat absorbed from the hot bath per cycle
        eta      efficiency W / Q_h (= 1 - omega_c / omega_h when W > 0)
        P        mean power
        dP       power fluctuations: lim Var(W_t) / t
        Sigma    entropy production rate
        var_w    asymptotic variance per cycle, theta''(0)
        tau_cyc  cycle duration
        p1, p2   excited population at the end of the hot / cold contacts
    """
    omega_h, omega_c, tau_h, tau_c = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (omega_h, omega_c, tau_h, tau_c)))
    p = params
    a_h = excited_population(omega_h, p.T_h)
    a_c = excited_population(omega_c, p.T_c)
    x_h = np.exp(-p.gamma_h * tau_h)
    x_c = np.exp(-p.gamma_c * tau_c)
    D = x_h * x_c                      # det T(chi), independent of chi
    dw = omega_h - omega_c             # d_omega

    Mh = relaxation_matrix(a_h, x_h)
    Mc = relaxation_matrix(a_c, x_c)

    # tr T(chi) = c0 + c_plus e^{chi d_omega} + c_minus e^{-chi d_omega}
    c_plus = Mh[..., 0, 1] * Mc[..., 1, 0]
    c_minus = Mh[..., 1, 0] * Mc[..., 0, 1]

    one_m_D = 1.0 - D
    t1 = dw * (c_plus - c_minus)             # tr'(0)
    t2 = dw ** 2 * (c_plus + c_minus)        # tr''(0)
    # lambda'(0) = t'/(1-D),  lambda''(0) = t''/(1-D) - 2D t'^2/(1-D)^3,  lambda(0) = 1
    lam1 = t1 / one_m_D
    lam2 = t2 / one_m_D - 2.0 * D * t1 ** 2 / one_m_D ** 3
    mean_w = lam1                            # theta'(0)
    var_w = lam2 - lam1 ** 2                 # theta''(0)

    # Deterministic limit-cycle populations (consistency check)
    p1 = (a_h * (1 - x_h) + x_h * a_c * (1 - x_c)) / one_m_D
    p2 = a_c * (1 - x_c) + p1 * x_c

    tau_cyc = tau_h + tau_c + 2.0 * p.tau_q
    Q_h = omega_h * (p1 - p2)
    # eta = W / Q_h = d_omega / omega_h exactly (both are proportional to p1 - p2);
    # the closed form avoids 0/0 noise when p1 - p2 -> 0.
    eta = np.where(p1 - p2 > 0, dw / omega_h, 0.0)

    # Entropy production rate: Sigma = (Q_c / T_c - Q_h / T_h) / tau, with
    # Q_c = omega_c (p1 - p2) the heat released to the cold bath. Equivalent to
    # Sigma = P (eta_C - eta) / (eta T_c), without the 0/0 when eta -> 0.
    Sigma = (p1 - p2) * (omega_c / p.T_c - omega_h / p.T_h) / tau_cyc

    return dict(W=mean_w, Q_h=Q_h, eta=eta, P=mean_w / tau_cyc, dP=var_w / tau_cyc,
                Sigma=Sigma, var_w=var_w, tau_cyc=tau_cyc, p1=p1, p2=p2)


def tur_ratio(P, dP, eta, params: OttoParams = OttoParams()):
    """Ratio of the thermodynamic uncertainty relation for heat engines
    (Pietzonka & Seifert, PRL 120, 190602, 2018):

        eta <= eta_C / (1 + 2 T_c P / dP)   <=>   xi = 2 T_c P eta / (dP (eta_C - eta)) <= 1

    The bound holds for steady-state engines; periodically driven engines such
    as this one may violate it slightly (xi > 1)."""
    eta_c = params.eta_carnot
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.0 * params.T_c * P * eta / (dP * (eta_c - eta))


def monte_carlo_work(omega_h, omega_c, tau_h, tau_c, params: OttoParams = OttoParams(),
                     n_chains=4000, n_cycles=2000, burn_in=50, seed=0):
    """Stochastic trajectory simulation (validation). Returns the mean work per
    cycle and the asymptotic variance per cycle, Var(sum w) / n."""
    rng = np.random.default_rng(seed)
    p = params
    a_h = float(excited_population(omega_h, p.T_h))
    a_c = float(excited_population(omega_c, p.T_c))
    x_h, x_c = np.exp(-p.gamma_h * tau_h), np.exp(-p.gamma_c * tau_c)
    dw = omega_h - omega_c
    s = (rng.random(n_chains) < 0.5).astype(float)
    total = np.zeros(n_chains)
    for k in range(burn_in + n_cycles):
        s1 = (rng.random(n_chains) < a_h + (s - a_h) * x_h).astype(float)
        s2 = (rng.random(n_chains) < a_c + (s1 - a_c) * x_c).astype(float)
        if k >= burn_in:
            total += dw * (s1 - s2)
        s = s2
    return total.mean() / n_cycles, total.var(ddof=1) / n_cycles
