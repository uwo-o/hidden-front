"""Reverse-osmosis (RO) desalination plant: a sizing problem with a free scale.

Model (dimensionless, standard one-dimensional module model with complete salt rejection, no
concentration polarization and no axial pressure drop):

* The plant has ``n`` identical pressure vessels in parallel (continuous relaxation). Each vessel
  receives a feed flow ``q_f`` at applied pressure ``p``.
* Pressures are in units of the feed osmotic pressure pi_0 (about 27 bar for seawater) and flows in
  units of L_p A pi_0, the permeate flow of one vessel at unit net driving pressure.
* Along the vessel the retentate flow Q(x), x in [0, 1], obeys
      dQ/dx = -(p - pi(x)),   pi(x) = q_f / Q(x)
  (salt mass balance with complete rejection). With c = q_f / p and y = Q - c,
      y + c ln y = (q_f - c) + c ln(q_f - c) - p,
  which is solved in closed form with the Wright omega function. Flux requires p > 1.
* Recovery r = q / q_f with permeate q = q_f - Q(1). The thermodynamic restriction
  p >= pi_brine = 1 / (1 - r) holds automatically.
* Pump power with an isobaric energy-recovery device of efficiency eta_E and pump efficiency eta_p:
      w = p (q_f - eta_E Q(1)) / eta_p.

Plant objectives (all per plant): production Q = n q, recovery r, pump power W = n w, membrane
vessels n (capital), and brine discharge B = n Q(1) = Q (1/r - 1). Production, power, vessels and
brine are extensive (proportional to n at fixed intensive design (q_f, p)); recovery is intensive.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import wrightomega


@dataclass(frozen=True)
class ROParams:
    eta_pump: float = 0.8          # high-pressure pump efficiency
    eta_erd: float = 0.95          # energy-recovery device efficiency (isobaric pressure exchanger)
    p_max: float = 3.0             # maximum applied pressure / pi_0 (about 82 bar for seawater)


def vessel(q_f, p, params: ROParams = ROParams()):
    """Per-vessel permeate q, brine Q1, recovery r and pump power w (arrays broadcast)."""
    q_f = np.asarray(q_f, dtype=float)
    p = np.asarray(p, dtype=float)
    c = q_f / p
    y0 = q_f - c
    K = y0 + c * np.log(y0) - p
    omega = np.real(wrightomega(K / c - np.log(c)))
    Q1 = c + c * omega
    q = q_f - Q1
    r = q / q_f
    w = p * (q_f - params.eta_erd * Q1) / params.eta_pump
    return dict(q=q, Q1=Q1, r=r, w=w)


def plant(n, q_f, p, params: ROParams = ROParams()):
    """Plant-level quantities for n vessels in parallel."""
    v = vessel(q_f, p, params)
    n = np.asarray(n, dtype=float)
    return dict(Q=n * v["q"], B=n * v["Q1"], r=v["r"], W=n * v["w"], n=n * np.ones_like(v["q"]),
                sec=v["w"] / v["q"], q=v["q"])


def sec_thermodynamic_limit(r, params: ROParams = ROParams()):
    """Specific energy w/q at the thermodynamic limit p = 1/(1-r) (infinite membrane area)."""
    r = np.asarray(r, dtype=float)
    return (1.0 - params.eta_erd * (1.0 - r)) / (params.eta_pump * r * (1.0 - r))
