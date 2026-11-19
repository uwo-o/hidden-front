"""Tests for the piecewise-controlled quantum-dot engine."""
import numpy as np

from hidden_front.otto import OttoParams, cycle_statistics
from hidden_front.piecewise import QDParams, cycle_stats


def test_two_steps_reduce_to_otto():
    p = QDParams()
    for uh, uc, tau in [(0.9, 0.5, 1.0), (0.6, 0.4, 5.0)]:
        s = cycle_stats(np.array([[uh, uc]]), np.array([[True, False]]), np.array([tau]), p)
        r = cycle_statistics(p.E0 * uh, p.E0 * uc, tau / 2, tau / 2,
                             OttoParams(T_h=1 / p.beta_h, T_c=1 / p.beta_c, tau_q=0.0))
        for k in ["P", "dP", "Sigma", "eta"]:
            assert np.isclose(s[k][0], float(r[k]), rtol=1e-9, atol=1e-14)


def test_first_law_random_protocols():
    rng = np.random.default_rng(0)
    u = rng.uniform(0.2, 1.1, (50, 8))
    hot = rng.random((50, 8)) < 0.5
    tau = rng.uniform(0.1, 10, 50)
    s = cycle_stats(u, hot, tau)
    assert np.allclose(s["W"], s["Qsum"], atol=1e-12)
    assert np.all(s["var_w"] >= -1e-12)
    assert np.all(s["Sigma"] >= -1e-12)          # second law


def test_monte_carlo_random_protocol():
    rng = np.random.default_rng(1)
    p = QDParams()
    N = 6
    u = rng.uniform(0.2, 1.1, N)
    hot = rng.random(N) < 0.5
    tau = 2.0
    s = cycle_stats(u[None], hot[None], np.array([tau]), p)
    eps = p.E0 * u
    beta = np.where(hot, p.beta_h, p.beta_c)
    a = 1 / (1 + np.exp(beta * eps))
    x = np.exp(-p.gamma * tau / N)
    nc, L = 20000, 600
    st = (rng.random(nc) < 0.5).astype(float)
    tot = np.zeros(nc)
    for cyc in range(L + 30):
        for k in range(N):
            if cyc >= 30:
                tot += -(eps[k] - eps[k - 1]) * st
            st = (rng.random(nc) < a[k] + (st - a[k]) * x).astype(float)
    assert np.isclose(tot.mean() / L, s["W"][0], rtol=0.05, atol=2e-3)
    assert np.isclose(tot.var() / L, s["var_w"][0], rtol=0.08)
