"""Tests for the single-qubit Otto engine model."""
import numpy as np
import pytest

from hidden_front.otto import (OttoParams, cycle_statistics, excited_population,
                               monte_carlo_work, relaxation_matrix)

CASES = [(2.0, 1.4, 0.5, 0.7), (1.0, 0.7, 0.1, 0.1), (4.0, 2.5, 2.0, 1.0)]


def test_relaxation_matrix_is_stochastic_with_eigenvalue_x():
    M = relaxation_matrix(0.3, 0.6)
    assert np.allclose(M.sum(-1), 1.0)
    assert np.allclose(sorted(np.linalg.eigvals(M).real), [0.6, 1.0])


def test_mean_work_matches_deterministic_populations():
    for wh, wc, th, tc in CASES:
        s = cycle_statistics(wh, wc, th, tc)
        assert np.isclose(s["W"], (wh - wc) * (s["p1"] - s["p2"]))


def test_otto_efficiency():
    s = cycle_statistics(2.0, 1.4, 0.5, 0.7)
    assert np.isclose(s["eta"], 1 - 1.4 / 2.0)


def test_quasistatic_limit():
    """With full thermalization, p1 and p2 are the equilibrium populations."""
    p = OttoParams()
    s = cycle_statistics(2.0, 1.4, 50.0, 50.0, p)
    assert np.isclose(s["p1"], excited_population(2.0, p.T_h))
    assert np.isclose(s["p2"], excited_population(1.4, p.T_c))


def test_tilted_derivatives_against_finite_differences():
    """Analytic theta'(0), theta''(0) against finite differences of the numerical eigenvalue."""
    p = OttoParams()
    wh, wc, th, tc = CASES[0]
    Mh = relaxation_matrix(excited_population(wh, p.T_h), np.exp(-th))
    Mc = relaxation_matrix(excited_population(wc, p.T_c), np.exp(-tc))
    dw = wh - wc

    def theta(chi):
        E1 = np.diag([1.0, np.exp(chi * dw)])
        E2 = np.diag([1.0, np.exp(-chi * dw)])
        return np.log(max(np.linalg.eigvals(Mh @ E1 @ Mc @ E2).real))

    h = 1e-3
    d1 = (theta(h) - theta(-h)) / (2 * h)
    d2 = (theta(h) - 2 * theta(0) + theta(-h)) / h ** 2
    s = cycle_statistics(wh, wc, th, tc, p)
    assert np.isclose(s["W"], d1, rtol=1e-5)
    assert np.isclose(s["var_w"], d2, rtol=1e-4)


@pytest.mark.parametrize("case", CASES)
def test_monte_carlo(case):
    s = cycle_statistics(*case)
    mw, vw = monte_carlo_work(*case, n_chains=4000, n_cycles=1500, seed=1)
    assert np.isclose(mw, s["W"], rtol=0.03, atol=2e-4)
    assert np.isclose(vw, s["var_w"], rtol=0.08)
