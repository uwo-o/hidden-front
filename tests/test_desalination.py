import numpy as np
import pytest
from scipy.integrate import solve_ivp

from hidden_front.desalination import ROParams, plant, sec_thermodynamic_limit, vessel
from hidden_front.problems import DesalinationProblem


@pytest.mark.parametrize("q_f,p", [(0.5, 1.5), (2.0, 2.5), (0.05, 3.0), (5.0, 1.2)])
def test_closed_form_matches_ode(q_f, p):
    sol = solve_ivp(lambda x, Q: -(p - q_f / Q), [0, 1], [q_f], rtol=1e-11, atol=1e-13)
    assert vessel(q_f, p)["Q1"] == pytest.approx(sol.y[0, -1], rel=1e-8)


def test_thermodynamic_restriction():
    rng = np.random.default_rng(0)
    q_f, p = 10 ** rng.uniform(-1.5, 1, 1000), rng.uniform(1.02, 3, 1000)
    r = vessel(q_f, p)["r"]
    assert np.all(p >= 1 / (1 - r) - 1e-9)          # applied pressure >= brine osmotic pressure
    assert np.all((r > 0) & (r < 1))


def test_sec_approaches_thermodynamic_limit():
    # small feed per vessel (long residence) at p slightly above the brine osmotic pressure
    r_target = 0.4
    p = 1 / (1 - r_target) + 1e-3
    v = vessel(1e-3, p)
    assert v["r"] == pytest.approx(1 - 1 / p, rel=1e-6)
    assert v["w"] / v["q"] == pytest.approx(sec_thermodynamic_limit(v["r"]), rel=1e-3)


def test_scale_homogeneity():
    a, b = plant(3.0, 0.7, 2.0), plant(7.5, 0.7, 2.0)
    for k in ("Q", "B", "W", "n"):
        assert b[k] / a[k] == pytest.approx(2.5)
    assert a["r"] == pytest.approx(b["r"])


def test_problem_formulations_share_solutions():
    X = np.array([[10.0, -0.3, 2.0], [50.0, 0.2, 2.8]])
    Fr = DesalinationProblem(objectives="recovery").evaluate(X, return_values_of=["F"])
    Fb = DesalinationProblem(objectives="brine").evaluate(X, return_values_of=["F"])
    Q, r, B = -Fr[:, 0], -Fr[:, 1], Fb[:, 1]
    assert np.allclose(B, Q * (1 / r - 1))
    assert np.allclose(Fr[:, [0, 2, 3]], Fb[:, [0, 2, 3]])


def test_params_default():
    assert ROParams().eta_erd < 1 and ROParams().eta_pump < 1
