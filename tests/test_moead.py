"""Tests for the MOEA/D implementation."""
import numpy as np
from pymoo.problems import get_problem

from hidden_front.moead import MOEAD, sparsity, ws_transform


def test_ws_transform_is_self_inverse():
    lam = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]])
    assert np.allclose(ws_transform(ws_transform(lam, eps=0), eps=0), lam)


def test_sparsity_duplicates_are_zero():
    A = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    sl = sparsity(A)
    assert sl[0] == 0 and sl[1] == 0 and sl[2] > 0


def test_awa_keeps_population_size_and_adjusts():
    pb = get_problem("dtlz2", n_var=12, n_obj=3)
    alg = MOEAD(pb, n_partitions=6, max_gen=30, ws=True, awa=True, wag=2, seed=0).run()
    assert len(alg.F) == alg.N == len(alg.lam) == len(alg.X)
    assert alg.n_adjust > 0
    assert len(alg.EP_F) <= alg.ep_cap


def test_moead_converges_on_dtlz2():
    pb = get_problem("dtlz2", n_var=12, n_obj=3)
    alg = MOEAD(pb, n_partitions=6, max_gen=100, ws=True, awa=False, seed=0).run()
    # on the DTLZ2 front, sum f_i^2 = 1
    r = np.sqrt((alg.result_F ** 2).sum(1))
    assert np.median(r) < 1.05


def test_weighted_sum_scalarization_runs():
    pb = get_problem("dtlz2", n_var=12, n_obj=3)
    alg = MOEAD(pb, n_partitions=6, max_gen=20, scalarization="ws", seed=0).run()
    assert len(alg.result_F) > 0


def test_lp_scalarization_tends_to_tchebycheff():
    import numpy as np
    from hidden_front.moead import MOEAD
    from hidden_front.problems import OttoProblem
    a = MOEAD(OttoProblem(), scalarization="lp", p=200.0)
    a.z, a.nadir = np.zeros(3), np.ones(3)
    F = np.array([[0.2, 0.5, 0.1], [0.4, 0.1, 0.3]])
    lam = np.array([0.3, 0.3, 0.4])
    tch = np.max(lam * F, axis=-1)
    assert np.allclose(a._g(F, lam), tch, rtol=0.05)
