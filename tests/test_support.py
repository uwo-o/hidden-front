"""Tests for the support test and the theorems it is used to check."""
import numpy as np

from hidden_front.support import support_gap


def _normalize(F):
    return (F - F.min(0)) / (F.max(0) - F.min(0))


def test_convex_front_is_fully_supported():
    t = np.linspace(0, np.pi / 2, 40)
    F = np.column_stack([1 - np.cos(t), 1 - np.sin(t)])     # convex (quarter circle)
    assert np.all(support_gap(F, F) < 1e-9)


def test_concave_front_interior_is_unsupported():
    t = np.linspace(0, np.pi / 2, 40)
    F = np.column_stack([np.cos(t), np.sin(t)])             # concave: only the end points
    gaps = support_gap(F, F)
    assert gaps[0] < 1e-9 and gaps[-1] < 1e-9
    assert np.all(gaps[1:-1] > 1e-3)


def test_product_structure_is_a_saddle_and_perspective_is_convex():
    """Theorem 1 and Theorem 2 on a small synthetic product-structured front."""
    pts = []
    for r in np.linspace(0.1, 0.9, 12):
        for a in np.linspace(0.05, 1.0, 12) * 4 * r * (1 - r):
            pts.append((a, r, a * 2 * r / (1 - r), a * (1 / r - 1)))
    a, r, c, b = np.array(pts).T
    ratio = support_gap(_normalize(np.column_stack([-a, -r, c])),
                        _normalize(np.column_stack([-a, -r, c])))
    extensive = support_gap(_normalize(np.column_stack([-a, b, c])),
                            _normalize(np.column_stack([-a, b, c])))
    assert (ratio > 1e-3).mean() > 0.5
    assert np.all(extensive < 1e-3)
