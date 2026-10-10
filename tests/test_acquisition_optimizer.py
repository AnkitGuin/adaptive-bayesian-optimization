"""Acquisition-optimizer unit tests (grid + refinement, deterministic)."""

import numpy as np
import pytest

from src.base import SearchSpace
from src.optimization import AcquisitionOptimizer


def test_1d_finds_known_maximum():
    space = SearchSpace(lower=[0.0], upper=[1.0])
    opt = AcquisitionOptimizer(space, n_grid_1d=201, n_restarts=3)

    def acq(X):
        X = np.atleast_2d(X)
        return -((X[:, 0] - 0.37) ** 2) / 0.01 + 0.05 * X[:, 0]

    # f'(x) = -200 (x-0.37) + 0.05 = 0  ->  x = 0.37 + 0.05/200 = 0.37025
    x, v = opt(acq)
    assert x.shape == (1,)
    assert x[0] == pytest.approx(0.37025, abs=1e-4)


def test_1d_boundary_maximum():
    space = SearchSpace(lower=[-2.0], upper=[2.0])
    opt = AcquisitionOptimizer(space, n_grid_1d=301, n_restarts=3)
    x, v = opt(lambda X: -np.atleast_2d(X)[:, 0] ** 2)  # max at 0
    assert x[0] == pytest.approx(0.0, abs=1e-3)
    x2, _ = opt(lambda X: np.atleast_2d(X)[:, 0])  # max at boundary +2
    assert x2[0] == pytest.approx(2.0, abs=1e-6)


def test_2d_finds_known_maximum():
    space = SearchSpace(lower=[0.0, 0.0], upper=[1.0, 1.0])
    opt = AcquisitionOptimizer(space, n_grid_d=21, n_restarts=3)

    def acq(X):
        X = np.atleast_2d(X)
        return -((X[:, 0] - 0.3) ** 2 + (X[:, 1] - 0.7) ** 2)

    x, v = opt(acq)
    assert x[0] == pytest.approx(0.3, abs=1e-3)
    assert x[1] == pytest.approx(0.7, abs=1e-3)


def test_3d_finds_known_maximum():
    space = SearchSpace(lower=[0.0, 0.0, 0.0], upper=[1.0, 1.0, 1.0])
    opt = AcquisitionOptimizer(space, n_grid_d=9, n_restarts=2)

    def acq(X):
        X = np.atleast_2d(X)
        return -np.sum((X - np.array([0.2, 0.5, 0.9])) ** 2, axis=1)

    x, v = opt(acq)
    np.testing.assert_allclose(x, [0.2, 0.5, 0.9], atol=1e-2)


def test_deterministic():
    space = SearchSpace(lower=[0.0, 0.0], upper=[1.0, 1.0])
    opt = AcquisitionOptimizer(space, n_grid_d=17, n_restarts=3)

    def acq(X):
        X = np.atleast_2d(X)
        return np.sin(3 * X[:, 0]) * np.cos(2 * X[:, 1]) + 0.1 * X[:, 0]

    a = opt(acq)
    b = opt(acq)
    np.testing.assert_allclose(a[0], b[0], atol=0)
    assert a[1] == b[1]


def test_supports_at_most_3d():
    with pytest.raises(ValueError):
        AcquisitionOptimizer(SearchSpace(lower=[0] * 4, upper=[1] * 4))


def test_result_inside_bounds():
    space = SearchSpace(lower=[0.0, 0.0, 0.0], upper=[1.0, 1.0, 1.0])
    opt = AcquisitionOptimizer(space, n_grid_d=9, n_restarts=3)
    x, _ = opt(lambda X: -np.atleast_2d(X)[:, 0] - 0.5 * np.atleast_2d(X)[:, 2])
    assert space.contains(x.reshape(1, -1))
