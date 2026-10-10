"""Regret / tracking metric unit tests on hand-built trajectories (spec §41)."""

import numpy as np
import pytest

from src.base import Observation, Trajectory
from src.evaluation import (
    adaptation_regret,
    best_so_far,
    cumulative_regret,
    exploration_rate,
    instantaneous_regret,
    recovery_time,
    tracking_error,
)


def _make_traj(f_true, f_opt, x, x_opt):
    traj = Trajectory()
    for t in range(len(f_true)):
        traj.observations.append(Observation(
            t=t, x=np.asarray(x[t], float), y=float(f_true[t]) + 0.01,
            f_true=float(f_true[t]), f_opt=float(f_opt[t]),
            x_opt=np.asarray(x_opt[t], float), is_initial=t < 2,
        ))
    return traj


def test_instantaneous_and_cumulative_regret_handmade():
    # f_opt = 1 always; f_true = [1, 0.8, 0.6] -> regrets [0, 0.2, 0.4]
    traj = _make_traj([1.0, 0.8, 0.6], [1.0, 1.0, 1.0],
                      [[0.5], [0.4], [0.3]], [[0.5], [0.5], [0.5]])
    r = instantaneous_regret(traj)
    np.testing.assert_allclose(r, [0.0, 0.2, 0.4], atol=1e-12)
    np.testing.assert_allclose(cumulative_regret(traj), [0.0, 0.2, 0.6], atol=1e-12)


def test_best_so_far_uses_noise_free_values():
    traj = _make_traj([0.9, 0.5, 0.95], [1.0] * 3,
                      [[0.5], [0.5], [0.5]], [[0.5]] * 3)
    np.testing.assert_allclose(best_so_far(traj, noise_free=True),
                               [0.9, 0.9, 0.95], atol=1e-12)


def test_recovery_time_handmade():
    # change at t=5; quality recovers to within 5% of optimum at t=9
    f_opt = np.ones(12)
    f_true = np.array([1.0, 1.0, 1.0, 1.0, 1.0,   # before change
                       0.2, 0.1, 0.3, 0.5,        # post-change slump
                       0.96, 0.97, 0.98])          # recovery
    traj = _make_traj(f_true, f_opt, [[0.5]] * 12, [[0.5]] * 12)
    assert recovery_time(traj, t_change=5, tol_frac=0.05) == pytest.approx(4.0)
    # never recovering -> nan
    f_bad = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    traj_bad = _make_traj(f_bad, f_opt, [[0.5]] * 12, [[0.5]] * 12)
    assert np.isnan(recovery_time(traj_bad, t_change=5, tol_frac=0.05))


def test_adaptation_regret_window():
    f_opt = np.ones(10)
    f_true = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.5, 1.0, 1.0])
    traj = _make_traj(f_true, f_opt, [[0.5]] * 10, [[0.5]] * 10)
    # window [5, 8): regrets 1 + 1 + 0.5 = 2.5
    assert adaptation_regret(traj, t_change=5, window=3) == pytest.approx(2.5)


def test_tracking_error_zero_at_optimum():
    traj = _make_traj([1.0] * 4, [1.0] * 4,
                      [[0.5], [0.5], [0.5], [0.5]],
                      [[0.5], [0.5], [0.5], [0.5]])

    class _Bench:
        def optima_set(self, t):
            return np.asarray([[0.5]])

    err = tracking_error(traj, _Bench())
    np.testing.assert_allclose(err, [0.0, 0.0, 0.0, 0.0])


def test_tracking_error_uses_nearest_equivalent_optimum():
    traj = _make_traj([1.0, 1.0], [1.0, 1.0],
                      [[0.1], [3.2]],
                      [[0.1], [3.2]])

    class _Bench:
        def optima_set(self, t):
            # two equivalent optima (sinusoid-like): 0.1 and 3.2
            return np.asarray([[0.1], [3.2]])

    err = tracking_error(traj, _Bench())
    np.testing.assert_allclose(err, [0.0, 0.0], atol=1e-12)


def test_exploration_rate():
    # sequential points (t>=2): f_true 0.9, 0.1 with f_opt 1 -> 1 of 2 exploratory
    traj = _make_traj([1.0, 1.0, 0.9, 0.1], [1.0] * 4,
                      [[0.5]] * 4, [[0.5]] * 4)
    assert exploration_rate(traj, quantile=0.5) == pytest.approx(0.5)
