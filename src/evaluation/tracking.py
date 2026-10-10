"""Tracking metrics for moving optima (spec §27, metric 4) plus a few
descriptive behaviour metrics.
"""

from __future__ import annotations

import numpy as np

__all__ = ["tracking_error", "exploration_rate"]


def tracking_error(traj, benchmark) -> np.ndarray:
    """``min_j ||x_t - x^(j)*_t||_2`` against *all* equivalent optima.

    Benchmarks report ``optima_set(t)`` (e.g. the sinusoid has two equivalent
    maxima); the error is the distance to the closest of them, so the metric
    does not penalise being at an equally good optimum.
    """
    X = traj.x()
    n = X.shape[0]
    err = np.empty(n, dtype=float)
    for t in range(n):
        opts = benchmark.optima_set(t)  # (k, d)
        d2 = np.sum((X[t][None, :] - opts) ** 2, axis=1)
        err[t] = float(np.sqrt(d2.min()))
    return err


def exploration_rate(traj, quantile: float = 0.5) -> float:
    """Fraction of *sequential* points outside the high-value region.

    A point is called exploratory when its noise-free value is below
    ``quantile * f_opt(t)``.  This is a descriptive metric, not a verdict:
    some exploration is necessary to keep up with a moving optimum.
    """
    mask = ~traj.is_initial()
    f_true = traj.f_true()
    f_opt = traj.f_opt()
    if not mask.any():
        return 0.0
    frac = f_true[mask] / np.maximum(f_opt[mask], 1e-12)
    return float((frac < quantile).mean())
