"""Regret and recovery metrics.

All regret quantities are computed from **noise-free** objective values
(spec §8, §27, §33): the noisy observation ``y`` is never used in metric
computation.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "instantaneous_regret",
    "cumulative_regret",
    "best_so_far",
    "adaptation_regret",
    "recovery_time",
]


def instantaneous_regret(traj) -> np.ndarray:
    """``R_t = f_t(x*_t) - f_t(x_t)`` for every observation t."""
    return traj.f_opt() - traj.f_true()


def cumulative_regret(traj) -> np.ndarray:
    """Running cumulative regret ``CR_t = sum_{i<=t} R_i`` (array over t)."""
    return np.cumsum(instantaneous_regret(traj))


def best_so_far(traj, noise_free: bool = True) -> np.ndarray:
    """``max_{i<=t} f_i(x_i)`` -- meaningful mainly for stationary studies.

    With ``noise_free=True`` (default) the noise-free values are used so the
    curve reflects the true objective quality found.
    """
    values = traj.f_true() if noise_free else traj.y()
    return np.maximum.accumulate(values)


def adaptation_regret(traj, t_change: int, window: int) -> float:
    """Regret summed over the post-change window ``[t_change, t_change+window)``."""
    r = instantaneous_regret(traj)
    lo = max(int(t_change), 0)
    hi = min(int(t_change) + int(window), r.size)
    if hi <= lo:
        return float("nan")
    return float(r[lo:hi].sum())


def recovery_time(traj, t_change: int, tol_frac: float = 0.05,
                  tol_abs: float = 0.0, max_wait: int | None = None) -> float:
    """First ``t >= t_change`` with ``f_t(x_t) >= f_t(x*_t) - tol``.

    ``tol = max(tol_abs, tol_frac * f_opt_t)``.  Returns ``nan`` if the
    criterion is never met within the horizon (or ``max_wait`` after the
    change point).
    """
    f_true = traj.f_true()
    f_opt = traj.f_opt()
    n = f_true.size
    hi = n if max_wait is None else min(n, int(t_change) + int(max_wait))
    for t in range(int(t_change), hi):
        tol = max(float(tol_abs), float(tol_frac) * float(f_opt[t]))
        if f_true[t] >= f_opt[t] - tol:
            return float(t - int(t_change))
    return float("nan")
