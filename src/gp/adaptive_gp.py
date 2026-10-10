"""Time-decayed (weighted) Gaussian-process regression.

Standard GP regression treats every historical observation as equally
reliable.  When the objective moves, old observations become stale and should
matter less.  We implement this with a mathematically clean, exact mechanism
(spec §16, strategy C + §17): *heteroscedastic observation noise derived from
time-decay weights*.

For an observation ``i`` made at time ``t_i`` and evaluated at current time
``t`` we define the age ``a_i = t - t_i`` and the decay weight

    w_i = exp(-lambda * a_i),      lambda = ln(2) / h,

where ``h`` is the half-life (``w_i = 0.5`` when ``a_i = h``).  The
per-observation noise variance is

    sigma_i^2 = sigma_base^2 / max(w_i, w_min).

Recent observations (high ``w_i``) get a *low* effective noise, old ones a
*high* effective noise.  GP regression with observation covariance
``K + diag(sigma_i^2)`` is exact (no approximation), so the posterior mean
and variance are computed with the standard GP equations and the noise matrix
simply becomes diagonal-heteroscedastic.  This down-weights stale data while
retaining it, which is numerically well behaved (no point is ever fully
forgotten).

The target is standardised with *weighted* mean / variance so that the
normalisation is consistent with the weights.
"""

from __future__ import annotations

import numpy as np

from .gaussian_process import GaussianProcess, _PSDSolver
from .kernels import Kernel

__all__ = ["WeightedGaussianProcess", "decay_weights"]


def decay_weights(age: np.ndarray, half_life: float | None, weight_min: float = 1e-3) -> np.ndarray:
    """``w = exp(-ln(2) * age / h)`` clipped to ``[weight_min, 1]``.

    ``half_life = None`` or ``inf`` gives all weights equal to 1 (no
    forgetting).
    """
    age = np.asarray(age, dtype=float)
    if not np.isfinite(age).all():
        raise ValueError("age must contain only finite values")
    weight_min = float(weight_min)
    if not 0.0 < weight_min <= 1.0:
        raise ValueError("weight_min must be in (0, 1]")
    if half_life is None or np.isinf(float(half_life)):
        return np.ones_like(age)
    if not np.isfinite(float(half_life)) or float(half_life) <= 0.0:
        raise ValueError("half_life must be positive, finite, or infinity")
    lam = np.log(2.0) / float(half_life)
    w = np.exp(-lam * np.maximum(age, 0.0))
    return np.clip(w, weight_min, 1.0)


class WeightedGaussianProcess(GaussianProcess):
    """GP regression with time-decay-derived heteroscedastic observation noise.

    The class reuses :class:`GaussianProcess` for the posterior machinery;
    only the construction of the noise matrix and the target standardisation
    differ.
    """

    def __init__(
        self,
        kernel: Kernel,
        base_noise_var: float,
        weight_min: float = 1e-3,
        normalize: bool = True,
        jitter: float = 1e-8,
        max_jitter: float = 1e-4,
    ) -> None:
        super().__init__(kernel, noise_var=base_noise_var, normalize=normalize,
                         jitter=jitter, max_jitter=max_jitter)
        self.base_noise_var = float(base_noise_var)
        self.weight_min = float(weight_min)
        if not np.isfinite(self.base_noise_var) or self.base_noise_var < 0.0:
            raise ValueError("base_noise_var must be finite and non-negative")
        if not 0.0 < self.weight_min <= 1.0:
            raise ValueError("weight_min must be in (0, 1]")
        self.last_weights: np.ndarray | None = None
        self.last_effective_noise_var: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        t_obs: np.ndarray,
        t_now: float,
        half_life: float | None,
    ) -> "WeightedGaussianProcess":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        t_obs = np.asarray(t_obs, dtype=float).ravel()
        if X.shape[0] != y.shape[0] or t_obs.shape[0] != y.shape[0]:
            raise ValueError("X, y and t_obs have inconsistent sizes")
        if X.shape[0] < 2:
            raise ValueError("at least two observations are required")
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2D array")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("X and y must contain only finite values")
        if not np.isfinite(float(t_now)):
            raise ValueError("t_now must be finite")

        age = np.maximum(float(t_now) - t_obs, 0.0)
        w = decay_weights(age, half_life, self.weight_min)

        # Weighted standardisation of the target.
        sw = float(w.sum())
        mu_w = float((w * y).sum() / sw)
        var_w = float((w * (y - mu_w) ** 2).sum() / sw)
        sd_w = float(np.sqrt(var_w)) if var_w > 1e-12 else 1.0
        y_n = (y - mu_w) / sd_w

        # Heteroscedastic observation noise in standardised units.
        eff_noise_var = (self.base_noise_var / w) / (sd_w * sd_w)
        K = self.kernel(X, X)
        A = K + np.diag(eff_noise_var)
        self._solver = _PSDSolver.from_matrix(A, self.jitter, self.max_jitter)
        self._alpha = self._solver.solve(y_n)
        self.X_train = X
        self._train_y = y
        self.y_mu, self.y_sd = mu_w, sd_w
        self.last_weights = w
        self.last_effective_noise_var = self.base_noise_var / w
        self._fitted = True
        return self
