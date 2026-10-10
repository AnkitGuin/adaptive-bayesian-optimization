"""scikit-learn GP wrapper -- used *only* to cross-validate the custom GP.

Spec §10/§50: scikit-learn is a permitted library, but the core GP logic must
be our own.  We therefore keep the custom numpy GP as the production code
path and use :class:`SklearnGaussianProcess` as an independent reference
implementation in the tests (identical kernel parameters, identical data).
"""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern as _Matern
from sklearn.gaussian_process.kernels import RBF

__all__ = ["SklearnGaussianProcess"]


def _sklearn_kernel(kind: str, lengthscale: float, signal_variance: float):
    # Note: this scikit-learn build's kernels have no amplitude parameter
    # (fixed at 1.0), so the signal variance is applied by scaling the kernel.
    kind = str(kind).lower()
    if kind == "matern52":
        base = _Matern(length_scale=lengthscale, nu=2.5,
                       length_scale_bounds=(lengthscale, lengthscale))
    elif kind == "matern32":
        base = _Matern(length_scale=lengthscale, nu=1.5,
                       length_scale_bounds=(lengthscale, lengthscale))
    elif kind == "rbf":
        base = RBF(length_scale=lengthscale,
                   length_scale_bounds=(lengthscale, lengthscale))
    else:
        raise ValueError(f"unsupported kernel: {kind}")
    if signal_variance != 1.0:
        base = signal_variance * base
    return base


class _FixedParamGPR(GaussianProcessRegressor):
    """GPR that skips hyperparameter optimization.

    All kernel hyperparameters are constructed *fixed* (bounds collapsed to a
    point), so the optimization step is a no-op.  This also sidesteps an
    incompatibility of this scikit-learn/scipy build with the optimizer-result
    handling in ``fit``.
    """

    def _constrained_optimization(self, obj_func, theta_initial, bound):  # noqa: N803
        # Nothing to optimize (all parameters fixed): report the initial theta.
        neg_lml = float(obj_func(np.asarray(theta_initial, dtype=float),
                                 eval_gradient=False))
        return np.asarray(theta_initial, dtype=float), neg_lml


class SklearnGaussianProcess:
    """Thin wrapper matching the custom GP's API on a toy interface."""

    def __init__(self, kind: str = "matern52", lengthscale: float = 0.25,
                 signal_variance: float = 1.0, noise_var: float = 0.0,
                 normalize: bool = True) -> None:
        self.kind = kind
        self.lengthscale = float(lengthscale)
        self.signal_variance = float(signal_variance)
        self.noise_var = float(noise_var)
        self.normalize = bool(normalize)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnGaussianProcess":
        kernel = _sklearn_kernel(self.kind, self.lengthscale, self.signal_variance)
        if self.noise_var > 0:
            from sklearn.gaussian_process.kernels import WhiteKernel
            kernel = kernel + WhiteKernel(noise_level=self.noise_var,
                                           noise_level_bounds=(self.noise_var, self.noise_var))
        self.model = _FixedParamGPR(
            kernel=kernel,
            normalize_y=self.normalize,
            alpha=0.0,
            n_restarts_optimizer=0,
        )
        self.model.fit(np.atleast_2d(np.asarray(X, float)), np.asarray(y, float))
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu, sd = self.model.predict(np.atleast_2d(np.asarray(X, float)), return_std=True)
        return mu, sd ** 2
