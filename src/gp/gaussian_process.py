"""Homoscedastic Gaussian-process regression implemented in pure numpy.

Design notes
------------
* The implementation is deliberately small and transparent: Cholesky-based
  GP regression with a *deterministic* increasing-jitter fallback, optional
  standardisation of the target, and posterior mean / standard deviation.
* scikit-learn's ``GaussianProcessRegressor`` is used **only** to cross-check
  this implementation in the test suite (see ``src/gp/sklearn_ref.py`` and
  ``tests/test_gp.py``).  Keeping the core GP in our own code is what makes
  the weighted (heteroscedastic) adaptive GP of ``adaptive_gp.py`` possible,
  because scikit-learn does not expose per-observation noise in a form that
  supports our per-iteration lengthscale adaptation.
"""

from __future__ import annotations

import numpy as np

from .kernels import Kernel

__all__ = ["GaussianProcess"]


class _PSDSolver:
    """A^{-1} b for a symmetric positive (semi)definite matrix.

    Prefers a Cholesky factor; falls back to an eigendecomposition with
    eigenvalue clipping so that the solver stays deterministic even for
    near-singular matrices.
    """

    def __init__(self) -> None:
        self._L: np.ndarray | None = None
        self._Q: np.ndarray | None = None
        self._inv_eig: np.ndarray | None = None

    @classmethod
    def from_matrix(cls, A: np.ndarray, base_jitter: float, max_jitter: float) -> "_PSDSolver":
        s = cls()
        n = A.shape[0]
        jitter = base_jitter
        for _ in range(8):
            try:
                s._L = np.linalg.cholesky(A + jitter * np.eye(n))
                return s
            except np.linalg.LinAlgError:
                jitter = min(jitter * 10.0, max_jitter)
        # Deterministic fallback: eigendecomposition with clipping.
        eigvals, Q = np.linalg.eigh(A)
        eigvals = np.maximum(eigvals, 1e-10)
        s._Q = Q
        s._inv_eig = 1.0 / eigvals
        return s

    def solve(self, b: np.ndarray) -> np.ndarray:
        b = np.asarray(b, dtype=float)
        if self._L is not None:
            lb = np.linalg.solve(self._L, b)
            return np.linalg.solve(self._L.T, lb)
        return (self._Q * self._inv_eig) @ (self._Q.T @ b)

    def solve_transpose(self, b: np.ndarray) -> np.ndarray:
        """A^{-T} b == A^{-1} b for symmetric A."""
        return self.solve(b)


class GaussianProcess:
    """Standard GP regression: ``y = f(x) + eps``, ``f ~ GP(0, k)``,
    ``eps ~ N(0, noise_var)`` (iid).

    Parameters
    ----------
    kernel:
        Kernel instance (Matérn 5/2, Matérn 3/2 or RBF).
    noise_var:
        Observation-noise variance assumed by the model.
    normalize:
        Standardise the target before fitting (zero mean / unit variance) and
        undo the standardisation at prediction time.
    """

    def __init__(
        self,
        kernel: Kernel,
        noise_var: float = 0.0,
        normalize: bool = True,
        jitter: float = 1e-8,
        max_jitter: float = 1e-4,
    ) -> None:
        self.kernel = kernel
        self.noise_var = float(noise_var)
        if not np.isfinite(self.noise_var) or self.noise_var < 0.0:
            raise ValueError("noise_var must be finite and non-negative")
        self.normalize = bool(normalize)
        self.jitter = float(jitter)
        self.max_jitter = float(max_jitter)
        if not np.isfinite(self.jitter) or self.jitter < 0.0:
            raise ValueError("jitter must be finite and non-negative")
        if not np.isfinite(self.max_jitter) or self.max_jitter < self.jitter:
            raise ValueError("max_jitter must be finite and >= jitter")
        self.X_train: np.ndarray | None = None
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def _standardise(self, y: np.ndarray) -> tuple[float, float, np.ndarray]:
        if self.normalize and y.size > 1:
            mu = float(y.mean())
            sd = float(y.std())
            if sd > 1e-12:
                return mu, sd, (y - mu) / sd
        return 0.0, 1.0, y.copy()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have inconsistent sizes")
        if X.shape[0] < 2:
            raise ValueError("at least two observations are required")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("X and y must contain only finite values")
        self.y_mu, self.y_sd, y_n = self._standardise(y)

        K = self.kernel(X, X)
        noise_scaled = self.noise_var / (self.y_sd * self.y_sd)
        A = K + noise_scaled * np.eye(X.shape[0])
        self._solver = _PSDSolver.from_matrix(A, self.jitter, self.max_jitter)
        self._alpha = self._solver.solve(y_n)
        self.X_train = X
        self._train_y = y
        self._fitted = True
        return self

    # --------------------------------------------------------------- predict
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("GaussianProcess is not fitted")

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior ``(mean, variance)`` at the points of ``X``."""
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.ndim != 2 or X.shape[1] != self.X_train.shape[1]:
            raise ValueError("X has the wrong number of features")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        k = self.kernel(X, self.X_train)  # (m, n)
        mu_n = k @ self._alpha
        # k_* A^{-1} k_*^T:  (k @ A^{-1}) = (A^{-1} k^T)^T  (A symmetric)
        cov_diag = self.kernel.diag(X) - np.sum(k * self._solver.solve(k.T).T, axis=1)
        mu = self.y_mu + self.y_sd * mu_n
        var = (self.y_sd * self.y_sd) * np.maximum(cov_diag, 0.0)
        return mu, var

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        _, var = self.predict(X)
        return np.sqrt(var)

    # ------------------------------------------------------------ bookkeeping
    def log_marginal_likelihood(self) -> float:
        """Log marginal likelihood of the (standardised) training data."""
        self._check_fitted()
        n = self.X_train.shape[0]
        y_n = (np.asarray(self._train_y, dtype=float) - self.y_mu) / self.y_sd
        # Reconstruct A from the solver is not stored; recompute K.
        K = self.kernel(self.X_train, self.X_train)
        A = K + (self.noise_var / (self.y_sd * self.y_sd)) * np.eye(n)
        try:
            L = np.linalg.cholesky(A + 1e-10 * np.eye(n))
            logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            eigvals, _ = np.linalg.eigh(A)
            logdet = float(np.sum(np.log(np.maximum(eigvals, 1e-10))))
        a = np.linalg.solve(A + 1e-10 * np.eye(n), y_n)
        ll = -0.5 * float(y_n @ a) - 0.5 * logdet - 0.5 * n * np.log(2.0 * np.pi)
        return ll

    def _set_train_y(self, y: np.ndarray) -> None:  # used by log-likelihood
        self._train_y = y
