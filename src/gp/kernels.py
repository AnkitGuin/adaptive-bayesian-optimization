"""Stationary GP kernels implemented directly with numpy.

Only three kernels are used (spec §9): Matérn 5/2 (default), Matérn 3/2 and
RBF.  Keeping the kernels in our own code means the *same* kernel object can
be used by the custom weighted GP and by the scikit-learn reference GP, and
the lengthscale adaptation of the adaptive GP stays fully transparent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_SQRT3 = float(np.sqrt(3.0))
_SQRT5 = float(np.sqrt(5.0))


def _as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X


def sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Numerically stable squared Euclidean distances, shape ``(n, m)``."""
    A = _as_2d(A)
    B = _as_2d(B)
    d2 = (
        np.sum(A * A, axis=1)[:, None]
        + np.sum(B * B, axis=1)[None, :]
        - 2.0 * (A @ B.T)
    )
    return np.maximum(d2, 0.0)


@dataclass
class Kernel:
    """Isotropic stationary kernel with a single lengthscale."""

    kind: str = "matern52"
    lengthscale: float = 0.3
    signal_variance: float = 1.0

    def __post_init__(self) -> None:
        self.kind = str(self.kind).lower()
        if self.kind not in {"matern52", "matern32", "rbf"}:
            raise ValueError(f"unsupported kernel: {self.kind}")
        self.lengthscale = float(self.lengthscale)
        self.signal_variance = float(self.signal_variance)
        if self.lengthscale <= 0:
            raise ValueError("lengthscale must be positive")
        if self.signal_variance <= 0:
            raise ValueError("signal_variance must be positive")

    # ------------------------------------------------------------------- api
    def scaled_distance(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return np.sqrt(sqdist(A, B)) / self.lengthscale

    def __call__(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return self.k(A, B)

    def k(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        r = self.scaled_distance(A, B)
        s2 = self.signal_variance
        if self.kind == "rbf":
            return s2 * np.exp(-0.5 * r * r)
        if self.kind == "matern32":
            t = _SQRT3 * r
            return s2 * (1.0 + t) * np.exp(-t)
        # matern52
        t = _SQRT5 * r
        return s2 * (1.0 + t + t * t / 3.0) * np.exp(-t)

    def diag(self, A: np.ndarray) -> np.ndarray:
        return np.full(_as_2d(A).shape[0], self.signal_variance, dtype=float)

    def with_lengthscale(self, lengthscale: float) -> "Kernel":
        return Kernel(kind=self.kind, lengthscale=float(lengthscale),
                      signal_variance=self.signal_variance)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "lengthscale": self.lengthscale,
                "signal_variance": self.signal_variance}


def make_kernel(kind: str, lengthscale: float, signal_variance: float = 1.0) -> Kernel:
    return Kernel(kind=kind, lengthscale=lengthscale, signal_variance=signal_variance)
