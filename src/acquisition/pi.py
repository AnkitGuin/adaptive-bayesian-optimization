"""Probability of Improvement (PI) for *maximization*.

    PI(x) = Phi(Z),   Z = (mu(x) - f_best - xi) / sigma(x)

PI is used as a baseline acquisition that is known to become overly
exploitative (spec §11).

Zero-variance limit: PI -> 1{mu > f_best + xi}.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .base import safe_sigma

__all__ = ["probability_of_improvement"]


def probability_of_improvement(mu: np.ndarray, sigma: np.ndarray, f_best: float,
                               xi: float = 0.01) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma_safe, tiny = safe_sigma(sigma)
    z = (mu - f_best - xi) / sigma_safe
    pi = norm.cdf(z)
    pi = np.where(tiny, (mu > f_best + xi).astype(float), pi)
    return np.clip(pi, 0.0, 1.0)
