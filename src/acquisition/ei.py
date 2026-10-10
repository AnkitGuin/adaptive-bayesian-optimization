"""Expected Improvement (EI) for *maximization*.

    EI(x) = (mu(x) - f_best - xi) * Phi(Z) + sigma(x) * phi(Z)
    Z     = (mu(x) - f_best - xi) / sigma(x)

Zero-variance limit: EI -> max(mu - f_best - xi, 0).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .base import safe_sigma

__all__ = ["expected_improvement"]


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, f_best: float,
                         xi: float = 0.01) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    imp = mu - f_best - xi
    sigma_safe, tiny = safe_sigma(sigma)
    z = imp / sigma_safe
    ei = imp * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(tiny, np.maximum(imp, 0.0), ei)
    return np.maximum(ei, 0.0)
