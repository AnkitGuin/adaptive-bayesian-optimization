"""Shared numerics for acquisition functions.

All acquisition functions are vectorised over candidate points and must never
emit NaN / Inf, in particular in the zero-variance limit (spec §11).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["safe_sigma", "z_of", "SIGMA_EPS"]

SIGMA_EPS = 1e-12


def safe_sigma(sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(sigma_safe, tiny)`` where ``tiny`` marks sigma ~= 0."""
    sigma = np.asarray(sigma, dtype=float)
    tiny = sigma < SIGMA_EPS
    return np.where(tiny, SIGMA_EPS, sigma), tiny


def z_of(mu: np.ndarray, sigma: np.ndarray, f_best: float, xi: float) -> np.ndarray:
    sigma_safe, _ = safe_sigma(sigma)
    return (np.asarray(mu, dtype=float) - f_best - xi) / sigma_safe
