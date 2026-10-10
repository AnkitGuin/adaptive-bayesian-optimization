"""Pure exploitation: ``a(x) = mu(x)`` (spec §24).

Useful baseline: shows what happens when exploration is removed entirely.
"""

from __future__ import annotations

import numpy as np

__all__ = ["posterior_mean_acquisition"]


def posterior_mean_acquisition(mu: np.ndarray, sigma: np.ndarray, f_best: float | None = None,
                               **_: object) -> np.ndarray:
    return np.asarray(mu, dtype=float)
