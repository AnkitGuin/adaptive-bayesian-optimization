"""Upper Confidence Bound (UCB) for *maximization*.

    UCB(x) = mu(x) + beta * sigma(x)

``beta`` is the explicit exploration knob.  In DA-BO this is the parameter
that gets boosted after drift detection (spec §20):
``beta_t = beta_base`` in stable periods and ``beta_t = beta_high`` during
the recovery window.
"""

from __future__ import annotations

import numpy as np

__all__ = ["upper_confidence_bound"]


def upper_confidence_bound(mu: np.ndarray, sigma: np.ndarray, beta: float = 2.0,
                           f_best: float | None = None) -> np.ndarray:
    return np.asarray(mu, dtype=float) + float(beta) * np.asarray(sigma, dtype=float)
