"""Page–Hinkley change-point detector for the standardized-residual stream.

At every sequential iteration the optimiser obtains a standardized GP
residual

    r_t = |y_t - mu_t(x_t)| / (sigma_pred,t(x_t) + eps)

where ``sigma_pred`` includes both latent GP uncertainty and the known
observation-noise standard deviation.  This is the appropriate scale for a
noisy observation; dividing only by latent uncertainty would create
systematic false alarms as measurement noise increases.

which is approximately |N(0, 1)| when the surrogate is well calibrated and
stays *elevated* while the environment has moved and the surrogate lags.

This implements the classical Page–Hinkley statistic with a **post-detection
restart**:

    M_t = (1/t) * sum_{i <= t} r_i          (mean since the last reset)
    G_t = max(0, G_{t-1} + (r_t - M_t - delta))

Drift is declared when ``G_t > threshold``; on detection the statistic and
the mean accumulator reset, re-baselining on the post-change residuals, so
one change event produces one detection.

Design note: a *windowed* running mean would adapt to a sustained level
shift (the pre-change data falls out of the window) and therefore could
never fire on the exact signal we care about.  The global (post-reset) mean
lags the level shift, which is what makes the statistic grow.  The restart
keeps the detector alive for *subsequent* changes.
"""

from __future__ import annotations

import numpy as np

__all__ = ["PageHinkley"]


class PageHinkley:
    def __init__(self, delta: float = 0.5, threshold: float = 25.0,
                 min_samples: int = 10) -> None:
        if not np.isfinite(delta) or delta < 0.0:
            raise ValueError("delta must be finite and non-negative")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if min_samples < 2:
            raise ValueError("min_samples must be >= 2")
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.min_samples = int(min_samples)
        self._sum = 0.0
        self._n = 0
        self._G = 0.0
        self._last_detection = -1

    # ------------------------------------------------------------------ api
    def update(self, r: float) -> bool:
        """Feed one residual; return ``True`` iff drift was detected."""
        detected = False
        if self._n >= self.min_samples:
            baseline = self._sum / self._n
            self._G = max(0.0, self._G + (float(r) - baseline - self.delta))
            if self._G > self.threshold:
                detected = True
                self._last_detection = self._n
                # restart: re-baseline on post-change residuals
                self._G = 0.0
                self._sum = 0.0
                self._n = 0
        self._sum += float(r)
        self._n += 1
        return detected

    def reset(self) -> None:
        self._G = 0.0
        self._sum = 0.0
        self._n = 0
        self._last_detection = -1

    # -------------------------------------------------------------- state
    @property
    def statistic(self) -> float:
        return self._G

    @property
    def n_updates(self) -> int:
        return self._n

    @property
    def last_detection(self) -> int:
        return self._last_detection

    def state(self) -> dict:
        return {
            "G": self._G,
            "n_updates": self._n,
            "last_detection": self._last_detection,
            "delta": self.delta,
            "threshold": self.threshold,
        }
