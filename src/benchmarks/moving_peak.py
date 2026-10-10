"""Moving Gaussian-peak benchmarks (1D and 2D).

Function B of spec §7 (1D) and Function D (2D):

    f_t(x) = exp(-||x - mu_t||^2 / (2 s^2))

with a known optimum ``x*_t = mu_t`` and ``f_opt(t) = 1``.
"""

from __future__ import annotations

import numpy as np

from ..base import SearchSpace
from .scenarios import ScenarioSpec, build_state_path

__all__ = ["MovingGaussianPeak", "MovingGaussianPeak2D"]


class MovingGaussianPeak:
    """1D moving peak on [0, 1]."""

    name = "peak"
    dim = 1
    peak_sigma = 0.12

    def __init__(self, spec: ScenarioSpec, rng: np.random.Generator,
                 start: float = 0.3, bounds: tuple[float, float] = (0.1, 0.9),
                 sigma: float = 0.12) -> None:
        self.spec = spec
        self.horizon = spec.horizon
        self.peak_sigma = sigma
        self.space = SearchSpace(lower=[0.0], upper=[1.0])
        self.mu_path = build_state_path(
            spec, start= start, horizon=spec.horizon, dim=1,
            bounds=bounds, rng=rng,
        )[:, 0]

    # ------------------------------------------------------------------ api
    def evaluate_true(self, X, t: int) -> np.ndarray:
        X = np.atleast_1d(np.asarray(X, dtype=float))
        s = self.peak_sigma
        return np.exp(-((X - self.mu_path[t]) ** 2) / (2.0 * s * s))

    def optimum(self, t: int) -> tuple[np.ndarray, float]:
        mu = float(self.mu_path[t])
        return np.asarray([mu]), 1.0

    def optima_set(self, t: int) -> np.ndarray:
        return np.asarray([[self.mu_path[t]]])

    def base_noise_std(self, t: int) -> float | None:
        if self.spec.type == "noise_shift":
            return self.spec.noise_from if t < self.spec.t_change else self.spec.noise_to
        return None

    def describe(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "domain": self.space.bounds,
            "peak_sigma": self.peak_sigma,
            "scenario": self.spec.describe(),
            "mu_path": self.mu_path.tolist(),
        }


class MovingGaussianPeak2D:
    """2D moving peak on [0, 1]^2."""

    name = "peak2d"
    dim = 2
    peak_sigma = 0.15

    def __init__(self, spec: ScenarioSpec, rng: np.random.Generator,
                 start: tuple = (0.35, 0.35),
                 bounds: tuple[float, float] = (0.2, 0.8),
                 sigma: float = 0.15) -> None:
        self.spec = spec
        self.horizon = spec.horizon
        self.peak_sigma = sigma
        self.space = SearchSpace(lower=[0.0, 0.0], upper=[1.0, 1.0])
        self.mu_path = build_state_path(
            spec, start=start, horizon=spec.horizon, dim=2,
            bounds=bounds, rng=rng,
        )

    def evaluate_true(self, X, t: int) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        d2 = np.sum((X - self.mu_path[t]) ** 2, axis=1)
        return np.exp(-d2 / (2.0 * self.peak_sigma ** 2))

    def optimum(self, t: int) -> tuple[np.ndarray, float]:
        mu = self.mu_path[t].copy()
        return mu, 1.0

    def optima_set(self, t: int) -> np.ndarray:
        return self.mu_path[t][None, :].copy()

    def base_noise_std(self, t: int) -> float | None:
        if self.spec.type == "noise_shift":
            return self.spec.noise_from if t < self.spec.t_change else self.spec.noise_to
        return None

    def describe(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "domain": self.space.bounds,
            "peak_sigma": self.peak_sigma,
            "scenario": self.spec.describe(),
            "mu_path": self.mu_path.tolist(),
        }
