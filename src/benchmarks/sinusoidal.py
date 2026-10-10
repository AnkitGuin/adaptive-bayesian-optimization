"""1D sinusoidal benchmark (Function A of spec §7).

    f_t(x) = sin(x + phi_t),      x in [0, 4*pi]

The phase ``phi_t`` drifts (slow scenario) or follows a random walk.  The
phase is an angular state and is therefore not clipped to the x-domain; the
objective itself is periodic in ``phi_t``.  There are normally two equivalent
global maxima per horizon step (period ``2 pi``, domain = two periods), both
with value 1; :meth:`optima_set` returns them, which is used by the
tracking-error metric.
"""

from __future__ import annotations

import numpy as np

from ..base import SearchSpace
from .scenarios import ScenarioSpec, build_state_path

__all__ = ["Sinusoidal"]


class Sinusoidal:
    name = "sinusoidal"
    dim = 1

    def __init__(self, spec: ScenarioSpec, rng: np.random.Generator,
                 start_phase: float = 0.0) -> None:
        self.spec = spec
        self.horizon = spec.horizon
        self.space = SearchSpace(lower=[0.0], upper=[4.0 * np.pi])
        self.phi_path = build_state_path(
            spec, start=start_phase, horizon=spec.horizon, dim=1,
            bounds=None, rng=rng, kind="phase",
        )[:, 0]

    def evaluate_true(self, X, t: int) -> np.ndarray:
        X = np.atleast_1d(np.asarray(X, dtype=float))
        return np.sin(X + self.phi_path[t])

    def _maxima(self, phi: float) -> np.ndarray:
        base = (np.pi / 2.0 - phi) % (2.0 * np.pi)
        out = [base]
        if base + 2.0 * np.pi <= 4.0 * np.pi + 1e-9:
            out.append(base + 2.0 * np.pi)
        return np.asarray(out, dtype=float)

    def optimum(self, t: int) -> tuple[np.ndarray, float]:
        xs = self._maxima(float(self.phi_path[t]))
        return np.asarray([xs[0]]), 1.0

    def optima_set(self, t: int) -> np.ndarray:
        return self._maxima(float(self.phi_path[t])).reshape(-1, 1)

    def base_noise_std(self, t: int) -> float | None:
        if self.spec.type == "noise_shift":
            return self.spec.noise_from if t < self.spec.t_change else self.spec.noise_to
        return None

    def describe(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "domain": self.space.bounds,
            "scenario": self.spec.describe(),
            "phi_path": self.phi_path.tolist(),
        }
