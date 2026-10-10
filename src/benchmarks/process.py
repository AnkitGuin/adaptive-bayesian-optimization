"""Real-world-inspired application: engineering process under changing
operating conditions (spec §34).

A *controlled benchmark inspired by process optimization* (not a claim about
a real plant): we "operate" a process with three set-points

    x1 = temperature  T in [150, 250] degC
    x2 = pressure     P in [1, 5] bar
    x3 = stirring speed S in [20, 80] rpm

and want to **maximize** the (simulated) process efficiency

    eff(T, P, S; t) =
        10 + 82 * exp(-((T - T_opt(t)) / 30)^2)
                  * exp(-((P - P_opt(t)) / 1.8)^2)
                  * exp(-((S - S_opt)     / 18)^2)

The environment changes in two controlled ways:

* **equipment degradation**: the temperature optimum drifts
  ``T_opt(t) = 190 + 0.15 t`` (slow, one-directional);
* **demand cycle**: the pressure optimum oscillates
  ``P_opt(t) = 2.6 + 0.8 sin(2 pi t / 60)``.

The optimum is always interior, so ``x*_t = (T_opt(t), P_opt(t), 45)`` and
``f_opt(t) = 92`` exactly.
"""

from __future__ import annotations

import numpy as np

from ..base import SearchSpace

__all__ = ["ProcessSimulator"]


class ProcessSimulator:
    name = "process"
    dim = 3

    def __init__(self, spec, horizon: int, rng=None,
                 t0: float = 190.0, drift_per_step: float = 0.15,
                 p_center: float = 2.6, p_osc: float = 0.8, period: int = 60,
                 s_opt: float = 45.0) -> None:
        self.spec = spec
        self.horizon = horizon
        self.space = SearchSpace(lower=[150.0, 1.0, 20.0], upper=[250.0, 5.0, 80.0])
        self.drift_per_step = drift_per_step
        self.period = int(period)
        t = np.arange(horizon + 1)
        # Keep the physical (unconstrained) centres separate from the
        # constrained optimum.  For the default horizon they are interior,
        # but a longer horizon or different parameters may move a centre
        # outside the search box.
        self.t_opt_path = t0 + drift_per_step * t
        self.p_opt_path = p_center + p_osc * np.sin(2.0 * np.pi * t / max(self.period, 1))
        self.s_opt = s_opt

    def evaluate_true(self, X, t: int) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        g1 = np.exp(-((X[:, 0] - self.t_opt_path[t]) / 30.0) ** 2)
        g2 = np.exp(-((X[:, 1] - self.p_opt_path[t]) / 1.8) ** 2)
        g3 = np.exp(-((X[:, 2] - self.s_opt) / 18.0) ** 2)
        return 10.0 + 82.0 * g1 * g2 * g3

    def optimum(self, t: int) -> tuple[np.ndarray, float]:
        mu = np.asarray([self.t_opt_path[t], self.p_opt_path[t], self.s_opt])
        mu = np.clip(mu, self.space.lower, self.space.upper)
        f_opt = float(self.evaluate_true(mu.reshape(1, -1), t)[0])
        return mu, f_opt

    def optima_set(self, t: int) -> np.ndarray:
        return self.optimum(t)[0][None, :]

    def base_noise_std(self, t: int) -> float | None:
        return None  # noise level is fixed by the experiment

    def describe(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "domain": self.space.bounds,
            "t_opt_path": self.t_opt_path.tolist(),
            "p_opt_path": self.p_opt_path.tolist(),
            "s_opt": self.s_opt,
        }
