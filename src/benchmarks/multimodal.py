"""Non-stationary 1D multimodal benchmark (Function C of spec §7, Scenario F).

    f_t(x) = sum_j a_j(t) * exp(-(x - m_j)^2 / w_j^2),   x in [0, 1]

Three peaks with fixed locations/widths and *time-varying amplitudes*.  Under
the ``switch`` scenario the dominant peak hands over:

    peak 1 (t < T1)  ->  peak 2 (T1 <= t < T2)  ->  peak 3 (t >= T2)

with a smooth transition of width ``spec.transition``.  The true optimum is
precomputed on a dense grid + golden-section refinement, *from the noise-free
function* (spec §8).
"""

from __future__ import annotations

import numpy as np

from ..base import SearchSpace

__all__ = ["Multimodal"]

_PEAK_CENTERS = (0.20, 0.50, 0.80)
_PEAK_WIDTHS = (0.08, 0.08, 0.08)


def _smoothstep(t: np.ndarray, t0: int, t1: int) -> np.ndarray:
    """0 before t0, 1 after t1, smooth in between."""
    u = (np.asarray(t, dtype=float) - t0) / max(t1 - t0, 1)
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


class Multimodal:
    name = "multimodal"
    dim = 1

    def __init__(self, spec, rng: np.random.Generator,
                 centers=_PEAK_CENTERS, widths=_PEAK_WIDTHS,
                 amps_stationary: tuple = (1.0, 0.55, 0.75)) -> None:
        self.spec = spec
        self.horizon = spec.horizon
        self.space = SearchSpace(lower=[0.0], upper=[1.0])
        self.centers = np.asarray(centers, dtype=float)
        self.widths = np.asarray(widths, dtype=float)

        T = spec.horizon
        t = np.arange(T + 1, dtype=float)
        if spec.type == "switch":
            u1 = _smoothstep(t, spec.t_switch1, spec.t_switch1 + spec.transition)
            u2 = _smoothstep(t, spec.t_switch2, spec.t_switch2 + spec.transition)
            a1 = 1.0 - u1 + u2 * 0.0            # peak 1: dominant -> subordinate
            a2 = 0.35 + u1 - u2                  # peak 2: subordinate -> dominant -> subordinate
            a3 = 0.55 + u2                       # peak 3: subordinate -> dominant
            # Keep these amplitudes explicit; the exact global optimum is
            # computed below from the resulting summed landscape.  Do not
            # confuse a component amplitude with the summed objective value.
            self.amps_path = np.stack([a1, a2, a3], axis=1)
        else:
            self.amps_path = np.broadcast_to(
                np.asarray(amps_stationary, dtype=float), (T + 1, 3)).copy()

        self._precompute_optima()

    # ------------------------------------------------------------- internal
    def _f_at(self, x: np.ndarray, t: int) -> float:
        a = self.amps_path[t]
        return float(np.sum(a * np.exp(-((x - self.centers) ** 2) / self.widths ** 2)))

    def _precompute_optima(self) -> None:
        T = self.horizon
        xs = np.linspace(0.0, 1.0, 20001)
        # G[j, x] for peak j; amps_path[t, j].  F[x, t] = sum_j G[j, x]*amps[t, j]
        G = np.exp(-((xs[None, :] - self.centers[:, None]) ** 2) / self.widths[:, None] ** 2)
        F = G.T @ self.amps_path.T  # (20001, T+1)
        idx = F.argmax(axis=0)
        x_opt = np.empty(T + 1)
        f_opt = np.empty(T + 1)
        step = 1.0 / 20000
        for t in range(T + 1):
            x0 = xs[idx[t]]
            lo, hi = max(0.0, x0 - 3 * step), min(1.0, x0 + 3 * step)
            invphi = 0.5 * (np.sqrt(5.0) - 1.0)
            a_, b_ = lo, hi
            c = b_ - invphi * (b_ - a_)
            dd = a_ + invphi * (b_ - a_)
            fc, fd = self._f_at(c, t), self._f_at(dd, t)
            for _ in range(60):
                if fc > fd:
                    b_, dd, fd = c, c, fc
                    c = b_ - invphi * (b_ - a_)
                    fc = self._f_at(c, t)
                else:
                    a_, c, fc = dd, dd, fd
                    dd = a_ + invphi * (b_ - a_)
                    fd = self._f_at(dd, t)
            x_star = 0.5 * (a_ + b_)
            x_opt[t] = x_star
            f_opt[t] = self._f_at(x_star, t)
        self.x_opt_path = x_opt
        self.f_opt_path = f_opt

    # ------------------------------------------------------------------ api
    def evaluate_true(self, X, t: int) -> np.ndarray:
        X = np.atleast_1d(np.asarray(X, dtype=float))
        a = self.amps_path[t]
        d2 = ((X[None, :] - self.centers[:, None]) ** 2) / self.widths[:, None] ** 2
        return a @ np.exp(-d2)  # (m,)

    def optimum(self, t: int) -> tuple[np.ndarray, float]:
        return np.asarray([self.x_opt_path[t]]), float(self.f_opt_path[t])

    def optima_set(self, t: int) -> np.ndarray:
        return np.asarray([[self.x_opt_path[t]]])

    def base_noise_std(self, t: int) -> float | None:
        if self.spec.type == "noise_shift":
            return self.spec.noise_from if t < self.spec.t_change else self.spec.noise_to
        return None

    def describe(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "domain": self.space.bounds,
            "centers": self.centers.tolist(),
            "widths": self.widths.tolist(),
            "scenario": self.spec.describe(),
            "amps_path": self.amps_path.tolist(),
        }
