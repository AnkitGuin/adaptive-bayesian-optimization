"""Non-stationarity scenario definitions (spec §25).

A scenario controls the *state* of a benchmark over the horizon
(optimum location, phase, peak amplitudes, noise level).  The state path is
precomputed **once per (study, seed)** before any optimisation starts, which
guarantees that every algorithm compared inside one experiment sees exactly
the same objective trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

__all__ = ["ScenarioSpec", "SCENARIO_TYPES"]

SCENARIO_TYPES = (
    "stationary",      # A: f_t = f
    "slow",            # B: state drifts linearly,  s_t = s_0 + v t
    "abrupt",          # C: s_t = s_A  (t < T) | s_B (t >= T)
    "random_walk",     # D: s_{t+1} = clip(s_t + eta_t),  eta ~ N(0, sigma_drift^2)
    "noise_shift",     # E: fixed function, sigma_noise changes at T
    "switch",          # F: multimodal amplitude hand-over
)


@dataclass
class ScenarioSpec:
    type: str = "stationary"
    horizon: int = 100
    t_change: int = 50
    # slow
    v: float = 0.0
    phase_speed: float = 0.05
    # abrupt
    from_state: Union[float, tuple] = 0.3
    to_state: Union[float, tuple] = 0.7
    # random walk
    sigma_drift: float = 0.01
    # noise shift (absolute noise std before / after t_change)
    noise_from: float = 0.02
    noise_to: float = 0.15
    # multimodal amplitude switches
    t_switch1: int = 40
    t_switch2: int = 80
    transition: int = 4

    def __post_init__(self) -> None:
        if self.type not in SCENARIO_TYPES:
            raise ValueError(f"unknown scenario type: {self.type}")
        if self.type in ("abrupt", "noise_shift"):
            if not (0 <= self.t_change <= self.horizon):
                raise ValueError("t_change must be within [0, horizon]")

    def describe(self) -> dict:
        return {
            "type": self.type,
            "horizon": self.horizon,
            "t_change": self.t_change,
            "v": self.v,
            "phase_speed": self.phase_speed,
            "from_state": (list(self.from_state) if isinstance(self.from_state, (tuple, list))
                           else self.from_state),
            "to_state": (list(self.to_state) if isinstance(self.to_state, (tuple, list))
                         else self.to_state),
            "sigma_drift": self.sigma_drift,
            "noise_from": self.noise_from,
            "noise_to": self.noise_to,
            "t_switch1": self.t_switch1,
            "t_switch2": self.t_switch2,
        }


def _as_array(state: Union[float, tuple, list], n: int, dim: int) -> np.ndarray:
    if isinstance(state, (tuple, list)):
        arr = np.asarray(list(state), dtype=float)
        assert arr.size == dim, f"state has {arr.size} entries, expected {dim}"
    else:
        arr = np.full(dim, float(state), dtype=float)
    return np.broadcast_to(arr, (n, dim)).copy()


def build_state_path(
    spec: ScenarioSpec,
    start: Union[float, tuple],
    horizon: int,
    dim: int,
    bounds: Optional[tuple[float, float]] = None,
    rng: Optional[np.random.Generator] = None,
    kind: str = "state",
) -> np.ndarray:
    """Precompute the state path ``s_t`` for t = 0..horizon.

    Parameters
    ----------
    kind:
        ``"state"`` uses ``v`` for the slow scenario; ``"phase"`` uses
        ``phase_speed`` (for the sinusoidal benchmark).
    """
    n = horizon + 1
    s0 = np.asarray(
        [float(v) for v in (start if isinstance(start, (tuple, list)) else [start] * dim)],
        dtype=float)

    if spec.type == "stationary":
        path = np.broadcast_to(s0, (n, dim)).copy()
    elif spec.type == "slow":
        speed = spec.phase_speed if kind == "phase" else spec.v
        path = s0[None, :] + np.arange(n)[:, None] * float(speed)
    elif spec.type == "abrupt":
        a = _as_array(spec.from_state, n, dim)
        b = _as_array(spec.to_state, n, dim)
        path = np.where(np.arange(n)[:, None] < spec.t_change, a, b)
    elif spec.type == "random_walk":
        path = np.empty((n, dim), dtype=float)
        s = s0.copy()
        for t in range(n):
            path[t] = s
            step = rng.normal(0.0, spec.sigma_drift, size=dim) \
                if rng is not None else np.zeros(dim)
            s = s + np.asarray(step, dtype=float)
        if bounds is not None:
            path = np.clip(path, bounds[0], bounds[1])
    elif spec.type in ("noise_shift", "switch"):
        # state is fixed for these scenarios
        path = np.broadcast_to(s0, (n, dim)).copy()
    else:  # pragma: no cover
        raise ValueError(spec.type)

    if bounds is not None:
        path = np.clip(path, bounds[0], bounds[1])
    return path
