"""Core data containers and search-space utilities.

The project keeps a strict separation between

* ``y``  – the **noisy** observation the optimiser is allowed to see, and
* ``f``  – the **noise-free** objective value used for regret / tracking.

``Observation`` stores both, so downstream analysis can never accidentally mix
them up (spec §33).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Observation:
    """A single query of the environment.

    Attributes
    ----------
    t:
        Iteration index (0-based).  Used as the "time" of the observation for
        time-decay weighting.
    x:
        Decision vector, shape ``(d,)``.
    y:
        Noisy observation ``f_t(x) + eps_t`` -- the only quantity an optimiser
        is allowed to use.
    f_true:
        Noise-free objective value ``f_t(x)`` -- used for metrics only.
    f_opt:
        Noise-free optimum value ``f_t(x*_t)`` at this iteration.
    x_opt:
        Noise-free optimum location ``x*_t``.
    is_initial:
        ``True`` for points coming from the initial design.
    """

    t: int
    x: np.ndarray
    y: float
    f_true: float
    f_opt: float
    x_opt: np.ndarray
    is_initial: bool = False
    mu_pred: Optional[float] = None
    sigma_pred: Optional[float] = None
    residual: Optional[float] = None
    acquisition_value: Optional[float] = None
    beta_used: Optional[float] = None
    half_life_used: Optional[float] = None
    lengthscale_used: Optional[float] = None
    drift_detected: bool = False

    @property
    def regret(self) -> float:
        """Instantaneous regret computed from *noise-free* values."""
        return float(self.f_opt - self.f_true)


@dataclass
class Trajectory:
    """Full record of one optimisation run (spec §42 reproducibility)."""

    observations: list[Observation] = field(default_factory=list)
    drift_events: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ array
    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.observations)

    def x(self) -> np.ndarray:
        return np.asarray([o.x for o in self.observations], dtype=float)

    def y(self) -> np.ndarray:
        return np.asarray([o.y for o in self.observations], dtype=float)

    def f_true(self) -> np.ndarray:
        return np.asarray([o.f_true for o in self.observations], dtype=float)

    def f_opt(self) -> np.ndarray:
        return np.asarray([o.f_opt for o in self.observations], dtype=float)

    def x_opt(self) -> np.ndarray:
        return np.asarray([o.x_opt for o in self.observations], dtype=float)

    def t(self) -> np.ndarray:
        return np.asarray([o.t for o in self.observations], dtype=int)

    def residual(self) -> np.ndarray:
        return np.asarray(
            [np.nan if o.residual is None else o.residual for o in self.observations], dtype=float
        )

    def is_initial(self) -> np.ndarray:
        return np.asarray([o.is_initial for o in self.observations], dtype=bool)


@dataclass
class SearchSpace:
    """Box-constrained continuous search space ``[lower, upper]^d``."""

    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        self.lower = np.atleast_1d(np.asarray(self.lower, dtype=float))
        self.upper = np.atleast_1d(np.asarray(self.upper, dtype=float))
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper must have the same shape")
        if np.any(self.upper <= self.lower):
            raise ValueError("upper must be strictly greater than lower")

    # ------------------------------------------------------------------ props
    @property
    def dim(self) -> int:
        return int(self.lower.size)

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return list(zip(self.lower.tolist(), self.upper.tolist()))

    @property
    def range(self) -> np.ndarray:
        return self.upper - self.lower

    def center(self) -> np.ndarray:
        return 0.5 * (self.lower + self.upper)

    # ------------------------------------------------------------------- ops
    def clip(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.clip(X, self.lower, self.upper)

    def contains(self, X: np.ndarray, tol: float = 1e-9) -> bool:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return bool(
            np.all(X >= self.lower - tol) and np.all(X <= self.upper + tol)
        )

    def uniform(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.lower + rng.random((n, self.dim)) * self.range

    def grid(self, n_per_dim: int) -> np.ndarray:
        """Dense Cartesian grid (only sensible for d <= 3)."""
        axes = [np.linspace(lo, hi, n_per_dim) for lo, hi in zip(self.lower, self.upper)]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.stack([m.ravel() for m in mesh], axis=1)

    def to_dict(self) -> dict:
        return {"lower": self.lower.tolist(), "upper": self.upper.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "SearchSpace":
        return cls(lower=np.asarray(d["lower"], float), upper=np.asarray(d["upper"], float))

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"SearchSpace(lower={self.lower.tolist()}, upper={self.upper.tolist()})"


def initial_design(
    space: SearchSpace, n: int, rng: np.random.Generator, kind: str = "lhs"
) -> np.ndarray:
    """Generate an initial design without using any objective information.

    ``lhs`` (default, spec §13) uses stratified Latin Hypercube Sampling;
    ``random`` uses i.i.d. uniform samples.
    """
    kind = str(kind).lower()
    if n <= 0:
        raise ValueError("initial design size must be positive")
    if kind == "random":
        return space.uniform(n, rng)
    if kind == "lhs":
        return latin_hypercube(space, n, rng)
    raise ValueError(f"unknown initial design kind: {kind}")


def latin_hypercube(space: SearchSpace, n: int, rng: np.random.Generator) -> np.ndarray:
    d = space.dim
    unit = np.empty((n, d), dtype=float)
    for j in range(d):
        perm = rng.permutation(n)
        unit[:, j] = (perm + rng.random(n)) / n
    return space.lower + unit * space.range
