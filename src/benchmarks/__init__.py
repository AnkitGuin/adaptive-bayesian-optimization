"""Synthetic benchmarks and the shared environment.

``make_benchmark(name, spec, rng, **kw)`` instantiates the benchmark; the
scenario state path is precomputed from ``rng`` (seeded per (study, seed),
see ``src/utils/seeds.py``) so all methods of one run see the same moving
objective.
"""

from __future__ import annotations

import numpy as np

from .environment import Environment, EvalResult
from .moving_peak import MovingGaussianPeak, MovingGaussianPeak2D
from .multimodal import Multimodal
from .process import ProcessSimulator
from .scenarios import SCENARIO_TYPES, ScenarioSpec, build_state_path
from .sinusoidal import Sinusoidal

__all__ = [
    "Environment",
    "EvalResult",
    "MovingGaussianPeak",
    "MovingGaussianPeak2D",
    "Multimodal",
    "ProcessSimulator",
    "Sinusoidal",
    "ScenarioSpec",
    "build_state_path",
    "SCENARIO_TYPES",
    "make_benchmark",
    "BENCHMARK_NAMES",
]

BENCHMARK_NAMES = ("peak", "peak2d", "multimodal", "sinusoidal", "process")


def make_benchmark(name: str, spec: ScenarioSpec, rng: np.random.Generator,
                   **kw) -> object:
    name = str(name).lower()
    if name == "peak":
        return MovingGaussianPeak(spec, rng, **kw)
    if name == "peak2d":
        return MovingGaussianPeak2D(spec, rng, **kw)
    if name == "multimodal":
        return Multimodal(spec, rng, **kw)
    if name == "sinusoidal":
        return Sinusoidal(spec, rng, **kw)
    if name == "process":
        return ProcessSimulator(spec, spec.horizon, rng, **kw)
    raise ValueError(f"unknown benchmark: {name}")
