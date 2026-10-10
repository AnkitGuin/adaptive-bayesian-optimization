"""The environment the optimisers query.

The environment owns the *only* observation-noise RNG of an experiment run.
Because every method of one (study, seed) evaluates its chosen ``x_t`` at the
same iteration index ``t`` using the same RNG, all methods receive **the same
noise draw sequence** (spec §22).  The noise-free value ``f_true`` and the
true optimum are returned alongside so that metrics can be computed without
contamination (spec §33).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["EvalResult", "Environment"]


@dataclass
class EvalResult:
    y: float          # noisy observation (what the optimiser sees)
    f_true: float     # noise-free objective value (metrics only)
    f_opt: float      # noise-free optimum value at time t
    x_opt: np.ndarray # noise-free optimum location at time t


class Environment:
    def __init__(self, benchmark, base_noise_std: float,
                 rng: np.random.Generator) -> None:
        self.benchmark = benchmark
        self.base_noise_std = float(base_noise_std)
        self.rng = rng
        self.n_evals = 0

    def evaluate(self, x, t: int) -> EvalResult:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        f_true = float(np.atleast_1d(self.benchmark.evaluate_true(x, t))[0])
        x_opt, f_opt = self.benchmark.optimum(t)
        noise_std = self.benchmark.base_noise_std(t)
        if noise_std is None:
            noise_std = self.base_noise_std
        eps = float(self.rng.normal(0.0, noise_std))
        self.n_evals += 1
        return EvalResult(y=f_true + eps, f_true=f_true, f_opt=float(f_opt),
                          x_opt=np.asarray(x_opt, dtype=float))
