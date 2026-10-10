"""Acquisition-function registry.

``make_acquisition(name, params)`` returns a callable with signature
``fn(mu, sigma, f_best, **params) -> values``.
"""

from __future__ import annotations

from .ei import expected_improvement
from .mean import posterior_mean_acquisition
from .pi import probability_of_improvement
from .ucb import upper_confidence_bound

__all__ = [
    "expected_improvement",
    "upper_confidence_bound",
    "probability_of_improvement",
    "posterior_mean_acquisition",
    "make_acquisition",
    "ACQUISITION_NAMES",
]

ACQUISITION_NAMES = ("ei", "ucb", "pi", "mean")


def make_acquisition(name: str, params: dict | None = None):
    params = dict(params or {})
    key = str(name).lower()
    if key == "ei":
        xi = float(params.get("xi", 0.01))

        def fn(mu, sigma, f_best, **_):
            return expected_improvement(mu, sigma, f_best, xi=xi)

        return fn
    if key == "ucb":
        beta = float(params.get("beta", 2.0))

        def fn(mu, sigma, f_best=None, **_):
            return upper_confidence_bound(mu, sigma, beta=beta)

        return fn
    if key == "pi":
        xi = float(params.get("xi", 0.01))

        def fn(mu, sigma, f_best, **_):
            return probability_of_improvement(mu, sigma, f_best, xi=xi)

        return fn
    if key == "mean":
        return posterior_mean_acquisition
    raise ValueError(f"unknown acquisition function: {name}")
