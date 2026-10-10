"""Deterministic seed management.

Every stochastic component of the project (initial design, observation noise,
random search, drift trajectories, bootstrap resampling) draws its randomness
from a seed that is a *deterministic function* of

    (study, scenario, method, seed_id, component)

so that two algorithms compared inside one experiment receive *identical*
initial designs and *identical* noise realisations, while different
experiments stay independent.  Nothing here depends on wall-clock time or on
global numpy state.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

import numpy as np

_MASK64 = (1 << 64) - 1


def derive_seed(*parts: object) -> int:
    """Derive a stable 64-bit seed from an arbitrary tuple of hashable parts.

    Stable across processes and python runs (unlike ``hash``), because it is
    based on SHA-256 of the ``repr`` of the parts.
    """
    payload = "|".join(repr(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & _MASK64


def rng_for(*parts: object) -> np.random.Generator:
    """Return a fresh ``np.random.Generator`` seeded from ``derive_seed``."""
    return np.random.default_rng(derive_seed(*parts))


def seed_sequence(base_seed: int, n: int) -> list[int]:
    """Deterministic list of ``n`` seeds derived from ``base_seed``."""
    return [derive_seed(base_seed, i) for i in range(n)]


def format_seed_list(seeds: Iterable[int]) -> Sequence[int]:
    return list(seeds)
