"""Method factory: canonical configurations for every algorithm in the study.

The *single* loop in :mod:`bayesian_optimizer` is reused; these factories only
switch configuration.  This is what guarantees the fair-comparison requirement
(spec §22): same loop, same initial design, same environment/noise stream,
same budget.

Canonical methods
-----------------
``random_search``   : uniform random points (non-GP baseline, spec §23)
``static_ei``       : static GP + EI
``static_ucb``      : static GP + UCB(beta_base)
``static_pi``       : static GP + PI
``exploitation``    : static GP + posterior mean (pure exploitation, spec §24)
``abo``             : time-decayed GP + UCB (the "Adaptive BO" of spec §15)
``dabo``            : ABO + drift detection + exploration boost + temporary
                      increased forgetting + memory shortening (DA-BO)

Ablation variants of DA-BO are produced by :func:`dabo_ablation`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

from ..base import SearchSpace
from .bayesian_optimizer import OptimizerConfig

__all__ = ["StudyDefaults", "method_config", "dabo_ablation", "METHOD_NAMES"]


@dataclass
class StudyDefaults:
    """Per-study constants shared by every method (fair comparison)."""

    space: SearchSpace
    budget: int = 100
    n_initial: int = 10
    initial_design: str = "lhs"
    kernel: str = "matern52"
    lengthscale: float = 0.25
    noise_std: float = 0.05
    xi: float = 0.01
    beta_base: float = 2.0
    # DA-BO hyperparameters (frozen after the development study, spec §30)
    half_life: float = 32.0
    beta_high: float = 6.0
    recovery_len: int = 20
    ph_delta: float = 0.5
    ph_threshold: float = 25.0
    boost_forgetting: float = 2.0
    ls_shrink: float = 0.5
    # acquisition maximisation
    n_grid_1d: int = 401
    n_grid_d: int = 21
    n_restarts: int = 5


METHOD_NAMES = (
    "random_search", "static_ei", "static_ucb", "static_pi",
    "exploitation", "abo", "dabo",
)


def _base(d: StudyDefaults, name: str, **overrides) -> OptimizerConfig:
    kwargs = dict(
        name=name,
        space=d.space,
        budget=d.budget,
        n_initial=d.n_initial,
        initial_design=d.initial_design,
        kernel=d.kernel,
        lengthscale=d.lengthscale,
        noise_std=d.noise_std,
        acquisition="ucb",
        acquisition_params={"beta": d.beta_base},
        use_gp=True,
        half_life=None,
        drift_detection=False,
        ph_delta=d.ph_delta,
        ph_threshold=d.ph_threshold,
        beta_high=d.beta_high,
        recovery_len=d.recovery_len,
        boost_forgetting=d.boost_forgetting,
        ls_shrink=d.ls_shrink,
        n_grid_1d=d.n_grid_1d,
        n_grid_d=d.n_grid_d,
        n_restarts=d.n_restarts,
    )
    kwargs.update(overrides)
    return OptimizerConfig(**kwargs)


def method_config(name: str, d: StudyDefaults, **overrides) -> OptimizerConfig:
    """Build the canonical configuration for a method name."""
    name = str(name).lower()
    if name == "random_search":
        cfg = _base(d, name, use_gp=False, acquisition="ucb")
    elif name == "static_ei":
        cfg = _base(d, name, acquisition="ei", acquisition_params={"xi": d.xi})
    elif name == "static_ucb":
        cfg = _base(d, name, acquisition="ucb", acquisition_params={"beta": d.beta_base})
    elif name == "static_pi":
        cfg = _base(d, name, acquisition="pi", acquisition_params={"xi": d.xi})
    elif name == "exploitation":
        cfg = _base(d, name, acquisition="mean")
    elif name == "abo":
        cfg = _base(d, name, acquisition="ucb",
                    acquisition_params={"beta": d.beta_base}, half_life=d.half_life)
    elif name == "dabo":
        cfg = _base(d, name, acquisition="ucb",
                    acquisition_params={"beta": d.beta_base},
                    half_life=d.half_life,
                    drift_detection=True,
                    ls_shrink=d.ls_shrink)
    else:
        raise ValueError(f"unknown method: {name}")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def dabo_ablation(
    d: StudyDefaults,
    no_forgetting: bool = False,
    no_drift_detection: bool = False,
    no_exploration_boost: bool = False,
    no_memory_shortening: bool = False,
    half_life: Optional[float] = None,
    ph_threshold: Optional[float] = None,
    recovery_len: Optional[int] = None,
    beta_high: Optional[float] = None,
) -> OptimizerConfig:
    """DA-BO with individual components switched off (spec §29)."""
    cfg = method_config("dabo", d)
    if no_forgetting:
        cfg.half_life = None
    if no_drift_detection:
        cfg.drift_detection = False
        cfg.ls_shrink = 1.0
        cfg.boost_forgetting = 1.0
    if no_exploration_boost:
        cfg.beta_high = cfg.acquisition_params["beta"]  # beta never changes
    if no_memory_shortening:
        cfg.boost_forgetting = 1.0
        cfg.ls_shrink = 1.0
    if half_life is not None:
        cfg.half_life = None if half_life in ("inf", float("inf")) else float(half_life)
    if ph_threshold is not None:
        cfg.ph_threshold = float(ph_threshold)
    if recovery_len is not None:
        cfg.recovery_len = int(recovery_len)
    if beta_high is not None:
        cfg.beta_high = float(beta_high)
    cfg.name = "dabo_ablated"
    return cfg
