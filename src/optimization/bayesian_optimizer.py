"""The sequential Bayesian-optimization loop (shared by all methods).

One class, :class:`GPOptimizer`, implements *every* method in the study by
switching configuration (spec §22, fair comparison):

* **Random search**            -- ``use_gp = False``.
* **Static BO + {EI, UCB, PI}** -- static GP, fixed acquisition, no weights.
* **Pure exploitation**        -- static GP, acquisition = posterior mean.
* **ABO (time-decayed BO)**    -- weighted GP (half-life), UCB, no detection.
* **DA-BO (drift-aware ABO)**  -- ABO + Page-Hinkley drift detection on
  standardized residuals + recovery window with (i) exploration boost
  (beta_high), (ii) temporarily increased forgetting and (iii) shortened
  effective memory (lengthscale shrink).

Fairness guarantees, enforced by construction:

* the initial design is *passed in* (identical for all methods);
* the environment holds the noise RNG (identical noise draws per iteration
  for all methods of one (study, seed));
* the loop never uses information from ``t`` before ``x_t`` is evaluated;
* the budget is exactly ``cfg.budget`` evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..base import Observation, SearchSpace, Trajectory
from ..acquisition import make_acquisition
from ..drift import PageHinkley
from ..gp import GaussianProcess, WeightedGaussianProcess, make_kernel
from ..utils.seeds import rng_for
from .acquisition_optimizer import AcquisitionOptimizer

__all__ = ["OptimizerConfig", "GPOptimizer"]


@dataclass
class OptimizerConfig:
    name: str
    space: SearchSpace
    budget: int = 100
    n_initial: int = 10
    initial_design: str = "lhs"
    kernel: str = "matern52"
    lengthscale: float = 0.25
    noise_std: float = 0.05
    acquisition: str = "ucb"
    acquisition_params: dict = field(default_factory=lambda: {"beta": 2.0})
    use_gp: bool = True
    # ---- adaptive (ABO / DA-BO) -----------------------------------------
    half_life: Optional[float] = None          # None/inf -> no forgetting
    half_life_schedule: str = "fixed"          # "fixed" | "ramp"
    drift_detection: bool = False
    ph_delta: float = 0.5
    ph_threshold: float = 25.0
    beta_high: float = 6.0
    recovery_len: int = 20
    boost_forgetting: float = 2.0              # during recovery: h -> h / this
    ls_shrink: float = 1.0                     # during recovery: ls -> ls * this
    # ---- acquisition maximisation ---------------------------------------
    n_grid_1d: int = 401
    n_grid_d: int = 21
    n_restarts: int = 5

    def __post_init__(self) -> None:
        if self.budget <= self.n_initial:
            raise ValueError("budget must exceed the initial design size")
        if self.n_initial < 2:
            raise ValueError("need at least two initial observations")
        if self.noise_std < 0.0 or not np.isfinite(self.noise_std):
            raise ValueError("noise_std must be finite and non-negative")
        if self.half_life is not None and not (
            (np.isinf(float(self.half_life)) and float(self.half_life) > 0.0)
            or (np.isfinite(float(self.half_life)) and float(self.half_life) > 0.0)
        ):
            raise ValueError("half_life must be positive, infinity, or None")
        if self.half_life_schedule not in {"fixed", "ramp"}:
            raise ValueError("half_life_schedule must be 'fixed' or 'ramp'")
        if self.recovery_len < 0:
            raise ValueError("recovery_len must be non-negative")
        if self.boost_forgetting < 1.0 or not np.isfinite(self.boost_forgetting):
            raise ValueError("boost_forgetting must be finite and >= 1")
        if self.ls_shrink <= 0.0 or self.ls_shrink > 1.0 or not np.isfinite(self.ls_shrink):
            raise ValueError("ls_shrink must be in (0, 1]")


class GPOptimizer:
    def __init__(self, cfg: OptimizerConfig, study: str, seed_id: int) -> None:
        self.cfg = cfg
        self.study = study
        self.seed_id = seed_id
        # Only random search needs its own RNG (its own stream keeps the
        # method deterministic and independent from the shared env noise).
        self.rng_random = rng_for(study, seed_id, "random_search")
        self.acq_optimizer = AcquisitionOptimizer(
            cfg.space, n_grid_1d=cfg.n_grid_1d, n_grid_d=cfg.n_grid_d,
            n_restarts=cfg.n_restarts,
        )

    # ------------------------------------------------------------------ api
    def run(self, env: "Environment", initial_design: np.ndarray) -> Trajectory:  # noqa: F821
        cfg = self.cfg
        traj = Trajectory(meta={
            "method": cfg.name, "study": self.study, "seed": self.seed_id,
            "config": _config_to_dict(cfg),
        })

        X0 = np.atleast_2d(np.asarray(initial_design, dtype=float))
        if X0.shape[0] != cfg.n_initial:
            raise ValueError("initial design size mismatch")

        # ---------------- initial design ----------------
        for i, x in enumerate(X0):
            res = env.evaluate(x, i)
            traj.observations.append(Observation(
                t=i, x=x, y=res.y, f_true=res.f_true, f_opt=res.f_opt,
                x_opt=res.x_opt, is_initial=True,
            ))

        # ---------------- sequential loop ----------------
        ph = PageHinkley(delta=cfg.ph_delta,
                         threshold=cfg.ph_threshold) if cfg.drift_detection else None
        last_drift = -10**9

        for t in range(cfg.n_initial, cfg.budget):
            hist = traj.observations
            in_recovery = cfg.drift_detection and (t - last_drift) < cfg.recovery_len

            x_t = None
            mu_at = sigma_at = acq_val = beta_used = None
            half_life_t = self._current_half_life(t, in_recovery)
            ls_t = float(cfg.lengthscale)
            detected = False
            residual = None

            if cfg.use_gp:
                Xh = np.asarray([o.x for o in hist])
                yh = np.asarray([o.y for o in hist])
                th = np.asarray([o.t for o in hist], dtype=float)

                if in_recovery and cfg.ls_shrink < 1.0:
                    ls_t = float(cfg.lengthscale) * float(cfg.ls_shrink)
                kernel = make_kernel(cfg.kernel, ls_t)

                if half_life_t is None or not np.isfinite(half_life_t):
                    gp = GaussianProcess(kernel, noise_var=cfg.noise_std ** 2,
                                         normalize=True)
                    gp.fit(Xh, yh)
                else:
                    gp = WeightedGaussianProcess(kernel, base_noise_var=cfg.noise_std ** 2)
                    gp.fit(Xh, yh, th, t_now=float(t), half_life=half_life_t)

                f_best = float(max(o.y for o in hist))
                params_t = dict(cfg.acquisition_params)
                if cfg.acquisition == "ucb" and in_recovery:
                    params_t["beta"] = float(cfg.beta_high)
                beta_used = params_t.get("beta", cfg.acquisition_params.get("beta"))
                acq_fn = make_acquisition(cfg.acquisition, params_t)

                def acq(Xq: np.ndarray) -> np.ndarray:
                    mu, var = gp.predict(Xq)
                    return acq_fn(mu, np.sqrt(var), f_best)

                x_t, acq_val = self.acq_optimizer(acq)
                mu_at_v, var_at = gp.predict(x_t.reshape(1, -1))
                mu_at = float(mu_at_v[0])
                sigma_at = float(np.sqrt(max(float(var_at[0]), 0.0)))
            else:
                x_t = cfg.space.uniform(1, self.rng_random)[0]
                half_life_t = None

            # ---------------- evaluate the (noisy) objective ----------------
            res = env.evaluate(x_t, t)

            # Standardized residual of the *pre-evaluation* posterior.  Since
            # y is noisy, its predictive standard deviation includes both
            # latent-function uncertainty and the configured observation
            # noise.  Using only sigma_at would systematically overstate
            # surprises whenever measurement noise is non-zero.
            if cfg.use_gp and sigma_at is not None:
                predictive_sigma = float(np.sqrt(sigma_at * sigma_at + cfg.noise_std ** 2))
                residual = abs(res.y - mu_at) / (predictive_sigma + 1e-12)
                if ph is not None:
                    detected = ph.update(residual)
                    if detected:
                        last_drift = t
                        traj.drift_events.append({
                            "t": int(t),
                            "residual": float(residual),
                            "mu_pred": mu_at,
                            "sigma_pred": sigma_at,
                            "predictive_sigma": predictive_sigma,
                            "y": float(res.y),
                        })

            traj.observations.append(Observation(
                t=t, x=x_t, y=res.y, f_true=res.f_true, f_opt=res.f_opt,
                x_opt=res.x_opt, is_initial=False,
                mu_pred=mu_at, sigma_pred=sigma_at, residual=residual,
                acquisition_value=acq_val, beta_used=beta_used,
                half_life_used=half_life_t, lengthscale_used=ls_t,
                drift_detected=detected,
            ))

        if ph is not None:
            traj.meta["ph_final_state"] = ph.state()
        return traj

    # ------------------------------------------------------------- helpers
    def _current_half_life(self, t: int, in_recovery: bool) -> Optional[float]:
        h = self.cfg.half_life
        if h is None:
            return None
        h = float(h)
        if not np.isfinite(h):
            return None
        if self.cfg.half_life_schedule == "ramp":
            h = h / (1.0 + 2.0 * (t / max(self.cfg.budget - 1, 1)))
        if in_recovery and self.cfg.boost_forgetting > 1.0:
            h = h / float(self.cfg.boost_forgetting)
        return h


def _config_to_dict(cfg: OptimizerConfig) -> dict:
    from dataclasses import asdict
    d = asdict(cfg)
    d["space"] = cfg.space.to_dict()
    return d
