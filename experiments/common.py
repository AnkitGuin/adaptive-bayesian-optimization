"""Shared machinery for all experiment scripts.

Fair-comparison contract enforced here (spec §22):

* one benchmark instance per (study, seed)  -> one objective state path;
* one identically seeded :class:`Environment` per (study, seed, method) ->
  every method observes the *same noise-draw sequence* (a fresh RNG seeded
  from the same parts per method, so draw t is identical for all methods);
* one initial design per (study, seed) shared by every method;
* identical budget / domain / seeds for all methods of a study.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.base import SearchSpace, Trajectory, initial_design
from src.benchmarks import Environment, ScenarioSpec, make_benchmark
from src.optimization import StudyDefaults, GPOptimizer, method_config, dabo_ablation
from src.utils.config import load_config
from src.utils.seeds import rng_for

__all__ = [
    "build_spec", "build_benchmark", "build_defaults", "run_one",
    "save_trajectory", "save_study_config", "load_run",
]

_SPEC_FIELDS = {
    "t_change", "v", "phase_speed", "from_state", "to_state", "sigma_drift",
    "noise_from", "noise_to", "t_switch1", "t_switch2", "transition",
}
_KEYMAP = {"from": "from_state", "to": "to_state"}


def build_spec(scenario_cfg: dict, horizon: int, design_horizon: int = 100) -> ScenarioSpec:
    """Build a ScenarioSpec, scaling change points to the actual horizon.

    Configs specify change points for a *designed* horizon (default 100).
    Smoke/quick runs use a shorter budget, so t_change / t_switch* are scaled
    by ``horizon / design_horizon`` to keep the same *relative* placement.
    """
    cfg = dict(scenario_cfg)
    typ = cfg.pop("type")
    scale = horizon / design_horizon
    kwargs = {"horizon": int(horizon)}
    has_t_change = False
    for k, v in cfg.items():
        key = _KEYMAP.get(k, k)
        if key in _SPEC_FIELDS:
            if key in ("t_change", "t_switch1", "t_switch2") and v is not None:
                kwargs[key] = max(1, int(round(float(v) * scale)))
                if key == "t_change":
                    has_t_change = True
            else:
                kwargs[key] = v
    if typ in ("abrupt", "noise_shift") and not has_t_change:
        # config omitted t_change: scale the ScenarioSpec default (50) too
        kwargs["t_change"] = max(1, min(int(round(50.0 * scale)), int(horizon)))
    return ScenarioSpec(type=typ, **kwargs)


def build_benchmark(name: str, bench_cfg: dict, scenario_name: str, horizon: int,
                    rng: np.random.Generator):
    scen_cfg = dict(bench_cfg["scenarios"][scenario_name])
    start = scen_cfg.pop("start", bench_cfg.get("start"))
    spec = build_spec(scenario_cfg=spec_from(scen_cfg), horizon=horizon)
    if name == "peak":
        kw = {"start": float(start), "sigma": float(bench_cfg.get("peak_sigma", 0.12))}
    elif name == "peak2d":
        s = [float(v) for v in start] if isinstance(start, (list, tuple)) else [0.35, 0.35]
        kw = {"start": tuple(s), "sigma": float(bench_cfg.get("peak_sigma", 0.15))}
    elif name == "multimodal":
        kw = {"centers": tuple(bench_cfg.get("centers", (0.2, 0.5, 0.8))),
              "widths": tuple(bench_cfg.get("widths", (0.08, 0.08, 0.08)))}
    elif name == "sinusoidal":
        kw = {"start_phase": float(start or 0.0)}
    elif name == "process":
        kw = {
            "t0": float(bench_cfg.get("t0", 190.0)),
            "drift_per_step": float(bench_cfg.get("drift_per_step", 0.15)),
            "p_center": float(bench_cfg.get("p_center", 2.6)),
            "p_osc": float(bench_cfg.get("p_osc", 0.8)),
            "period": int(bench_cfg.get("period", 60)),
            "s_opt": float(bench_cfg.get("s_opt", 45.0)),
        }
    else:
        raise ValueError(f"unknown benchmark {name}")
    return make_benchmark(name, spec, rng, **kw)


def spec_from(scen_cfg: dict) -> dict:
    return scen_cfg


def build_defaults(cfg, space: SearchSpace, noise_std: float, n_initial: int,
                   budget: int) -> StudyDefaults:
    d = space.dim
    ls_key = {1: "lengthscale_1d", 2: "lengthscale_2d"}.get(d, "lengthscale_3d")
    return StudyDefaults(
        space=space,
        budget=int(budget),
        n_initial=int(n_initial),
        initial_design="lhs",
        kernel=cfg.get("gp.kernel", "matern52"),
        lengthscale=float(cfg.get(f"gp.{ls_key}", 0.2)),
        noise_std=float(noise_std),
        xi=float(cfg.get("acquisition.ei.xi", 0.01)),
        beta_base=float(cfg.get("acquisition.ucb.beta_base", 2.0)),
        half_life=float(cfg.get("adaptive.half_life", 32.0)),
        beta_high=float(cfg.get("adaptive.beta_high", 6.0)),
        recovery_len=int(cfg.get("adaptive.recovery_len", 20)),
        ph_delta=float(cfg.get("adaptive.ph_delta", 0.5)),
        ph_threshold=float(cfg.get("adaptive.ph_threshold", 25.0)),
        boost_forgetting=float(cfg.get("adaptive.boost_forgetting", 2.0)),
        ls_shrink=float(cfg.get("adaptive.ls_shrink", 0.5)),
    )


def run_one(
    study: str,
    seed_id: int,
    benchmark,
    env: Environment,
    x0: np.ndarray,
    defaults: StudyDefaults,
    method: str,
    method_builder: Optional[Callable] = None,
) -> Trajectory:
    builder = method_builder or (lambda m, d: method_config(m, d))
    cfg_opt = builder(method, defaults)
    opt = GPOptimizer(cfg_opt, study, seed_id)
    return opt.run(env, x0)


# --------------------------------------------------------------------- I/O
def _method_dir_slug(method: str) -> str:
    return method.replace("/", "_")


def save_trajectory(traj: Trajectory, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obs = traj.observations
    d = len(obs[0].x)
    rows = []
    for o in obs:
        row = {
            "t": o.t,
            "is_initial": int(o.is_initial),
            **{f"x{i + 1}": float(o.x[i]) for i in range(d)},
            "y": float(o.y),
            "f_true": float(o.f_true),
            "f_opt": float(o.f_opt),
            **{f"xopt_{i + 1}": float(o.x_opt[i]) for i in range(d)},
            "regret": float(o.regret),
            "residual": np.nan if o.residual is None else float(o.residual),
            "mu_pred": np.nan if o.mu_pred is None else float(o.mu_pred),
            "sigma_pred": np.nan if o.sigma_pred is None else float(o.sigma_pred),
            "acq_value": np.nan if o.acquisition_value is None else float(o.acquisition_value),
            "beta": np.nan if o.beta_used is None else float(o.beta_used),
            "half_life": np.nan if o.half_life_used is None else float(o.half_life_used),
            "lengthscale": np.nan if o.lengthscale_used is None else float(o.lengthscale_used),
            "drift_detected": int(o.drift_detected),
        }
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_drift_events(traj: Trajectory, method: str, path: Path) -> None:
    if not traj.drift_events:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(traj.drift_events)
    df.insert(0, "method", method)
    df.to_csv(path, index=False)


def save_study_config(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def load_run(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def parse_common_args(description: str):
    import argparse

    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--benchmarks", default="configs/benchmarks.yaml")
    ap.add_argument("--experiments", default="configs/experiments.yaml")
    ap.add_argument("--output", default="results")
    ap.add_argument("--seed", type=int, default=None,
                    help="run a single seed instead of the full seed set")
    ap.add_argument("--quick", action="store_true",
                    help="smoke-test scale: smaller budget / seeds / grids")
    return ap


def load_all_configs(args):
    import yaml as _yaml

    cfg = load_config(args.config)
    with open(args.benchmarks, "r", encoding="utf-8") as fh:
        bench_cfgs = _yaml.safe_load(fh) or {}
    with open(args.experiments, "r", encoding="utf-8") as fh:
        exp_cfg = _yaml.safe_load(fh) or {}
    return cfg, bench_cfgs, exp_cfg


# ---------------------------------------------------------------- matrices
def seed_ids_for(cfg, quick: bool, count_key: str = "seeds.count") -> list[int]:
    if quick:
        n = int(cfg.get("budget.quick.seeds", 3))
    else:
        n = int(cfg.get(count_key, 20))
    base = int(cfg.get("random.base_seed", 20260923))
    return [base + i for i in range(n)]


def budget_for(cfg, quick: bool) -> int:
    if quick:
        return int(cfg.get("budget.quick.evaluations", 40))
    return int(cfg.get("budget.evaluations", 100))


def n_initial_for(cfg, space: SearchSpace, quick: bool) -> int:
    key = "n_initial_1d" if space.dim == 1 else "n_initial_2d3d"
    if quick:
        return int(cfg.get(f"budget.quick.{key}", 6))
    return int(cfg.get(f"budget.{key}", 10 if space.dim == 1 else 16))


def apply_quick(cfg_opt) -> None:
    """Shrink acquisition grids for smoke tests (deterministic either way)."""
    cfg_opt.n_grid_1d = 251
    cfg_opt.n_grid_d = 15
    cfg_opt.n_restarts = 3


def run_matrix(
    study: str,
    outdir: Path,
    cfg,
    bench_cfgs: dict,
    combos: list[tuple[str, str]],
    noises: list[str],
    methods: list[str],
    seeds: list[int],
    quick: bool = False,
    method_builder: Optional[Callable] = None,
    extra_defaults: Optional[dict] = None,
    label: str = "",
    budget_override: Optional[int] = None,
    noise_std_override: Optional[float] = None,
) -> dict:
    """Run one study matrix: combos x noises x methods x seeds.

    Returns a per-slug summary dict (also written to ``summary.json``).
    """
    from src.evaluation import cumulative_regret, tracking_error, summarize_seeds

    outdir = Path(outdir)
    budget = int(budget_override) if budget_override else budget_for(cfg, quick)
    horizon = budget
    all_summaries: dict = {}

    for bench_name, scen_name in combos:
        bench_cfg = bench_cfgs[bench_name]
        for noise in noises:
            noise = noise or "none"
            suffix = f"-{label}" if label else ""
            slug = f"{study}-{bench_name}-{scen_name}-{noise}{suffix}"
            raw_dir = outdir / "raw" / study / slug
            seed_summaries: dict[str, list] = {m: [] for m in methods}
            tracking_summaries: dict[str, list] = {m: [] for m in methods}
            t_start = time.time()

            for seed in seeds:
                bench = build_benchmark(
                    bench_name, bench_cfg, scen_name, horizon,
                    rng_for(study, seed, "scenario"),
                )
                spec = bench.spec
                if noise_std_override is not None:
                    noise_std = float(noise_std_override)
                elif noise == "none":
                    noise_std = float(getattr(spec, "noise_from", 0.0))
                else:
                    noise_std = float(cfg.get(f"noise.{noise}", 0.05))
                defaults = build_defaults(
                    cfg, bench.space, noise_std, n_initial_for(cfg, bench.space, quick), budget
                )
                if extra_defaults:
                    for k, v in extra_defaults.items():
                        setattr(defaults, k, v)
                n_init = int(defaults.n_initial)
                x0 = initial_design(bench.space, n_init,
                                    rng_for(study, seed, "init"), "lhs")
                if quick:
                    defaults.n_grid_1d = 251
                    defaults.n_grid_d = 15
                    defaults.n_restarts = 3

                for method in methods:
                    # Fresh, identically-seeded environment per method so
                    # that every method sees the same noise realization
                    # (spec §22): draw t of each stream is the same.
                    env = Environment(bench, noise_std, rng_for(study, seed, "env"))
                    traj = run_one(study, seed, bench, env, x0, defaults, method,
                                   method_builder=method_builder)
                    save_trajectory(traj, raw_dir / f"seed{seed}" / f"{method}.csv")
                    save_drift_events(traj, method, raw_dir / f"seed{seed}" / f"{method}_drift_events.csv")
                    cr = float(np.atleast_1d(cumulative_regret(traj))[-1])
                    te = float(np.mean(tracking_error(traj, bench)))
                    seed_summaries[method].append(cr)
                    tracking_summaries[method].append(te)

            # ---------------- summary for this (benchmark, noise) ----------------
            slug_summary = {"n_seeds": len(seeds), "methods": {}}
            for m in methods:
                s = summarize_seeds(np.asarray(seed_summaries[m]))
                st = summarize_seeds(np.asarray(tracking_summaries[m]))
                slug_summary["methods"][m] = {
                    "cumulative_regret": s,
                    "mean_tracking_error": st,
                }
            all_summaries[slug] = slug_summary
            save_study_config(raw_dir / "config.json", {
                "study": study, "slug": slug, "label": label,
                "benchmark": bench_name, "scenario": scen_name,
                "noise": noise, "budget": budget, "horizon": horizon,
                "methods": methods, "seeds": seeds, "quick": quick,
                "benchmark_description": bench.describe(),
                "study_defaults": {k: getattr(defaults, k) for k in
                                   ("kernel", "lengthscale", "noise_std", "xi",
                                    "beta_base", "half_life", "beta_high",
                                    "recovery_len", "ph_threshold")},
            })
            save_study_config(raw_dir / "summary.json", slug_summary)
            dt = time.time() - t_start
            row = " | ".join(
                f"{m}: CR={np.mean(seed_summaries[m]):.2f}±{np.std(seed_summaries[m], ddof=1) if len(seed_summaries[m]) > 1 else 0:.2f}"
                for m in methods
            )
            print(f"[{study}] {slug:48s} ({dt:5.1f}s) {row}", flush=True)
    return all_summaries
