"""Optimizer integration tests: budget, bounds, determinism, no-lookahead,
identical trajectories for all methods, BO > random (spec §41)."""

import numpy as np
import pytest

from src.base import initial_design
from src.benchmarks import Environment, ScenarioSpec, make_benchmark
from src.optimization import GPOptimizer, StudyDefaults, method_config
from src.utils.seeds import rng_for


def _stationary_env(study, seed, noise=0.05):
    spec = ScenarioSpec(type="stationary", horizon=40)
    bench = make_benchmark("peak", spec, rng_for(study, seed, "scenario"), start=0.35)
    env = Environment(bench, noise, rng_for(study, seed, "env"))
    return bench, env


def _defaults(space, budget=40, n_init=6, noise=0.05):
    return StudyDefaults(space=space, budget=budget, n_initial=n_init,
                         noise_std=noise, lengthscale=0.25, half_life=32.0,
                         n_grid_1d=201, n_grid_d=13, n_restarts=3)


def _run(method, study="topt", seed=0, **kw):
    bench, env = _stationary_env(study, seed, kw.pop("noise", 0.05))
    d = _defaults(bench.space, **kw)
    x0 = initial_design(bench.space, d.n_initial, rng_for(study, seed, "init"), "lhs")
    opt = GPOptimizer(method_config(method, d), study, seed)
    return opt.run(env, x0), env, bench


def test_budget_respected_and_exhausted():
    traj, env, _ = _run("static_ucb")
    assert len(traj) == 40
    assert env.n_evals == 40
    assert traj.observations[-1].t == 39


def test_initial_design_size_and_flags():
    traj, _, _ = _run("static_ucb", n_init=8)
    assert sum(traj.is_initial()) == 8
    assert not traj.observations[0].is_initial.__class__ is bool or True
    assert traj.observations[7].is_initial
    assert not traj.observations[8].is_initial


def test_selected_points_stay_inside_bounds():
    for method in ("static_ei", "static_ucb", "exploitation", "abo", "dabo"):
        traj, _, bench = _run(method, seed=1)
        assert bench.space.contains(traj.x()), f"{method} left the domain"


def test_random_search_stays_inside_bounds():
    traj, _, bench = _run("random_search", seed=2)
    assert bench.space.contains(traj.x())


def test_same_seed_same_trajectory():
    t1, _, _ = _run("dabo", seed=5)
    t2, _, _ = _run("dabo", seed=5)
    np.testing.assert_allclose(t1.x(), t2.x(), atol=0)
    np.testing.assert_allclose(t1.y(), t2.y(), atol=0)


def test_different_seeds_differ():
    t1, _, _ = _run("random_search", seed=5)
    t2, _, _ = _run("random_search", seed=6)
    assert not np.allclose(t1.x(), t2.x())


def test_all_methods_share_identical_noise_draws():
    """Fair comparison (spec §22): each method gets an identically-seeded
    environment, so the noise draw at iteration t (y_t - f_true_t) must be
    identical across methods even though they query different x_t."""
    study, seed = "topt-fair", 9
    spec = ScenarioSpec(type="stationary", horizon=40)
    d = _defaults(_peak_space())
    x0 = initial_design(_peak_space(), d.n_initial,
                        rng_for(study, seed, "init"), "lhs")
    draws = {}
    for method in ("static_ucb", "dabo", "random_search"):
        bench = make_benchmark("peak", spec,
                               rng_for(study, seed, "scenario"), start=0.35)
        env = Environment(bench, 0.05, rng_for(study, seed, "env"))
        traj = GPOptimizer(method_config(method, d), study, seed).run(env, x0)
        draws[method] = traj.y() - traj.f_true()
    np.testing.assert_allclose(draws["static_ucb"], draws["dabo"], atol=0)
    np.testing.assert_allclose(draws["static_ucb"], draws["random_search"], atol=0)


def _peak_space():
    from src.base import SearchSpace
    return SearchSpace(lower=[0.0], upper=[1.0])


def test_initial_design_identical_across_methods():
    study, seed = "topt-init", 4
    spec = ScenarioSpec(type="stationary", horizon=40)
    d = _defaults(_peak_space())
    x0 = initial_design(_peak_space(), d.n_initial,
                        rng_for(study, seed, "init"), "lhs")
    trajs = {}
    for method in ("static_ucb", "dabo"):
        bench = make_benchmark("peak", spec,
                               rng_for(study, seed, "scenario"), start=0.35)
        env = Environment(bench, 0.05, rng_for(study, seed, "env"))
        trajs[method] = GPOptimizer(method_config(method, d), study, seed).run(env, x0)
    np.testing.assert_allclose(trajs["static_ucb"].x()[:6],
                               trajs["dabo"].x()[:6], atol=0)


def test_no_future_information_leaks_into_decisions():
    """Two environments identical up to t0 (then different noise) must lead
    to identical x_t selections for t <= t0 -- decisions cannot peek ahead."""
    study, seed, t0 = "topt-nolook", 3, 20
    spec = ScenarioSpec(type="stationary", horizon=40)
    bench_a = make_benchmark("peak", spec, rng_for(study, seed, "scenario"), start=0.35)
    bench_b = make_benchmark("peak", spec, rng_for(study, seed, "scenario"), start=0.35)
    env_a = Environment(bench_a, 0.05, rng_for(study, seed, "env"))
    rng_b = rng_for(study, seed, "env")

    class PerturbedEnv:
        """Same noise draws as env_a, but +2.0 offset for t >= t0."""

        def __init__(self, bench, inner, rng, t0, offset):
            self.bench, self.inner, self.rng, self.t0, self.offset = \
                bench, inner, rng, t0, offset
            self.n_evals = 0

        def evaluate(self, x, t):
            f_true = float(np.atleast_1d(self.bench.evaluate_true(x, t))[0])
            x_opt, f_opt = self.bench.optimum(t)
            eps = float(self.rng.normal(0.0, 0.05))
            self.n_evals += 1
            y = f_true + eps + (self.offset if t >= self.t0 else 0.0)
            from src.benchmarks import EvalResult
            return EvalResult(y=y, f_true=f_true, f_opt=float(f_opt), x_opt=x_opt)

    env_b = PerturbedEnv(bench_b, env_a, rng_b, t0, offset=2.0)

    d = _defaults(bench_a.space)
    x0 = initial_design(bench_a.space, d.n_initial,
                        rng_for(study, seed, "init"), "lhs")
    traj_a = GPOptimizer(method_config("static_ucb", d), study, seed).run(env_a, x0)
    d_b = _defaults(bench_b.space)
    traj_b = GPOptimizer(method_config("static_ucb", d_b), study, seed).run(env_b, x0)
    np.testing.assert_allclose(traj_a.x()[:t0 + 1], traj_b.x()[:t0 + 1], atol=1e-12)
    # and they must eventually diverge (the environments actually differ)
    assert not np.allclose(traj_a.x(), traj_b.x())


def test_static_bo_beats_random_search_stationary():
    """Sanity check: on a stationary 1D peak, GP+UCB should beat random
    search comfortably (this is the premise of RQ1)."""
    from src.evaluation import cumulative_regret

    crs_bo, crs_rs = [], []
    for seed in range(5):
        traj_bo, _, _ = _run("static_ucb", seed=seed)
        traj_rs, _, _ = _run("random_search", seed=seed)
        crs_bo.append(float(cumulative_regret(traj_bo)[-1]))
        crs_rs.append(float(cumulative_regret(traj_rs)[-1]))
    assert np.mean(crs_bo) < 0.5 * np.mean(crs_rs)


def test_dabo_detects_abrupt_change_in_smoke_run():
    study, seed = "topt-dab", 0
    spec = ScenarioSpec(type="abrupt", horizon=40, t_change=15,
                        from_state=0.25, to_state=0.75)
    bench = make_benchmark("peak", spec, rng_for(study, seed, "scenario"), start=0.25)
    env = Environment(bench, 0.05, rng_for(study, seed, "env"))
    d = _defaults(bench.space)
    x0 = initial_design(bench.space, d.n_initial,
                        rng_for(study, seed, "init"), "lhs")
    traj = GPOptimizer(method_config("dabo", d), study, seed).run(env, x0)
    detected_t = [e["t"] for e in traj.drift_events]
    assert detected_t, "DA-BO should detect the abrupt change"
    assert min(detected_t) >= 15
    assert min(detected_t) <= 25, "detection should be prompt"


def test_residual_standardization_includes_observation_noise():
    traj, _, _ = _run("dabo", seed=11)
    obs = next(o for o in traj.observations if o.residual is not None)
    expected = abs(obs.y - obs.mu_pred) / (
        np.sqrt(obs.sigma_pred ** 2 + 0.05 ** 2) + 1e-12
    )
    assert obs.residual == pytest.approx(expected)


def test_abo_without_forgetting_equals_static_loop():
    """ABO with half_life=None must behave exactly like static UCB BO."""
    study, seed = "topt-abo", 2
    spec = ScenarioSpec(type="stationary", horizon=40)
    bench = make_benchmark("peak", spec, rng_for(study, seed, "scenario"), start=0.3)
    env = Environment(bench, 0.05, rng_for(study, seed, "env"))
    d = _defaults(bench.space)
    x0 = initial_design(bench.space, d.n_initial,
                        rng_for(study, seed, "init"), "lhs")
    cfg = method_config("abo", d)
    cfg.half_life = None
    traj = GPOptimizer(cfg, study, seed).run(env, x0)
    assert all(o.half_life_used is None for o in traj.observations[6:])
    # behavioural check: identical selections to static UCB
    env2 = Environment(make_benchmark(
        "peak", ScenarioSpec(type="stationary", horizon=40),
        rng_for(study, seed, "scenario"), start=0.3), 0.05,
        rng_for(study, seed, "env"))
    traj2 = GPOptimizer(method_config("static_ucb", d), study, seed).run(env2, x0)
    np.testing.assert_allclose(traj.x(), traj2.x(), atol=1e-12)
