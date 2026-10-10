"""Benchmark unit tests: known optima, moving optima, determinism (spec §41)."""

import numpy as np
import pytest

from src.benchmarks import (
    Environment,
    ScenarioSpec,
    make_benchmark,
)
from src.utils.seeds import rng_for


def test_peak_optimum_equals_mu_path():
    spec = ScenarioSpec(type="slow", horizon=30, v=0.01)
    bench = make_benchmark("peak", spec, rng_for("tb-peak", 0), start=0.2)
    for t in (0, 7, 29):
        x_opt, f_opt = bench.optimum(t)
        assert x_opt[0] == pytest.approx(bench.mu_path[t])
        assert f_opt == pytest.approx(1.0)
        # the reported optimum really maximises the noise-free function
        grid = np.linspace(0, 1, 2001)
        assert bench.evaluate_true(x_opt.reshape(1, -1), t)[0] == pytest.approx(
            bench.evaluate_true(grid, t).max(), abs=1e-6)


def test_peak2d_optimum_interior_and_exact():
    spec = ScenarioSpec(type="abrupt", horizon=20, t_change=10,
                        from_state=(0.3, 0.6), to_state=(0.7, 0.4))
    bench = make_benchmark("peak2d", spec, rng_for("tb-peak2d", 0),
                           start=(0.3, 0.6))
    for t in (3, 15):
        x_opt, f_opt = bench.optimum(t)
        assert bench.space.contains(x_opt.reshape(1, -1))
        assert f_opt == pytest.approx(1.0)
        val = bench.evaluate_true(x_opt.reshape(1, -1), t)[0]
        assert val == pytest.approx(1.0, abs=1e-12)


def test_sinusoidal_optima_are_true_maxima():
    spec = ScenarioSpec(type="slow", horizon=40, phase_speed=0.05)
    bench = make_benchmark("sinusoidal", spec, rng_for("tb-sin", 0), start_phase=0.3)
    for t in (0, 11, 40):
        opts = bench.optima_set(t)
        assert opts.shape[0] in (1, 2)
        for xs in opts:
            f = bench.evaluate_true(xs, t)[0]
            assert f == pytest.approx(1.0, abs=1e-12)
        x_opt, f_opt = bench.optimum(t)
        assert f_opt == pytest.approx(1.0)
        assert x_opt[0] == pytest.approx(opts[0, 0])


def test_sinusoidal_phase_is_not_clipped_to_search_domain():
    spec = ScenarioSpec(type="slow", horizon=100, phase_speed=0.2)
    bench = make_benchmark("sinusoidal", spec, rng_for("tb-sin-phase", 0))
    assert bench.phi_path[-1] == pytest.approx(20.0)
    # The wrapped phase still defines an exact maximum in the x-domain.
    x_opt, f_opt = bench.optimum(100)
    assert f_opt == pytest.approx(1.0, abs=1e-12)
    assert bench.space.contains(x_opt.reshape(1, -1))


def test_multimodal_switch_moves_dominant_peak():
    spec = ScenarioSpec(type="switch", horizon=40, t_switch1=10, t_switch2=30)
    bench = make_benchmark("multimodal", spec, rng_for("tb-mm", 0))
    early, mid, late = bench.x_opt_path[2], bench.x_opt_path[20], bench.x_opt_path[38]
    # dominant peak moves 1 -> 2 -> 3 (centres 0.2 / 0.5 / 0.8)
    assert abs(early - 0.2) < 0.05
    assert abs(mid - 0.5) < 0.05
    assert abs(late - 0.8) < 0.05
    # reported f_opt matches the noise-free function at x_opt
    for t in (2, 20, 38):
        x_star = np.array([bench.x_opt_path[t]])
        assert bench.f_opt_path[t] == pytest.approx(
            bench.evaluate_true(x_star, t)[0], abs=1e-10)


def test_random_walk_path_deterministic_under_seed():
    spec = ScenarioSpec(type="random_walk", horizon=25, sigma_drift=0.02)
    b1 = make_benchmark("peak", spec, rng_for("tb-rw", 7), start=0.5)
    b2 = make_benchmark("peak", spec, rng_for("tb-rw", 7), start=0.5)
    np.testing.assert_allclose(b1.mu_path, b2.mu_path, atol=0)
    # different seed -> (almost surely) a different path
    b3 = make_benchmark("peak", spec, rng_for("tb-rw", 8), start=0.5)
    assert not np.allclose(b1.mu_path, b3.mu_path)


def test_optima_stay_inside_domain():
    spec = ScenarioSpec(type="random_walk", horizon=200, sigma_drift=0.05)
    bench = make_benchmark("peak", spec, rng_for("tb-bounds", 3), start=0.5)
    for t in range(0, 201, 17):
        x_opt, _ = bench.optimum(t)
        assert bench.space.contains(x_opt.reshape(1, -1))


def test_noise_shift_schedule():
    spec = ScenarioSpec(type="noise_shift", horizon=30, t_change=15,
                        noise_from=0.02, noise_to=0.15)
    bench = make_benchmark("peak", spec, rng_for("tb-ns", 0), start=0.5)
    assert bench.base_noise_std(5) == pytest.approx(0.02)
    assert bench.base_noise_std(15) == pytest.approx(0.15)
    assert bench.base_noise_std(29) == pytest.approx(0.15)


def test_environment_shares_noise_draws_between_methods():
    """Two environments with the same (study, seed) stream must emit the same
    noise draws per iteration -- the fair-comparison mechanism (spec §22)."""
    spec = ScenarioSpec(type="stationary", horizon=20)
    b1 = make_benchmark("peak", spec, rng_for("tb-env", 1), start=0.4)
    b2 = make_benchmark("peak", spec, rng_for("tb-env", 1), start=0.4)
    e1 = Environment(b1, 0.1, rng_for("tb-env", 1, "env"))
    e2 = Environment(b2, 0.1, rng_for("tb-env", 1, "env"))
    for t in range(12):
        r1 = e1.evaluate(np.array([0.3]), t)
        r2 = e2.evaluate(np.array([0.3]), t)
        assert r1.y - r1.f_true == pytest.approx(r2.y - r2.f_true, abs=1e-15)


def test_environment_noise_free_values_unaffected_by_noise():
    spec = ScenarioSpec(type="stationary", horizon=10)
    bench = make_benchmark("peak", spec, rng_for("tb-env2", 2), start=0.5)
    env = Environment(bench, 0.5, rng_for("tb-env2", 2, "env"))
    r = env.evaluate(np.array([0.5]), 3)
    assert r.f_true == pytest.approx(1.0, abs=1e-12)
    assert r.f_opt == pytest.approx(1.0)
    assert r.x_opt[0] == pytest.approx(0.5)


def test_process_simulator_known_optimum():
    from src.benchmarks import ProcessSimulator

    spec = ScenarioSpec(type="slow", horizon=50)
    bench = ProcessSimulator(spec, spec.horizon)
    for t in (0, 10, 50):
        x_opt, f_opt = bench.optimum(t)
        assert f_opt == pytest.approx(92.0)
        val = bench.evaluate_true(x_opt.reshape(1, -1), t)[0]
        assert val == pytest.approx(92.0, abs=1e-9)
        # temperature optimum drifts, pressure optimum oscillates (quarter
        # period = 15 iterations for period 60)
    assert bench.t_opt_path[50] > bench.t_opt_path[0]
    assert abs(bench.p_opt_path[15] - bench.p_opt_path[0]) == pytest.approx(0.8, abs=1e-9)


def test_process_optimum_is_constrained_for_long_horizon():
    from src.benchmarks import ProcessSimulator

    spec = ScenarioSpec(type="slow", horizon=500)
    bench = ProcessSimulator(spec, spec.horizon, t0=190.0, drift_per_step=0.2)
    x_opt, f_opt = bench.optimum(500)
    assert bench.space.contains(x_opt.reshape(1, -1))
    assert f_opt == pytest.approx(bench.evaluate_true(x_opt.reshape(1, -1), 500)[0])
