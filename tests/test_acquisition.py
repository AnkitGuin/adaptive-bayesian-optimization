"""Acquisition-function unit tests (spec §41)."""

import numpy as np
import pytest

from src.acquisition import (
    expected_improvement,
    make_acquisition,
    probability_of_improvement,
    upper_confidence_bound,
)


def test_ei_non_negative():
    mu = np.linspace(-1, 2, 50)
    sigma = np.linspace(0.0, 1.0, 50)
    ei = expected_improvement(mu, sigma, f_best=0.0, xi=0.01)
    assert np.all(ei >= 0.0)
    assert np.all(np.isfinite(ei))


def test_ei_zero_when_no_improvement_and_no_uncertainty():
    mu = np.array([-0.5, 0.0, 0.005])
    sigma = np.array([1e-14, 1e-14, 1e-14])
    ei = expected_improvement(mu, sigma, f_best=0.0, xi=0.01)
    np.testing.assert_allclose(ei, [0.0, 0.0, 0.0], atol=1e-12)


def test_ei_limit_zero_sigma_positive_imp():
    mu = np.array([0.5])
    sigma = np.array([1e-14])
    ei = expected_improvement(mu, sigma, f_best=0.0, xi=0.01)
    np.testing.assert_allclose(ei, [0.49], atol=1e-6)


def test_ei_no_nan_at_zero_sigma():
    ei = expected_improvement(np.array([0.0]), np.array([0.0]), f_best=0.0)
    assert np.all(np.isfinite(ei))


def test_ucb_increases_with_uncertainty():
    mu = np.full(10, 0.5)
    sigma = np.linspace(0.0, 1.0, 10)
    ucb = upper_confidence_bound(mu, sigma, beta=2.0)
    assert np.all(np.diff(ucb) > 0)
    np.testing.assert_allclose(ucb[0], mu[0], atol=1e-12)  # sigma=0 -> mu


def test_ucb_linear_in_beta():
    mu, sigma = np.array([0.3]), np.array([0.7])
    np.testing.assert_allclose(
        upper_confidence_bound(mu, sigma, beta=4.0),
        upper_confidence_bound(mu, sigma, beta=2.0) + 2.0 * 0.7, atol=1e-12)


def test_pi_in_unit_interval():
    mu = np.linspace(-2, 2, 60)
    sigma = np.linspace(0.0, 1.5, 60)
    pi = probability_of_improvement(mu, sigma, f_best=0.0)
    assert np.all((pi >= 0.0) & (pi <= 1.0))
    assert np.all(np.isfinite(pi))


def test_pi_zero_sigma_limits():
    # mu > f_best + xi  <=>  PI = 1 in the zero-variance limit
    mu = np.array([-1.0, 0.005, 0.1, 1.0])
    sigma = np.array([0.0, 0.0, 0.0, 0.0])
    pi = probability_of_improvement(mu, sigma, f_best=0.0, xi=0.01)
    np.testing.assert_allclose(pi, [0.0, 0.0, 1.0, 1.0], atol=1e-12)


def test_pi_approaches_one_for_large_improvement():
    pi = probability_of_improvement(np.array([10.0]), np.array([0.1]), f_best=0.0)
    assert pi[0] > 0.9999


def test_registry_returns_callable_with_unified_signature():
    for name, params in [("ei", {"xi": 0.01}), ("ucb", {"beta": 2.0}),
                         ("pi", {"xi": 0.01}), ("mean", {})]:
        fn = make_acquisition(name, params)
        out = fn(np.array([0.2]), np.array([0.1]), 0.0)
        assert out.shape == (1,)
        assert np.all(np.isfinite(out))
    with pytest.raises(ValueError):
        make_acquisition("bogus")


def test_registry_ucb_ignores_fbest():
    fn = make_acquisition("ucb", {"beta": 3.0})
    a = fn(np.array([0.5]), np.array([0.2]), f_best=-100.0)
    b = fn(np.array([0.5]), np.array([0.2]), f_best=100.0)
    np.testing.assert_allclose(a, b)
