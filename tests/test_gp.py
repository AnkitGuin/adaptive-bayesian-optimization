"""GP unit tests: shapes, non-negative variance, determinism, interpolation,
and cross-validation against scikit-learn (spec §41)."""

import numpy as np
import pytest

from src.gp import GaussianProcess, WeightedGaussianProcess, make_kernel
from src.gp.sklearn_ref import SklearnGaussianProcess
from src.utils.seeds import rng_for


@pytest.fixture
def toy_data():
    rng = rng_for("test-gp", 0)
    X = rng.random((12, 1)) * 2.0 - 1.0
    y = np.sin(2.0 * X[:, 0]) + 0.1 * rng.normal(size=12)
    return X, y


def test_posterior_shapes(toy_data):
    X, y = toy_data
    gp = GaussianProcess(make_kernel("matern52", 0.5), noise_var=0.05**2).fit(X, y)
    Xs = np.linspace(-1, 1, 50).reshape(-1, 1)
    mu, var = gp.predict(Xs)
    assert mu.shape == (50,)
    assert var.shape == (50,)
    assert np.all(var >= 0.0), "variance must be non-negative"


def test_predict_std_matches_sqrt_var(toy_data):
    X, y = toy_data
    gp = GaussianProcess(make_kernel("rbf", 0.5), noise_var=0.01).fit(X, y)
    Xs = rng_for("test-gp-std", 1).random((37, 1))
    sd = gp.predict_std(Xs)
    _, var = gp.predict(Xs)
    np.testing.assert_allclose(sd, np.sqrt(var), atol=1e-12)


def test_deterministic_under_fixed_configuration(toy_data):
    X, y = toy_data
    Xs = rng_for("test-gp-det", 2).random((20, 1))
    out = []
    for _ in range(2):
        gp = GaussianProcess(make_kernel("matern52", 0.4), noise_var=0.02).fit(X, y)
        out.append(gp.predict(Xs))
    np.testing.assert_allclose(out[0][0], out[1][0], atol=0)
    np.testing.assert_allclose(out[0][1], out[1][1], atol=0)


def test_zero_noise_interpolation(toy_data):
    X, y = toy_data
    gp = GaussianProcess(make_kernel("matern52", 0.3), noise_var=0.0).fit(X, y)
    mu, var = gp.predict(X)
    # interpolation up to the numerical jitter (1e-8) scale
    np.testing.assert_allclose(mu, y, atol=1e-4)
    assert np.all(var < 1e-6)


def test_variance_smaller_than_prior_away_from_data():
    X = np.array([[0.0], [1.0]])
    y = np.array([0.5, -0.5])
    gp = GaussianProcess(make_kernel("matern52", 0.4), noise_var=0.0).fit(X, y)
    _, var_far = gp.predict(np.array([[5.0]]))
    _, var_near = gp.predict(np.array([[0.5]]))
    assert var_far[0] > var_near[0]


def test_matches_sklearn_reference(toy_data):
    """The custom GP must agree with scikit-learn's GP on identical data/kernel.

    Notes on conventions:
    * comparison is done with ``normalize=False`` (sklearn's normalize_y does
      not rescale the WhiteKernel noise by 1/sd^2, a known quirk);
    * sklearn's predictive variance includes the WhiteKernel noise at the
      test point (variance of a new *noisy* observation), so it is larger
      than the posterior variance of f(x) by exactly the noise variance.
    """
    X, y = toy_data
    Xs = np.linspace(-1.0, 1.0, 30).reshape(-1, 1)
    kind, ls, noise = "matern52", 0.5, 0.05

    mine = GaussianProcess(make_kernel(kind, ls), noise_var=noise**2,
                           normalize=False).fit(X, y)
    ref = SklearnGaussianProcess(kind=kind, lengthscale=ls,
                                 signal_variance=1.0, noise_var=noise**2,
                                 normalize=False)
    ref.fit(X, y)
    mu_mine, var_mine = mine.predict(Xs)
    mu_ref, var_ref = ref.predict(Xs)
    np.testing.assert_allclose(mu_mine, mu_ref, atol=2e-6)
    np.testing.assert_allclose(var_mine, var_ref - noise**2, rtol=1e-5, atol=1e-8)


def test_matches_sklearn_rbf(toy_data):
    X, y = toy_data
    Xs = np.linspace(-1.0, 1.0, 25).reshape(-1, 1)
    noise = 0.2
    mine = GaussianProcess(make_kernel("rbf", 0.6), noise_var=noise**2,
                           normalize=False).fit(X, y)
    ref = SklearnGaussianProcess(kind="rbf", lengthscale=0.6,
                                 signal_variance=1.0, noise_var=noise**2,
                                 normalize=False)
    ref.fit(X, y)
    mu_mine, _ = mine.predict(Xs)
    mu_ref, _ = ref.predict(Xs)
    np.testing.assert_allclose(mu_mine, mu_ref, atol=2e-6)


def test_weighted_gp_shapes_and_finite():
    rng = rng_for("test-wgp", 0)
    X = rng.random((20, 1))
    y = np.sin(3 * X[:, 0]) + 0.05 * rng.normal(size=20)
    t_obs = np.arange(20, dtype=float)
    wgp = WeightedGaussianProcess(make_kernel("matern52", 0.3),
                                  base_noise_var=0.05**2)
    wgp.fit(X, y, t_obs, t_now=25.0, half_life=10.0)
    Xs = np.linspace(0, 1, 40).reshape(-1, 1)
    mu, var = wgp.predict(Xs)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(var))
    assert np.all(var >= 0)
    assert wgp.last_weights.shape == (20,)
    # observation i is (25 - i) iterations old with half-life 10
    expected = 2.0 ** (-np.arange(25.0, 5.0, -1.0) / 10.0)
    np.testing.assert_allclose(wgp.last_weights, expected, rtol=1e-9)
    assert np.all(np.diff(wgp.last_weights) > 0)  # newer -> larger weight


def test_weighted_gp_no_forgetting_matches_static():
    """half_life=inf must reproduce the homoscedastic GP."""
    rng = rng_for("test-wgp2", 1)
    X = rng.random((15, 1))
    y = np.cos(2 * X[:, 0]) + 0.03 * rng.normal(size=15)
    t_obs = np.arange(15, dtype=float)
    Xs = np.linspace(0, 1, 30).reshape(-1, 1)

    static = GaussianProcess(make_kernel("matern52", 0.35),
                             noise_var=0.03**2).fit(X, y)
    wgp = WeightedGaussianProcess(make_kernel("matern52", 0.35),
                                  base_noise_var=0.03**2)
    wgp.fit(X, y, t_obs, t_now=20.0, half_life=float("inf"))
    np.testing.assert_allclose(static.predict(Xs)[0], wgp.predict(Xs)[0], atol=1e-10)
    np.testing.assert_allclose(static.predict(Xs)[1], wgp.predict(Xs)[1],
                               rtol=1e-8, atol=1e-10)


def test_weighted_gp_forgets_stale_data():
    """With a short half-life, a recent observation should dominate the
    posterior mean over an old, contradictory one."""
    X_old = np.array([[0.5]])
    y_old = np.array([1.0])
    X_new = np.array([[0.52]])
    y_new = np.array([-1.0])
    Xs = np.array([[0.51]])

    static = GaussianProcess(make_kernel("matern52", 0.3),
                             noise_var=0.05**2).fit(np.vstack([X_old, X_new]),
                                                    np.concatenate([y_old, y_new]))
    wgp = WeightedGaussianProcess(make_kernel("matern52", 0.3),
                                  base_noise_var=0.05**2)
    wgp.fit(np.vstack([X_old, X_new]), np.concatenate([y_old, y_new]),
            t_obs=np.array([0.0, 10.0]), t_now=10.0, half_life=2.0)
    mu_static, _ = static.predict(Xs)
    mu_w, _ = wgp.predict(Xs)
    # the weighted GP should sit much closer to the recent (-1) observation
    assert abs(mu_w[0] - (-1.0)) < abs(mu_static[0] - (-1.0))


def test_gp_needs_two_points():
    gp = GaussianProcess(make_kernel("rbf", 0.3), noise_var=0.01)
    with pytest.raises(ValueError):
        gp.fit(np.array([[0.0]]), np.array([1.0]))


def test_gp_rejects_invalid_noise_and_nonfinite_data():
    with pytest.raises(ValueError):
        GaussianProcess(make_kernel("rbf", 0.3), noise_var=-1.0)
    gp = GaussianProcess(make_kernel("rbf", 0.3), noise_var=0.01)
    with pytest.raises(ValueError):
        gp.fit(np.array([[0.0], [np.nan]]), np.array([1.0, 2.0]))


def test_log_marginal_likelihood_finite(toy_data):
    X, y = toy_data
    gp = GaussianProcess(make_kernel("matern52", 0.5), noise_var=0.05**2).fit(X, y)
    ll = gp.log_marginal_likelihood()
    assert np.isfinite(ll)
