"""Focused regressions for posterior cancellation and SuSiE-style stopping."""

import numpy as np
import pytest

import MultiSuSiE
from MultiSuSiE import susiepy, susiepy_ss


@pytest.mark.parametrize(
    "kernel", ["compute_lbf_and_moments", "compute_lbf_and_moments_safe"]
)
def test_high_information_posterior_moments(kernel):
    v = np.array([0.2, 0.3, 0.1], dtype=np.float32)
    rho = np.full((3, 3), 0.75, dtype=np.float32)
    np.fill_diagonal(rho, 1)
    information = np.tile(
        np.array([20000, 50000, 10000], dtype=np.float32)[:, None], (1, 2)
    )
    scores = np.array([[0, 0, 0], [50, 60, 45]], dtype=np.float32)
    _, mean, second = getattr(susiepy_ss, kernel)(
        v, scores, information, rho, np.ones(3, dtype=np.float32), np.float32
    )
    a = rho.astype(float) * np.sqrt(np.outer(v.astype(float), v.astype(float)))
    covariance = np.linalg.inv(np.linalg.inv(a) + np.diag(information[:, 0]))
    expected_mean = covariance @ scores.T
    expected_second = covariance[:, :, None] + np.einsum(
        "ij,kj->ikj", expected_mean, expected_mean
    )
    if kernel == "compute_lbf_and_moments":
        expected_second = np.maximum(expected_second, 0)
    np.testing.assert_allclose(mean, expected_mean, rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(second, expected_second, rtol=1e-5, atol=1e-9)


def test_default_rss_precision_and_population_order(synthetic_data):
    data = synthetic_data
    common = dict(data.common)
    common.pop("float_type")
    rho = common.pop("rho")
    fits = []
    for order in [(0, 1, 2), (2, 1, 0)]:
        fit = MultiSuSiE.multisusie_rss(
            b_list=[data.beta_hat_list[k].copy() for k in order],
            s_list=[data.se_list[k].copy() for k in order],
            R_list=[data.r_list[k].copy() for k in order],
            varY_list=[data.vary_list[k] for k in order],
            population_sizes=[data.n_list[k] for k in order],
            rho=rho[np.ix_(order, order)],
            single_population_mac_thresh=0,
            **common,
        )
        assert fit.alpha.dtype == np.float64
        fits.append(fit)
    np.testing.assert_allclose(fits[0].pip, fits[1].pip, rtol=0, atol=1e-8)


@pytest.mark.parametrize("method", ["rss", "individual"])
@pytest.mark.parametrize("change", [0.0001, -0.0001, -1.0])
def test_elbo_stopping_matches_updated_susie(
    monkeypatch, synthetic_data, method, change
):
    objectives = iter([10.0, 11.0, 11.0 + change])
    module = susiepy_ss if method == "rss" else susiepy
    monkeypatch.setattr(module, "get_objective", lambda *args: next(objectives))
    data = synthetic_data
    common = dict(data.common)
    common.update(
        L=1,
        max_iter=3,
        tol=0.001,
        iter_before_zeroing_effects=0,
        estimate_prior_variance=False,
        estimate_prior_method=None,
        estimate_residual_variance=False,
    )
    if method == "rss":
        fit = MultiSuSiE.multisusie_rss(
            b_list=[b.copy() for b in data.beta_hat_list],
            s_list=[s.copy() for s in data.se_list],
            R_list=[r.copy() for r in data.r_list],
            varY_list=data.vary_list,
            population_sizes=data.n_list,
            single_population_mac_thresh=0,
            **common,
        )
    else:
        fit = MultiSuSiE.multisusie(
            X_list=[x.copy() for x in data.geno_list],
            Y_list=[y.copy() for y in data.y_list],
            **common,
        )
    assert fit.niter == 3
    assert fit.converged == (0 <= change < 0.001)
