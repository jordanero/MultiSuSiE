import numpy as np

import MultiSuSiE


def test_rss_output_shapes_and_attributes(synthetic_data):
    fit = MultiSuSiE.multisusie_rss(
        b_list=[b.copy() for b in synthetic_data.beta_hat_list],
        s_list=[s.copy() for s in synthetic_data.se_list],
        R_list=[r.copy() for r in synthetic_data.r_list],
        varY_list=synthetic_data.vary_list,
        population_sizes=synthetic_data.n_list,
        single_population_mac_thresh=0,
        low_memory_mode=False,
        **synthetic_data.common,
    )

    k = len(synthetic_data.n_list)
    p = synthetic_data.beta_hat_list[0].shape[0]
    l = synthetic_data.common["L"]

    assert fit.pip.shape == (p,)
    assert fit.coef.shape == (k, p)
    assert fit.coef_sd.shape == (k, p)
    assert fit.alpha.shape == (l, p)
    assert fit.mu.shape == (k, l, p)
    assert fit.mu2.shape == (k, k, l, p)
    assert fit.sigma2.shape == (k,)
    assert fit.V.shape == (k, l)
    assert fit.n.shape == (k,)
    assert hasattr(fit, "sets")


def test_individual_output_shapes_and_attributes(synthetic_data):
    fit = MultiSuSiE.multisusie(
        X_list=[x.copy() for x in synthetic_data.geno_list],
        Y_list=[y.copy() for y in synthetic_data.y_list],
        standardize=False,
        **synthetic_data.common,
    )

    k = len(synthetic_data.n_list)
    p = synthetic_data.beta_hat_list[0].shape[0]
    l = synthetic_data.common["L"]

    assert fit.pip.shape == (p,)
    assert fit.coef.shape == (k, p)
    assert fit.coef_sd.shape == (k, p)
    assert fit.alpha.shape == (l, p)
    assert fit.mu.shape == (k, l, p)
    assert fit.mu2.shape == (k, k, l, p)
    assert fit.sigma2.shape == (k,)
    assert fit.V.shape == (k, l)
    assert fit.n.shape == (k,)
    assert hasattr(fit, "sets")


def test_variant_ids_are_propagated_to_credible_sets(synthetic_data):
    p = synthetic_data.beta_hat_list[0].shape[0]
    variant_ids = [f"rs{i+1}" for i in range(p)]

    fit = MultiSuSiE.multisusie_rss(
        b_list=[b.copy() for b in synthetic_data.beta_hat_list],
        s_list=[s.copy() for s in synthetic_data.se_list],
        R_list=[r.copy() for r in synthetic_data.r_list],
        varY_list=synthetic_data.vary_list,
        population_sizes=synthetic_data.n_list,
        single_population_mac_thresh=0,
        low_memory_mode=False,
        variant_ids=variant_ids,
        **synthetic_data.common,
    )

    assert fit.variant_ids == variant_ids
    assert len(fit.sets) == 5
    cs_indices = fit.sets[0]
    cs_variant_ids = fit.sets[4]
    for idx_list, id_list in zip(cs_indices, cs_variant_ids):
        expected = [variant_ids[idx] for idx in idx_list]
        assert id_list == expected


def test_variant_ids_are_returned_when_no_credible_sets_survive():
    variant_ids = ["rs1", "rs2", "rs3"]

    fit = MultiSuSiE.multisusie_rss(
        b_list=[np.zeros(3)],
        s_list=[np.ones(3)],
        R_list=[np.eye(3)],
        varY_list=[1.0],
        population_sizes=[100],
        rho=np.eye(1),
        L=1,
        estimate_residual_variance=False,
        estimate_prior_method="EM",
        pop_spec_effect_priors=False,
        iter_before_zeroing_effects=0,
        max_iter=5,
        min_abs_corr=0,
        float_type=np.float64,
        single_population_mac_thresh=0,
        variant_ids=variant_ids,
    )

    assert fit.variant_ids == variant_ids
    assert isinstance(fit.sets, list)
    assert len(fit.sets) == 5
    assert fit.sets[0] == [[]]
    assert fit.sets[1] is None
    assert fit.sets[2] is None
    np.testing.assert_array_equal(fit.sets[3], [False])
    assert fit.sets[4] == [[]]


def test_individual_variant_ids_are_returned_when_no_credible_sets_survive():
    x = np.column_stack(
        (
            [1, -1, 1, -1, 1, -1, 1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1],
            [1, 1, 1, 1, -1, -1, -1, -1],
        )
    ).astype(float)
    y = np.array([1, -1, -1, 1, 1, -1, -1, 1], dtype=float)
    variant_ids = ["rs1", "rs2", "rs3"]

    fit = MultiSuSiE.multisusie(
        X_list=[x],
        Y_list=[y],
        rho=np.eye(1),
        L=1,
        estimate_residual_variance=False,
        estimate_prior_method="EM",
        pop_spec_effect_priors=False,
        iter_before_zeroing_effects=0,
        max_iter=5,
        min_abs_corr=0,
        variant_ids=variant_ids,
    )

    assert fit.variant_ids == variant_ids
    assert isinstance(fit.sets, list)
    assert len(fit.sets) == 5
    assert fit.sets[0] == [[]]
    assert fit.sets[1] is None
    assert fit.sets[2] is None
    np.testing.assert_array_equal(fit.sets[3], [False])
    assert fit.sets[4] == [[]]
