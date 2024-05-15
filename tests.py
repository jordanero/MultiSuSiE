import numpy as np
import sys
import MultiSuSiE

geno_YRI = np.loadtxt('example_data/geno_YRI.txt')
geno_CEU = np.loadtxt('example_data/geno_CEU.txt')
geno_JPT = np.loadtxt('example_data/geno_JPT.txt')
geno_list = [geno_CEU, geno_YRI, geno_JPT]

beta_YRI = np.zeros(40)
beta_CEU = np.zeros(40)
beta_JPT = np.zeros(40)
beta_YRI[10]=.75
beta_CEU[10]=1
beta_JPT[10]=.5
beta_YRI[3]=.5
beta_CEU[3]=.5
beta_JPT[3]=.5
beta_YRI[38]=1
beta_CEU[38]=0
beta_JPT[38]=0
beta_list = [beta_YRI, beta_CEU, beta_JPT]

rng = np.random.default_rng(1)
y_list = [geno.dot(beta) + rng.standard_normal(geno.shape[0]) for (geno, beta) in zip(geno_list, beta_list)]
y_list = [y - np.mean(y) for y in y_list]

XTY_list = [geno.T.dot(y) for (geno, y) in zip(geno_list, y_list)]
XTX_diagonal_list = [np.diagonal(geno.T.dot(geno)) for geno in geno_list]
with np.errstate(divide='ignore',invalid='ignore'):
    beta_hat_list = [XTY / XTX_diag for (XTY, XTX_diag) in zip(XTY_list, XTX_diagonal_list)]

N_list = [geno.shape[0] for geno in geno_list]
residuals_list = [np.expand_dims(y, 1) - (geno * beta) for (y, geno, beta) in zip(y_list, geno_list, beta_hat_list)]
sum_of_squared_residuals_list = [np.sum(resid ** 2, axis = 0) for resid in residuals_list]
se_list = [np.sqrt(ssr / ((N - 2) * XTX)) for (ssr, N, XTX) in zip(sum_of_squared_residuals_list, N_list, XTX_diagonal_list)]

with np.errstate(divide='ignore',invalid='ignore'):
    R_list = [np.corrcoef(geno, rowvar = False) for geno in geno_list]

YTY_list = [y.dot(y) for y in y_list]
varY_list = [np.var(y, ddof = 1) for y in y_list]

# TEST 1: test and summary statistic and individual level multiSuSiE give identical results
ss_fit = MultiSuSiE.multisusie_rss(
    b_list = beta_hat_list,
    s_list = se_list,
    R_list = R_list,
    varY_list = varY_list,
    rho = np.array([[1, 0.75, 0.75], [0.75, 1, 0.75], [0.75, 0.75, 1]]),
    population_sizes = N_list,
    L = 10,
    scaled_prior_variance = 0.2,
    low_memory_mode = False,
    min_abs_corr = 0,
    recover_R = False,
    float_type = np.float64,
    estimate_prior_method = 'EM',
    pop_spec_effect_priors = False,
    iter_before_zeroing_effects = 0,
    single_population_mac_thresh = 0,
)
indiv_fit = MultiSuSiE.multisusie(
    X_list = geno_list,
    Y_list = y_list,
    rho = np.array([[1, 0.75, 0.75], [0.75, 1, 0.75], [0.75, 0.75, 1]]),
    L = 10,
    scaled_prior_variance = 0.2,
    standardize = False,
    intercept = False,
    float_dtype = np.float64
)

assert(np.max(np.abs(ss_fit.pip - indiv_fit.pip)) < 1e-10)


# TEST 2: test that low_memory mode=False does not mutate anything
R_list_copy = [R.copy() for R in R_list]
beta_hat_list_copy = [b.copy() for b in beta_hat_list]
se_list_copy = [se.copy() for se in se_list]
ss_fit = MultiSuSiE.multisusie_rss(
    b_list = beta_hat_list,
    s_list = se_list,
    R_list = R_list,
    varY_list = varY_list,
    rho = np.array([[1, 0.75, 0.75], [0.75, 1, 0.75], [0.75, 0.75, 1]]),
    population_sizes = N_list,
    L = 10,
    scaled_prior_variance = 0.2,
    single_population_mac_thresh = 5,
    low_memory_mode = False,
    min_abs_corr = 0
)
assert(all([np.nanmax(np.abs(R_copy - R)) < 1e-10 for (R_copy, R) in zip(R_list_copy, R_list)]))
assert(all([(np.isnan(R_copy) == np.isnan(R)).all() for (R_copy, R) in zip(R_list_copy, R_list)]))
assert(all([np.nanmax(np.abs(beta_hat_copy - beta_hat)) < 1e-10 for (beta_hat_copy, beta_hat) in zip(beta_hat_list_copy, beta_hat_list)]))
assert(all([(np.isnan(beta_hat_copy) == np.isnan(beta_hat)).all() for (beta_hat_copy, beta_hat) in zip(beta_hat_list_copy, beta_hat_list)]))
assert(all([np.nanmax(np.abs(se_copy - se)) < 1e-10 for (se_copy, se) in zip(se_list_copy, se_list)]))
assert(all([(np.isnan(se_copy) == np.isnan(se)).all() for (se_copy, se) in zip(se_list_copy, se_list)]))


# TEST 3: test that recover_R_from_XTX recovers XTX
R_list_copy = [R.copy() for R in R_list]
XTX_list = []
XTY_list = []
for i in range(len(R_list)):
    XTX, XTY = MultiSuSiE.recover_XTX_and_XTY(
        beta_hat_list[i],
        se_list[i],
        R_list_copy[i],
        varY_list[i],
        N_list[i]
    )
    XTX_list.append(XTX)
    XTY_list.append(XTY)

X_l2_arr = np.array([np.diag(XTX) for XTX in XTX_list])
for i in range(len(XTX_list)):
    MultiSuSiE.recover_R_from_XTX(XTX_list[i], X_l2_arr[i])
assert(all([np.nanmax(np.abs(R - XTX)) < 1e-10 for (R, XTX) in zip(R_list, XTX_list)]))


# TEST 4: test that recover_XTX_and_XTY_from_Z recovers XTX and XTY with standardized genotypes and pehnotypes
z_list = [b/s for (b,s) in zip(beta_hat_list, se_list)]
with np.errstate(divide='ignore',invalid='ignore'):
    geno_std_list = [geno / np.std(geno, axis = 0, ddof = 1) for geno in geno_list]
y_std_list = [y / np.std(y, ddof = 1) for y in y_list]
XTX_std_list = [geno.T.dot(geno) for geno in geno_std_list]
XTY_std_list = [geno.T.dot(pheno) for (geno, pheno) in zip(geno_std_list, y_std_list)]

for i in range(len(z_list)):
    XTX, XTY = MultiSuSiE.recover_XTX_and_XTY_from_Z(
        z = z_list[i], 
        R = np.copy(R_list[i]),
        n = N_list[i],
        float_type = np.float64
    )
    assert(np.nanmax(np.abs(np.nan_to_num(XTX, 0) - XTX_std_list[i])) < 1e-10)
    assert(np.nanmax(np.abs(np.nan_to_num(XTY, 0) - XTY_std_list[i])) < 1e-10)

print('No tests failed!')


