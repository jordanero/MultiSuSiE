import numpy as np
import sys
#sys.path.insert(0,'/n/groups/price/jordan/MultiSuSiE/public_repo/MultiSuSiE/')
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
beta_list = [beta_YRI, beta_CEU, beta_JPT]

rng = np.random.default_rng(1)
y_list = [geno.dot(beta) + rng.standard_normal(geno.shape[0]) for (geno, beta) in zip(geno_list, beta_list)]
y_list = [y - np.mean(y) for y in y_list]

XTY_list = [geno.T.dot(y) for (geno, y) in zip(geno_list, y_list)]
XTX_diagonal_list = [np.diagonal(geno.T.dot(geno)) for geno in geno_list]
beta_hat_list = [XTY / XTX_diag for (XTY, XTX_diag) in zip(XTY_list, XTX_diagonal_list)]

N_list = [geno.shape[0] for geno in geno_list]
residuals_list = [np.expand_dims(y, 1) - (geno * beta) for (y, geno, beta) in zip(y_list, geno_list, beta_hat_list)]
sum_of_squared_residuals_list = [np.sum(resid ** 2, axis = 0) for resid in residuals_list]
se_list = [np.sqrt(ssr / (N * XTX)) for (ssr, N, XTX) in zip(sum_of_squared_residuals_list, N_list, XTX_diagonal_list)]

R_list = [np.corrcoef(geno, rowvar = False) for geno in geno_list]

YTY_list = [y.dot(y) for y in y_list]
varY_list = [np.var(y) for y in y_list]


# TEST 1: test that low_memory mode=False does not mutate R
R_list_copy = [R.copy() for R in R_list]
ss_fit = MultiSuSiE.susie_multi_rss(
    b_list = beta_hat_list[:2],
    s_list = se_list[:2],
    R_list = R_list[:2],
    varY_list = varY_list[:2],
    rho = np.array([[1, 0.75], [.75, 1]]),
    population_sizes = N_list[:2],
    L = 10,
    scaled_prior_variance = 0.2,
    mac_filter = 5,
    low_memory_mode = False,
    min_abs_corr = 0
)
assert(all([np.nanmax(np.abs(R_copy - R)) < 1e-10 for (R_copy, R) in zip(R_list_copy, R_list)]))
assert(all([(np.isnan(R_copy) == np.isnan(R)).all() for (R_copy, R) in zip(R_list_copy, R_list)]))


# TEST 2: test that recover_R_from_XTX recovers XTX
R_list_copy = [R.copy() for R in R_list]
XTX_list = []
XTY_list = []
for i in range(len(R_list)):
    XTX, XTY, n_censored = MultiSuSiE.recover_XTX_and_XTY(
        beta_hat_list[i],
        se_list[i],
        R_list_copy[i],
        varY_list[i],
        N_list[i],
        mac_filter = 0,
        maf_filter = 0
    )
    XTX_list.append(XTX)
    XTY_list.append(XTY)

X_l2_arr = np.array([np.diag(XTX) for XTX in XTX_list])
for i in range(len(XTX_list)):
    MultiSuSiE.recover_R_from_XTX(XTX_list[i], X_l2_arr[i])
assert(all([np.nanmax(np.abs(R - XTX)) < 1e-10 for (R, XTX) in zip(R_list, XTX_list)]))


