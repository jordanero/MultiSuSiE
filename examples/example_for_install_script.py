import MultiSuSiE
import numpy as np

print('Testing MultiSuSiE installation...')
print('Loading genotypes and creating simulated phenotypes')

geno_YRI = np.loadtxt('MultiSuSiE/example_data/geno_YRI.txt')
geno_CEU = np.loadtxt('MultiSuSiE/example_data/geno_CEU.txt')
geno_JPT = np.loadtxt('MultiSuSiE/example_data/geno_JPT.txt')
geno_list = [geno_YRI, geno_CEU, geno_JPT]

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

rng = np.random.default_rng(2)
y_list = [geno.dot(beta) + rng.standard_normal(geno.shape[0]) for (geno, beta) in zip(geno_list, beta_list)]
y_list = [y - np.mean(y) for y in y_list]

XTY_list = [geno.T.dot(y) for (geno, y) in zip(geno_list, y_list)]
XTX_diagonal_list = [np.diagonal(geno.T.dot(geno)) for geno in geno_list]
with np.errstate(divide='ignore',invalid='ignore'): # just to silence a divide by zero error
    beta_hat_list = [XTY / XTX_diag for (XTY, XTX_diag) in zip(XTY_list, XTX_diagonal_list)]

N_list = [geno.shape[0] for geno in geno_list]
residuals_list = [np.expand_dims(y, 1) - (geno * beta) for (y, geno, beta) in zip(y_list, geno_list, beta_hat_list)]
sum_of_squared_residuals_list = [np.sum(resid ** 2, axis = 0) for resid in residuals_list]
se_list = [np.sqrt(ssr / ((N - 2) * XTX)) for (ssr, N, XTX) in zip(sum_of_squared_residuals_list, N_list, XTX_diagonal_list)]

with np.errstate(divide='ignore',invalid='ignore'): # just to silence a divide by zero error
    R_list = [np.corrcoef(geno, rowvar = False) for geno in geno_list]

YTY_list = [y.dot(y) for y in y_list]
varY_list = [np.var(y, ddof = 1) for y in y_list]

z_list = [b/s for (b,s) in zip(beta_hat_list, se_list)]

print('Running MultiSuSiE with summary statistics')

ss_fit = MultiSuSiE.multisusie_rss(
    b_list = beta_hat_list,
    s_list = se_list,
    R_list = R_list,
    varY_list = varY_list,
    rho = np.array([[1, 0.75, 0.75], [0.75, 1, 0.75], [0.75, 0.75, 1]]),
    population_sizes = N_list,
    L = 10,
    low_memory_mode = False,
    recover_R = False,
    float_type = np.float64,
    single_population_mac_thresh = 0,
)

print('Running MultiSuSiE with individual level data')

indiv_fit = MultiSuSiE.multisusie(
    X_list = [g + 1 for g in geno_list],
    Y_list = y_list,
    rho = np.array([[1, 0.75, 0.75], [0.75, 1, 0.75], [0.75, 0.75, 1]]),
    L = 10,
    standardize = False,
    float_type = np.float64
)
print('Summary statistic and individual level fine-mapping have been run successfully')
print('The maximum difference in PIP between the two methods is {}. This number should be close to zero'.format(np.max(np.abs(ss_fit.pip - indiv_fit.pip))))
