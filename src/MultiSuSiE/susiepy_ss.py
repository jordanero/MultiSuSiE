import numpy as np
from tqdm import tqdm
import scipy.linalg as la
import time
import sys
import numba
import copy
import random
import functools
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize


class S: 
    def __init__(
            self, pop_sizes, L, XTX_list, scaled_prior_variance, 
            residual_variance, varY, prior_weights, float_type = np.float64):
        
        #code from init_setup
        num_pop = len(pop_sizes)
        p = XTX_list[0].shape[0]
        self.alpha = np.zeros((L, p), dtype=float_type) + 1.0/p
        self.mu = np.zeros((num_pop, L, p), dtype=float_type)
        self.mu2 = np.zeros((num_pop, num_pop, L, p), dtype=float_type)
        self.Xr_list = [np.zeros(XTX.shape[0], dtype=float_type) for XTX in XTX_list]
        self.sigma2 = residual_variance.astype(float_type)
        self.pi = prior_weights.astype(float_type)
        self.n = np.array(pop_sizes)
        self.L = L
        
        #code from init_finalize
        self.V = scaled_prior_variance * varY + np.zeros((num_pop, L), dtype=float_type)
        self.V = self.V.astype(float_type)
        assert np.all(self.V >= 0)
        self.ER2 = np.zeros(num_pop, dtype=float_type) + np.nan
        self.KL = np.zeros(L, dtype=float_type) + np.nan
        self.lbf = np.zeros(L, dtype=float_type) + np.nan
        
        self.converged = False

class SER_RESULTS:
    def __init__(self, alpha, mu, mu2, lbf, lbf_model, V):
        self.alpha = alpha
        self.mu = mu
        self.mu2 = mu2
        self.lbf = lbf
        self.V = V
        self.lbf_model = lbf_model

def susie_multi_rss(
    b_list,
    s_list,
    R_list,
    varY_list,
    rho,
    population_sizes,
    L = 10,
    scaled_prior_variance=0.2,
    prior_weights=None,
    standardize = False,
    pop_spec_standardization = False,
    estimate_residual_variance=True,
    estimate_prior_variance=True,
    estimate_prior_method='early_EM',
    pop_spec_effect_priors = True,
    iter_before_zeroing_effects = 5,
    prior_tol=1e-9,
    max_iter=100,
    tol=1e-3,
    verbose=False,
    coverage = 0.95,
    min_abs_corr = 0,
    float_type = np.float32,
    low_memory_mode = False,
    recover_R = False,
    mac_filter = 20,
    maf_filter = 0,
    ):
    """ Top-level function for running MultiSuSiE 

    This function takes takes standard GWAS summary statistics, converts
    them to sufficient statistics, and runs MultiSuSiE on them.

    Parameters
    ----------
    b_list: length K list of length P numpy arrays containing effect size 
        standard errors, one for each population
    s_list: length K list of length P numpy arrays containing effect size 
        standard errors, one for each population
    R_list: length K list of PxP numpy arrays representing the LD correlation 
        matrices for each population
    varY_list: length K list representing the sample variance of the outcome
        in each population
    rho: PxP numpy array representing the effect size correlation matrix
    population_sizes: list of integers representing the number of samples in
        the GWAS for each population
    L: integer representing the maximum number of causal variants
    scaled_prior_variance: float representing the effect size prior variance, 
        scaled by the residual variance
    prior_weights: numpy P-array of floats representing the prior probability
        of causality for each variant. Give None to use a uniform prior
    standardize: boolean, whether to adjust summmary statistics to be as if 
        genotypes were standardized to have mean 0 and variance 1
    pop_spec_standardization: boolean, if standardize is True, whether to 
        adjust summary statistics to be as if genotypes were standardized
        separately for each population, or pooled and then standardized
    estimate_residual_variance: boolean, whether to estimate the residual variance,
        $\sigma^2_k$ in the manuscript
    estimate_prior_variance: boolean, whether to estimate the prior variance,
        $A^{(l)}$ in the manuscript
    estimate_prior_method: string, method to estimate the prior variance. Recommended
        values are 'early_EM' or None
    pop_spec_effect_priors: boolean, whether to estimate separate prior 
        variance parameters for each population
    iter_before_zeroing_effects: integer, number of iterations to run before
        zeroing out component-population pairs (or components if 
        pop_spec_effect_priors is False) that have a lower likelihood than a 
        null model
    prior_tol: float which places a filter on the minimum prior variance
        for a component to be included when estimating PIPs
    max_iter: integer, maximum number of iterations to run
    tol: float, after iter_before_zeroing_effects iterations, results
        are returned if the ELBO increases by less than tol in an ieration
    verbose: boolean which indicates if an progress bar should be displayed
    coverage: float representing the minimum coverage of credible sets
    min_abs_corr: float representing the minimum absolute correlation between
        any pair of variants in a credible set. For each pair of variants,
        the max is taken across ancestries. In the case where min_abs_corr = 0,
        low_memory_mode = True, and recover_R = False, the purity of credible
        sets will not be calculated. 
    float_type: numpy float type used. Set to np.float32 to minimize memory
        consumption
    low_memory_mode: boolean, if True, the input R_list will be modified in place.
        Decreases memory consumption by a factor of two. BUT, THE INPUT R_list
        WILL BE OVERWRITTEN. If you need R_list, set low_memory_mode to False
    recover_R: boolean, if True, the R matrices will be recovered from XTX, 
        BUT variants with MAC/MAF estimate less than mac_filter/maf_filter will
        be censored. 
    mac_filter: float, if a variant has less than mac_filter minor alleles in a 
        population it will be censored in that population, but will still be 
        included with other populations. minor allele count is estimated under 
        an assumption of Hardy-Weinberg equilibrium
    maf_filter: float, if a variant has MAF less than maf_filter in a 
        population it will be censored in that population, but will still be 
        included with other populations. minor allele count is estimated under 
        an assumption of Hardy-Weinberg equilibrium

    
    Returns
    -------
    an object containing results with the following attributes:
        alpha: L x P numpy array of single-effect regression posterior 
            inclusion probabilities
        mu: K x L x P numpy array of single-effect regression effect size
            posterior means, conditional on each variant being the causal variant
        mu2: K x K x L x P numpy array of single-effect regression effect size
            posterior seconds moments, conditional on each variant being the 
            causal variant
        sigma2: length-K numpy array of residual variance estimates
        pi: length-P numpy array of prior inclusion probabilities
        n: length-K numpy array of sample sizes
        L: integer representing the maximum number of causal variants
        V: K x L numpy array of effect size prior variance estimates
        ER2: length-K numpy array of expected squared residuals
        KL: L x 1 numpy array of Kullback-Leibler divergences for each single
            effect regression
        lbf: L x 1 numpy array of log Bayes factors for each single effect 
            regression
        converged: boolean indicating whether the algorithm converged

    TODO
    ----
        - think about consequences of standardizing Y
        - Turn this into a package. Will probably have to figure out how to 
          deal with paths and imports. 
        - Add documentation throughout
        - Add command line interface?
        - Add tests, probably based on tests used for SuSiER
        - Make defaults match between sufficient statistic and summary statistic
          functions
    """


    if low_memory_mode:
        R_list_copy = R_list
        print('low memory mode is on. THE INPUT R MATRICES HAVE BEEN ' + \
            'TRANSFORMED INTO XTX AND CENSORED BASED ON MISSINGNESS. ' + \
            'THE INPUT R MATRICES HAVE BEEN CHANGED.')
    else:
        R_list_copy = [np.copy(R) for R in R_list]

    for i in range(len(b_list)):
        if b_list[i].dtype != float_type:
            b_list[i] = b_list[i].astype(float_type, copy = not low_memory_mode)
        if s_list[i].dtype != float_type:
            s_list[i] = s_list[i].astype(float_type, copy = not low_memory_mode)
        if R_list_copy[i].dtype != float_type:
            R_list_copy[i] = R_list_copy[i].astype(float_type, copy = not low_memory_mode)
        if rho.dtype != float_type:
            rho = rho.astype(float_type)
        varY_list = np.array(varY_list, dtype = float_type)


    XTX_list = []
    XTY_list = []
    YTY_list = []

    for i in range(len(b_list)):
        YTY = varY_list[i] * (population_sizes[i] - 1)
        XTX, XTY, n_censored = recover_XTX_and_XTY(
            b = b_list[i],
            s = s_list[i],
            R = R_list_copy[i],
            YTY = YTY,
            n = population_sizes[i],
            mac_filter = mac_filter,
            maf_filter = maf_filter
        )
        XTX_list.append(XTX)
        XTY_list.append(XTY)
        YTY_list.append(YTY)
        if n_censored > 0:
            print('censored %d variants in population %d'%(n_censored, i))


    return susie_multi_ss(
        XTX_list = XTX_list, 
        XTY_list = XTY_list, 
        YTY_list = YTY_list, 
        rho = rho, 
        population_sizes = population_sizes,
        L = L, 
        scaled_prior_variance = scaled_prior_variance,
        prior_weights = prior_weights,
        standardize = standardize,
        pop_spec_standardization = pop_spec_standardization,
        estimate_residual_variance = estimate_residual_variance,
        estimate_prior_variance = estimate_prior_variance,
        estimate_prior_method = estimate_prior_method,
        prior_tol = prior_tol,
        max_iter = max_iter,
        tol = tol,
        verbose = verbose,
        iter_before_zeroing_effects = iter_before_zeroing_effects,
        pop_spec_effect_priors = pop_spec_effect_priors,
        R_list = R_list,
        coverage = coverage,
        min_abs_corr = min_abs_corr,
        float_type = float_type,
        low_memory_mode = low_memory_mode,
        recover_R = recover_R
    )

# THIS FUNCTION MUTATES INPUT R. BE CAREFUL
def recover_XTX_and_XTY(b, s, R, YTY, n, mac_filter  = 0, maf_filter = 0):
    sigma2 = YTY / ((b / s) ** 2 + n - 2)
    XTY = np.nan_to_num(sigma2 * b / (s ** 2))
    dR = np.nan_to_num(sigma2 / (s ** 2))
    R *= np.expand_dims(np.sqrt(dR), axis = 1)
    R *=  np.sqrt(dR)

    maf_filter = np.maximum(maf_filter, mac_filter / (2 * n))

    var_x = np.minimum(dR / (n - 1), .5) # this can be > .5 due to HWE violations?
    maf = 1 / 2 - np.sqrt(1 - 2 * var_x) / 2
    mask = maf < maf_filter
    R[mask, :] = 0
    R[:, mask] = 0
    XTY[mask] = 0

    return(R, XTY, np.sum(mask))

def susie_multi_ss(
    XTX_list, XTY_list, YTY_list,
    rho,
    population_sizes,
    L = 10,
    scaled_prior_variance=0.2,
    prior_weights=None,
    standardize = True,
    pop_spec_standardization = False,
    estimate_residual_variance=True,
    estimate_prior_variance=True,
    estimate_prior_method='early_EM',
    pop_spec_effect_priors = True,
    iter_before_zeroing_effects = 5,
    prior_tol=1e-9,
    max_iter=100,
    tol=1e-3,
    verbose=False,
    R_list = None,
    coverage = .95,
    min_abs_corr = .5,
    float_type = np.float32,
    low_memory_mode = False,
    recover_R = False
    ):

    #check input
    assert len(XTX_list) == len(XTY_list)
    assert np.all([XTX.shape[1] == XTX_list[0].shape[1] for XTX in XTX_list])
    assert np.all([XTX.shape[0] == XTY.shape[0] for (XTX,XTY) in zip(XTX_list, XTY_list)])
    if prior_weights is not None:
        prior_weights = prior_weights.astype(float_type)

    assert not np.any([np.any(np.isnan(XTX)) for XTX in XTX_list])
    
        
    #compute w_pop (the relative size of each population)
    population_sizes = np.array(population_sizes)
    w_pop = (population_sizes/ population_sizes.sum()).astype(float_type)

    
    #compute rho properties
    rho = rho.astype(float_type)
    inv_rho = la.inv(rho).astype(float_type)
    logdet_rho_sign, logdet_rho = np.linalg.slogdet(rho)
    assert logdet_rho_sign>0


    X_l2_arr = np.array([np.diag(XTX) for XTX in XTX_list], dtype = float_type)

    if standardize:
        csd = np.sqrt(X_l2_arr/(np.expand_dims(population_sizes, axis = 1) - 1))
        if not pop_spec_standardization:
            csd = csd.T.dot(w_pop)
            csd = csd * np.ones((len(XTX_list), csd.shape[0]))
        is_constant_column = np.isclose(csd, 0.0)
        csd[is_constant_column] = 1.0
        for pop_i in range(len(XTX_list)):
            XTX_list[pop_i] *= (1 / csd[pop_i, :])
            XTX_list[pop_i] *= (1 / np.expand_dims(csd[pop_i, :], 1))
            XTY_list[pop_i] = XTY_list[pop_i] / csd[pop_i, :]
    else:
        csd = np.ones((len(XTX_list), XTX_list[0].shape[1]))

    X_l2_arr = np.array([np.diag(XTX) for XTX in XTX_list], dtype = float_type)
    
    #init setup
    p = XTX_list[0].shape[1]

    varY = np.array([YTY/(n-1) for (YTY, n) in zip(YTY_list, population_sizes)])
    varY_pooled = np.sum(YTY_list) / (np.sum(population_sizes) - 1)
    residual_variance = varY
    if np.isscalar(scaled_prior_variance) & standardize:
        assert 0 < scaled_prior_variance <= 1
    if prior_weights is None:
        prior_weights = np.zeros(p, dtype=float_type) + 1.0/p
    else:
        prior_weights = (prior_weights / np.sum(prior_weights)).astype(float_type)
    assert prior_weights.shape[0] == p
    if p<L: L=p
    s = S(
        population_sizes, L, XTX_list, scaled_prior_variance, residual_variance,
        varY_pooled, prior_weights, float_type = float_type
    )
    elbo = np.zeros(max_iter+1) + np.nan
    elbo[0] = -np.inf
    check_null_threshold = 0.0

    
    ### start iterations ###
    tqdm_iter = tqdm(list(range(max_iter)), disable=not verbose, file=sys.stdout)
    for i in tqdm_iter:
        tqdm_iter.set_description('iteration %d/%d'%(i+1, max_iter))

        if estimate_prior_method == 'early_EM':
            if i == 0: 
                current_estimate_prior_method = None
            else:
                current_estimate_prior_method = 'early_EM'
        elif (estimate_prior_method is not None) and (estimate_prior_method.split('_')[-1] == 'init'):
            if i == 0:
                current_estimate_prior_method = estimate_prior_method.split('_')[0]
            else:
                current_estimate_prior_method = 'EM'
        else:
            current_estimate_prior_method = estimate_prior_method

        if i < iter_before_zeroing_effects:
            current_check_null_threshold = -np.inf
        else:
            current_check_null_threshold = check_null_threshold

        old_s = copy.deepcopy(s)
        s = update_each_effect(
            XTX_list, XTY_list, s, X_l2_arr, w_pop,
            rho, inv_rho, logdet_rho,
            estimate_prior_variance, current_estimate_prior_method,
            verbose=verbose,
            check_null_threshold = current_check_null_threshold,
            pop_spec_effect_priors = pop_spec_effect_priors,
            float_type = float_type
        )
        if i == 0:
            import pickle
            pickle.dump(s, open('/n/groups/price/jordan/MultiSuSiE/data/misc/s_ss.pkl', 'wb'))

        update_ER2(XTX_list, XTY_list, YTY_list, s, X_l2_arr)
        elbo[i+1] = get_objective(XTX_list, XTY_list, s, YTY_list, X_l2_arr)
        if verbose:
            print('objective: %s'%(elbo[i+1]))
    
        if ((elbo[i+1] - elbo[i]) < tol) and (i >= (iter_before_zeroing_effects + 1)):
            s.converged = True
            tqdm_iter.close()
            break

        if estimate_residual_variance:
            s.sigma2 = estimate_residual_variance_func(
                XTX_list, XTY_list, YTY_list, s, X_l2_arr, population_sizes, 
                float_type = float_type
            )
            if np.any(s.sigma2 < 0):
                print('minimum resdiual variance less than 0. Is there mismatch between the correlation matrix and association statistics?')
    elbo = elbo[1:i+2] # Remove first (infinite) entry, and trailing NAs.
    s.elbo = elbo
    s.niter = i+1



    if not s.converged:
        print('IBSS algorithm did not converge in %d iterations'%(max_iter))
        
    s.intercept = np.zeros(len(XTX_list))
    s.fitted = s.Xr_list
        

    s.pip = susie_get_pip(s, prior_tol=prior_tol)
    if standardize:
        s.X_column_scale_factors = csd.copy()
        s.X_column_scale_factors[is_constant_column] = 0.0
    s.coef = np.array([np.squeeze(np.sum(s.mu[k] * s.alpha, axis=0) / csd[k, :]) for k in range(len(XTX_list))])
    s.coef_sd = np.array([(np.squeeze(np.sqrt(np.sum(s.alpha * s.mu2[k, k] - (s.alpha*s.mu[k])**2, axis=0)) / csd[k, :])) for k in range(len(XTX_list))])

    if (low_memory_mode and min_abs_corr > 0) or (low_memory_mode and recover_R):
        for i in range(len(R_list)):
            recover_R_from_XTX(R_list[i], X_l2_arr[i])

    s.sets = susie_get_cs(
        s = s, R_list = R_list, coverage = coverage, min_abs_corr = min_abs_corr, 
        dedup = True, n_purity = np.inf, 
        calculate_purity = (not low_memory_mode) or (min_abs_corr > 0) or recover_R
    )

    return s

def update_each_effect(
    XTX_list, XTY_list, s, X_l2_arr, w_pop, rho, inv_rho, logdet_rho,
    estimate_prior_variance=False, estimate_prior_method='optim',
    check_null_threshold=0.0, verbose=False, pop_spec_effect_priors = False, float_type = np.float64):

        
    if not estimate_prior_variance:
        estimate_prior_method = None
    L = s.alpha.shape[0]
    num_pop = len(XTX_list)

    for l in range(L):
        R_list = []
        for k in range(len(XTX_list)):
            s.Xr_list[k] -= XTX_list[k].dot(s.alpha[l] * s.mu[k,l])
            R_list.append(XTY_list[k] - s.Xr_list[k])
        
        res = single_effect_regression(
            R_list, XTX_list, s.V[:,l], X_l2_arr, w_pop, rho, inv_rho, 
            logdet_rho, residual_variance=s.sigma2, prior_weights=s.pi,
            optimize_V=estimate_prior_method, 
            check_null_threshold=check_null_threshold, verbose=verbose,
            pop_spec_effect_priors = pop_spec_effect_priors,
            alpha = s.alpha[l,:], mu2 = s.mu2[:,:,l,:],
            float_type = float_type
        )
              
        # Update the variational estimate of the posterior mean.
        s.mu[:,l,:] = res.mu
        s.alpha[l,:] = res.alpha
        s.mu2[:,:,l,:] = res.mu2
        s.V[:,l] = res.V
        s.lbf[l] = res.lbf_model
        s.KL[l] = -res.lbf_model + SER_posterior_e_loglik(X_l2_arr, R_list, s.sigma2, res.mu * res.alpha, res.mu2[range(num_pop), range(num_pop)] * res.alpha)
        for k in range(len(XTX_list)):
            s.Xr_list[k] += XTX_list[k].dot(s.alpha[l] * s.mu[k,l])
        
    return s

def single_effect_regression(
    XTY_list, XTX_list, V, X_l2_arr, w_pop, rho, inv_rho, logdet_rho,
    residual_variance, prior_weights=None, optimize_V=None, 
    check_null_threshold=0, verbose=False, pop_spec_effect_priors = False,
    alpha = None, mu2 = None, float_type = np.float64):
    
    #optimize V if needed (V is sigma_0^2 in the paper)
    compute_lbf_params = (XTY_list, XTX_list, X_l2_arr, rho, inv_rho, 
        logdet_rho, residual_variance, False, verbose, float_type)
    if optimize_V not in ['EM', 'EM_corrected', None]:
        V = optimize_prior_variance(
            optimize_V, prior_weights, rho, 
            compute_lbf_params=compute_lbf_params, alpha=alpha, post_mean2=mu2, 
            w_pop=w_pop, check_null_threshold=check_null_threshold, 
            pop_spec_effect_priors = pop_spec_effect_priors, current_V = V,
            float_type = float_type
        )
        
    #compute lbf (log Bayes-factors)
    lbf, post_mean, post_mean2 = compute_lbf(
        V, XTY_list, XTX_list, X_l2_arr, rho, inv_rho, logdet_rho, 
        residual_variance, return_moments=True, verbose=verbose,
        float_type = float_type
    )

    
    #compute alpha as defined in Appendix A.2
    maxlbf = np.max(lbf)
    w = np.exp(lbf - maxlbf)
    w_weighted = w * prior_weights
    weighted_sum_w = w_weighted.sum()
    alpha = w_weighted / weighted_sum_w
    
    #compute log-likelihood (equation A.5)
    lbf_model = maxlbf + np.log(weighted_sum_w)
        
    if optimize_V in ['EM', 'EM_corrected']:
        V = optimize_prior_variance(
            optimize_V, prior_weights, rho, 
            compute_lbf_params=compute_lbf_params, alpha=alpha, 
            post_mean2=post_mean2, w_pop=w_pop, 
            check_null_threshold=check_null_threshold, 
            pop_spec_effect_priors = pop_spec_effect_priors, 
            float_type = float_type
        )
        
    res = SER_RESULTS(alpha=alpha, mu=post_mean, mu2=post_mean2, lbf=lbf, lbf_model=lbf_model, V=V)


    return res

def optimize_prior_variance(
    optimize_V, prior_weights, rho, compute_lbf_params=None, alpha=None, post_mean2=None, w_pop=None, check_null_threshold=0, 
    pop_spec_effect_priors = False, current_V = None, float_type = np.float64):

    K = rho.shape[0]
    if optimize_V == 'optim':
        if pop_spec_effect_priors:
            raise Exception('estimate_prior_method="optim" with ' +
                            'pop_spec_effect_priors=True has not been implemented')
        else:
            neg_loglik_logscale = lambda lV: -loglik(np.array([np.exp(lV) for i in range(K)]), prior_weights, compute_lbf_params)
            opt_obj = minimize_scalar(neg_loglik_logscale, bounds=(-30,15))
            lV = opt_obj.x
            V = np.exp(lV)
    elif optimize_V in ['EM', 'early_EM']:
        V = np.array([np.sum(alpha * post_mean2[i, i]) for i in range(K)], dtype=float_type)
        if not pop_spec_effect_priors:
            V = (w_pop.dot(V)).astype(float_type)
    elif optimize_V == 'grid':
        if pop_spec_effect_priors:
            raise Exception('estimate_prior_method="grid" with ' +
                            'pop_spec_effect_priors=True has not been implemented')
        else:
            V_arr = np.logspace(-7, -1, 13)
            llik_arr = np.array([loglik(V, prior_weights, compute_lbf_params) for V in V_arr])
            V = V_arr[np.argmax(llik_arr)]
    else:
        raise ValueError('unknown optimization method')

    if not pop_spec_effect_priors:
        V = V * np.ones(post_mean2.shape[0])
        # set V exactly 0 if that beats the numerical value by check_null_threshold in loglik.
        delta_loglik = loglik(0, prior_weights, compute_lbf_params) + check_null_threshold - loglik(V, prior_weights, compute_lbf_params)
        if np.isclose(delta_loglik, 0) or delta_loglik >= 0:
            V=0
    else:
        if check_null_threshold == (-1 * np.inf):
            return V
        elif np.all(np.isclose(V, 0)):
            return 0
        # Compare our current effect prior to null models 
        else:
            V_list = [V]
            # zero out each population, one at a time
            for i in range(K):
                if not np.isclose(V[i],0):
                    V_copy = V.copy()
                    V_copy[i] = 0
                    V_list.append(V_copy)
            V_list.append(np.array([np.zeros(K, dtype=float_type)]))
        llik_arr = np.array([loglik(np.array(V), prior_weights, compute_lbf_params) for V in V_list])
        llik_arr = llik_arr + np.array([0] + [check_null_threshold for i in range(len(V_list) - 1)])
        V = V_list[np.argmax(llik_arr)]
    if isinstance(V, np.ndarray):
        V[V < 0] = 0
    elif V < 0:
        V = 0


    return V

def loglik(V, prior_weights, compute_lbf_params):
    lbf = compute_lbf(V, *compute_lbf_params)
    maxlbf = np.max(lbf)
    w = np.exp(lbf - maxlbf)
    w_weighted = w * prior_weights
    weighted_sum_w = w_weighted.sum()
    loglik = maxlbf + np.log(weighted_sum_w)
    return loglik

def compute_lbf(
    V, XTY_list, XTX_list, X_l2_arr, rho, inv_rho, logdet_rho, 
    residual_variance, return_moments=False, verbose=False, 
    float_type = np.float64):
    
    num_variables = XTX_list[0].shape[0]
    num_pops = len(XTX_list)

    if np.all(np.isclose(V, 0)):
        num_pops = len(XTY_list)
        num_variables = len(XTY_list[0])
        lbf = np.zeros(num_variables, dtype=float_type)
        if return_moments:
            post_mean = np.zeros((num_pops, num_variables), dtype=float_type)
            post_mean2 = np.zeros((num_pops, num_pops, num_variables), dtype=float_type)

    elif np.any(np.isclose(V, 0)):
        nonzero_pops = np.flatnonzero(~np.isclose(V, 0))
        lbf_out = compute_lbf(
            V[nonzero_pops], 
            [XTY_list[i] for i in nonzero_pops], 
            [XTX_list[i] for i in nonzero_pops], 
            X_l2_arr[nonzero_pops], rho[nonzero_pops, :][:, nonzero_pops], None, None, 
            residual_variance[nonzero_pops], return_moments, 
            verbose, float_type = float_type
        )
        if return_moments:
            post_mean = np.zeros((num_pops, num_variables), dtype=float_type)
            post_mean2 = np.zeros((num_pops, num_pops, num_variables), dtype=float_type)
            lbf = lbf_out[0]
            post_mean[nonzero_pops] = lbf_out[1]
            post_mean2[np.ix_(nonzero_pops, nonzero_pops)] = lbf_out[2]
        else:
            lbf = lbf_out

    else:
        XTY = np.ascontiguousarray(np.stack(XTY_list, axis = 1))
        if return_moments:
            try:
                lbf, post_mean, post_mean2 = compute_lbf_and_moments(
                    V, XTY, X_l2_arr, rho, inv_rho, logdet_rho, 
                    residual_variance, verbose, float_type = float_type
                )
            except:
                lbf, post_mean, post_mean2 = compute_lbf_and_moments_safe(
                    V, XTY, X_l2_arr, rho, inv_rho, logdet_rho, 
                    residual_variance, verbose, float_type = float_type
                )
        else:
            lbf = compute_lbf_no_moments(
                V, XTY, X_l2_arr, rho, inv_rho, logdet_rho,
                residual_variance, verbose, float_type = float_type
            )

    if return_moments:
        return lbf, post_mean, post_mean2
    else:
        return lbf

@numba.jit(nopython=True, cache=False)
def compute_lbf_no_moments(
    V, XTY, X_l2_arr, rho, inv_rho, logdet_rho, 
    residual_variance, verbose, float_type = np.float64):
    
    
    num_pops = XTY.shape[1]
    num_variables = XTY.shape[0]

    lbf = np.zeros(num_variables, dtype=float_type)
    
    YT_invD_Z = XTY / residual_variance

    #compute a (the effects covariance matrix) and its inverse and log-determinant
    A = rho * np.sqrt(np.outer(V, V))
    inv_A = np.linalg.inv(A)
    logdet_A_sign, logdetA = np.linalg.slogdet(A)

    for i in range(num_variables):
    
        #compute the diagonal of q = z.t * inv(d) * z (this is a diagonal matrix)
        # this is (n_pop)
        Q_diag = X_l2_arr[:,i]/residual_variance

        #compute log-determinent for inv(a)+q
        # this is (n_pop, n_pop)
        Ainv_plus_Q = inv_A + np.diag(Q_diag)
        logdet_Ainv_plus_Q_sign, logdet_Ainv_plus_Q = np.linalg.slogdet(Ainv_plus_Q)
        assert logdet_Ainv_plus_Q_sign>0
    
        #compute inv_ainv_plus_q_times_zt_invd_y
        inv_Ainv_plus_Q_times_ZT_invD_Y = np.linalg.solve(Ainv_plus_Q, YT_invD_Z[i,:])

        #compute log-bf for this variable
        lbf_1 = 0.5 * YT_invD_Z[i,:].dot(inv_Ainv_plus_Q_times_ZT_invD_Y)
        lbf_2 = -0.5 * (logdetA + logdet_Ainv_plus_Q)
        lbf[i] = lbf_1 + lbf_2
        
    return lbf


@numba.jit(nopython=True, cache=False)
def compute_lbf_and_moments(
    V, XTY, X_l2_arr, rho, inv_rho, logdet_rho, 
    residual_variance, verbose, float_type = np.float64):

    num_pops = XTY.shape[1]
    num_variables = XTY.shape[0]

    lbf = np.zeros(num_variables, dtype=float_type)
    post_mean = np.zeros((num_pops, num_variables), dtype=float_type)
    post_mean2 = np.zeros((num_pops, num_pops, num_variables), dtype=float_type)
        

    YT_invD_Z = XTY / residual_variance

    #compute a (the effects covariance matrix) and its inverse and log-determinant
    A = rho * np.sqrt(np.outer(V, V))
    inv_A = np.linalg.inv(A)
    logdet_A_sign, logdetA = np.linalg.slogdet(A)

    for i in range(num_variables):
    
        #compute the diagonal of q = z.t * inv(d) * z (this is a diagonal matrix)
        Q_diag = X_l2_arr[:,i] / residual_variance

        #compute log-determinent for inv(a)+q
        Ainv_plus_Q = inv_A + np.diag(Q_diag)
        logdet_Ainv_plus_Q_sign, logdet_Ainv_plus_Q = np.linalg.slogdet(Ainv_plus_Q)
        assert logdet_Ainv_plus_Q_sign>0
    
        #compute inv_ainv_plus_q_times_zt_invd_y
        inv_Ainv_plus_Q_times_ZT_invD_Y = np.linalg.solve(Ainv_plus_Q, YT_invD_Z[i, :])

        #compute log-bf for this variable
        lbf_1 = 0.5 * YT_invD_Z[i, :].dot(inv_Ainv_plus_Q_times_ZT_invD_Y)
        lbf_2 = -0.5 * (logdetA + logdet_Ainv_plus_Q)
        lbf[i] = lbf_1 + lbf_2

        #compute posterior moments for this variable
        AQ = A*Q_diag
        post_mean[:,i] = A.dot(YT_invD_Z[i, :]) - AQ.dot(inv_Ainv_plus_Q_times_ZT_invD_Y)
        post_covar_i = A - AQ.dot(A) + AQ.dot(np.linalg.solve(Ainv_plus_Q, AQ.T))
        post_mean2[:, :, i] = np.maximum(post_covar_i + np.outer(post_mean[:,i], post_mean[:,i]), 0)

    return lbf, post_mean, post_mean2

def compute_lbf_and_moments_safe(
    V, XTY, X_l2_arr, rho, inv_rho, logdet_rho, 
    residual_variance, verbose, float_type = np.float64):

    num_pops = XTY.shape[1]
    num_variables = XTY.shape[0]

    lbf = np.zeros(num_variables, dtype=float_type)
    post_mean = np.zeros((num_pops, num_variables), dtype=float_type)
    post_mean2 = np.zeros((num_pops, num_pops, num_variables), dtype=float_type)
        

    YT_invD_Z = XTY / residual_variance

    #compute a (the effects covariance matrix) and its inverse and log-determinant
    A = rho * np.sqrt(np.outer(V, V))
    inv_A = np.linalg.inv(A)
    logdet_A_sign, logdetA = np.linalg.slogdet(A)

    for i in range(num_variables):

        #compute the diagonal of q = z.t * inv(d) * z (this is a diagonal matrix)
        Q_diag = X_l2_arr[:,i]/residual_variance

        #compute log-determinent for inv(a)+q
        Ainv_plus_Q = inv_A + np.diag(Q_diag)
        logdet_Ainv_plus_Q_sign, logdet_Ainv_plus_Q = np.linalg.slogdet(Ainv_plus_Q)
        assert logdet_Ainv_plus_Q_sign>0

        #compute inv_ainv_plus_q_times_zt_invd_y
        inv_Ainv_plus_Q_times_ZT_invD_Y = np.linalg.solve(Ainv_plus_Q, YT_invD_Z[i, :])

        #compute log-bf for this variable
        lbf_1 = 0.5 * YT_invD_Z[i, :].dot(inv_Ainv_plus_Q_times_ZT_invD_Y)
        lbf_2 = -0.5 * (logdetA + logdet_Ainv_plus_Q)
        lbf[i] = lbf_1 + lbf_2

        #compute posterior moments for this variable
        AQ = A*Q_diag
        post_mean[:,i] = A.dot(YT_invD_Z[i, :]) - AQ.dot(inv_Ainv_plus_Q_times_ZT_invD_Y)
        post_covar_i = A - AQ.dot(A) + AQ.dot(np.linalg.solve(Ainv_plus_Q, AQ.T))
        post_mean2[:, :, i] = post_covar_i + np.outer(post_mean[:,i], post_mean[:,i])

    return lbf, post_mean, post_mean2

def get_objective(XTX_list, XTY_list, s, YTY_list, X_l2_arr):
    return Eloglik(XTX_list, XTY_list, s, YTY_list, X_l2_arr) - np.sum(s.KL)

def Eloglik(XTX_list, XTY_list, s, YTY_list, X_l2_arr):
    result = -0.5 * s.n.dot(np.log(2*np.pi*s.sigma2))
    for i in range(len(XTX_list)):
        result -= 0.5/s.sigma2[i] * s.ER2[i]
    return result

def update_ER2(XTX_list, XTY_list, YTY_list, s, X_l2_arr):
    for i in range(len(XTX_list)):
        s.ER2[i] = get_ER2(
            XTX_list[i], XTY_list[i], YTY_list[i], s.alpha, s.mu[i], 
            s.mu2[i, i], X_l2_arr[i]
        )

def get_ER2(XTX, XTY, YTY, alpha, mu, mu2, X_l2):
    B = alpha * mu # beta should be lxp
    XB2 = np.sum(B.T * np.dot(XTX, B.T))
    betabar = np.sum(B, axis = 0)
    postb2 = alpha * mu2

    result = YTY - 2 * betabar.dot(XTY) + betabar.dot(np.dot(XTX, betabar)) - XB2 + np.sum(X_l2 * postb2) 
    return result     

def SER_posterior_e_loglik(X_l2_arr, XTY_list, s2, Eb, Eb2):
    result = 0
    for i in range(len(X_l2_arr)):
        result -= .5 / s2[i] * (-2 * XTY_list[i].dot(Eb[i]) + X_l2_arr[i].dot(Eb2[i]))
    return result

def estimate_residual_variance_func(
        XTX_list, XTY_list, YTY_list, s, X_l2_arr, population_sizes, 
        float_type = np.float64):

    sigma2_arr = np.zeros(len(XTX_list), dtype=float_type)
    for i in range(len(XTX_list)):
        sigma2_arr[i] =  s.ER2[i] / population_sizes[i]
    return sigma2_arr

def susie_get_pip(s, prior_tol=1e-9):
    include_idx = np.any(s.V > prior_tol, axis = 0)
    if not np.any(include_idx):
        return np.zeros(s.alpha.shape[1])
    res = s.alpha[include_idx, :]
    pips = 1 - np.prod(1-res, axis=0)
    return pips

def susie_get_cs(
    s, R_list, coverage = 0.95, min_abs_corr = 0.5, dedup = True, n_purity = 100,
    calculate_purity = True):


    include_mask = np.any(s.V > 1e-9, axis = 0)

    status = in_CS(s.alpha, coverage)
    cs = [np.argwhere(status[i, :] == 1).flatten() for i in range(status.shape[0])]
    cs = [cs[i] if include_mask[i] else [] for i in range(len(cs))]
    claimed_coverage = np.array([np.sum(s.alpha[i, cs[i]]) for i in range(len(cs))])
    include_mask = include_mask & [len(csi) > 0 for csi in cs]

    if dedup:
        cs_set = set()
        for i in range(len(cs)):
            if tuple(cs[i]) in cs_set:
                include_mask[i] = False
            elif include_mask[i]:
                cs_set.add(tuple(cs[i]))
    if not np.any(include_mask):
        return ([[] for i in range(len(include_mask))], None, None, include_mask)

    if calculate_purity:
        purity = np.array([get_purity_x(cs[i], R_list, min_abs_corr, n_purity) if include_mask[i] else np.NaN for i in range(len(cs))])
        include_mask[purity < min_abs_corr] = False
    else:
        purity = np.array([np.NaN for i in range(len(cs))])

    return (cs, purity, claimed_coverage, include_mask)

def in_CS(alpha, coverage):
    return np.apply_along_axis(in_CS_x, 1, alpha, coverage)

def in_CS_x(alpha, coverage):
    n = n_in_CS_x(alpha, coverage)
    o = np.argsort(alpha)[::-1]
    result = np.zeros(alpha.shape, dtype=np.int32)
    result[o[:n]] = 1
    return result

def n_in_CS(alpha, coverage):
    return np.apply_along_axis(n_in_CS_x, 1, alpha, coverage)

def n_in_CS_x(alpha, coverage):
    return np.sum(np.cumsum(np.sort(alpha)[::-1]) < coverage) + 1

def get_purity_x(cs, R_list, min_abs_cor, n_purity):
    if len(cs) > n_purity:
        cs = random.sample(cs.tolist(), n_purity)
    abs_meta_R = functools.reduce(np.maximum, [np.abs(R[cs, :][:, cs]) for R in R_list])
    return np.min(abs_meta_R)

def recover_R_from_XTX(XTX, X_l2):
    assert((XTX[:,np.flatnonzero(X_l2 == 0)] == 0).all())
    assert((XTX[np.flatnonzero(X_l2 == 0),:] == 0).all())
    with np.errstate(divide='ignore', invalid = 'ignore'):
        XTX /= np.sqrt(np.nan_to_num(X_l2, 1))
        XTX /= np.sqrt(np.expand_dims(np.nan_to_num(X_l2, 1), 1))
