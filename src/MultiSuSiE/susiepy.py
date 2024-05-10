import numpy as np; np.set_printoptions(precision=3)
from tqdm import tqdm #
import scipy.stats as stats
import scipy.linalg as la
import logging
import time
import sys
from scipy.optimize import minimize_scalar

#TODO:
# - implement population specific effect priors


class S:
    def __init__(self, X_std_list, L, scaled_prior_variance, residual_variance, varY, prior_weights, null_weight, float_dtype):
        
        #code from init_setup
        num_pop = len(X_std_list)
        p = X_std_list[0].shape[1]
        self.alpha = np.zeros((L, p), dtype=float_dtype) + 1.0/p
        self.mu = np.zeros((num_pop, L, p), dtype=float_dtype)
        self.mu2 = np.zeros((num_pop, L, p), dtype=float_dtype)
        self.Xr_list = [np.zeros(X.shape[0], dtype=float_dtype) for X in X_std_list]
        self.sigma2 = residual_variance.astype(float_dtype)
        ###assert np.isscalar(self.sigma2)
        self.pi = prior_weights.astype(float_dtype)
        self.has_null_index = (null_weight is not None)
        self.n = np.array([X.shape[0] for X in X_std_list])
        
        #code from init_finalize
        self.V = scaled_prior_variance*varY + np.zeros(L, dtype=float_dtype)
        self.V = self.V.astype(float_dtype)
        assert np.all(self.V >= 0)
        self.KL = np.zeros(L, dtype=float_dtype) + np.nan
        self.lbf = np.zeros(L, dtype=float_dtype) + np.nan
        
        self.converged = False


class SER_RESULTS:
            def __init__(self, alpha, mu, mu2, lbf, lbf_model, V, loglik):
                self.alpha = alpha
                self.mu = mu
                self.mu2 = mu2
                self.lbf = lbf
                self.lbf_model = lbf_model
                self.V = V
                self.loglik = loglik


#expected squared residuals
def get_ER2(X_std, Y, alpha, mu, mu2, Xr, X_l2):
    Xr_L = X_std.dot((alpha*mu).T)
    postb2 = alpha*mu2
    r = Y - Xr
    result = r.dot(r) - np.einsum('ij,ij->',Xr_L, Xr_L) + np.sum(X_l2.dot(postb2.T))
    return result     


#posterior expected loglikelihood for a single effect regression (#Equation B.6 - B.9 (after expanding the L2-norm))
def SER_posterior_e_loglik(X_std_list, Y_list, s2, Eb, Eb2, X_l2_arr, n):
    ''' 
    Eb the posterior mean of b (p vector) (alpha * mu)
    Eb2 the posterior second moment of b (p vector) (alpha * mu2)
    '''
    result = -0.5 * n.dot(np.log(2*np.pi*s2))
    for i in range(len(Y_list)):
        Y = Y_list[i]
        X_std = X_std_list[i]
        result -= 0.5/s2[i] * (Y.dot(Y) - 2*Y.dot(X_std.dot(Eb[i])) + X_l2_arr[i].dot(Eb2[i]))
        #result -= 0.5/s2[i] * (np.sum(Y * Y) - 2*Y.dot(X_std.dot(Eb[i])) + X_l2_arr[i].dot(Eb2[i]))
        #result -= 0.5/s2[i] * (np.sum(Y * Y) - 2*Y.dot(X_std.dot(Eb[i])) + np.sum(X_l2_arr[i] * Eb2[i]))
    return result


#expected log-likelihood for a susie fit
def Eloglik(X_std_list, Y_list, s, X_l2_arr):
    result = -0.5 * s.n.dot(np.log(2*np.pi*s.sigma2))
    for i in range(len(Y_list)):
        result -= 0.5/s.sigma2[i] * get_ER2(X_std_list[i], Y_list[i], s.alpha, s.mu[i], s.mu2[i], s.Xr_list[i], X_l2_arr[i])
    return result


def get_objective(X_std_list, Y_list, s, X_l2_arr):
    return Eloglik(X_std_list, Y_list, s, X_l2_arr) - np.sum(s.KL)


def estimate_residual_variance_func(X_std_list, Y_list, s, X_l2_arr, float_dtype):
    sigma2_arr = np.zeros(len(Y_list), dtype=float_dtype)
    for i in range(len(Y_list)):
        sigma2_arr[i] =  get_ER2(X_std_list[i], Y_list[i], s.alpha, s.mu[i], s.mu2[i], s.Xr_list[i], X_l2_arr[i]) / s.n[i]
    return sigma2_arr
    
        

def loglik(V, prior_weights, compute_lbf_params):
    lbf = compute_lbf(V, *compute_lbf_params)
    maxlbf = np.max(lbf)
    w = np.exp(lbf - maxlbf)
    w_weighted = w * prior_weights
    weighted_sum_w = w_weighted.sum()
    loglik = maxlbf + np.log(weighted_sum_w)
    return loglik

        

def optimize_prior_variance(optimize_V, prior_weights, compute_lbf_params=None, alpha=None, post_mean2=None, w_pop=None, check_null_threshold=0, float_dtype = np.float64):
    if optimize_V == 'optim':
        neg_loglik_logscale = lambda lV: -loglik(np.exp(lV), prior_weights, compute_lbf_params)
        opt_obj = minimize_scalar(neg_loglik_logscale, bounds=(-30,15))
        lV = opt_obj.x
        V = np.exp(lV)
    elif optimize_V == 'EM':
        V_arr = np.array([np.sum(alpha * post_mean2[i]) for i in range(post_mean2.shape[0])], dtype=float_dtype)
        V = (w_pop.dot(V_arr)).astype(float_dtype)
    else:
        raise ValueError('unknown optimization method')
        
    # set V exactly 0 if that beats the numerical value by check_null_threshold in loglik.
    #if check_null_threshold>0:
    if loglik(0, prior_weights, compute_lbf_params) + check_null_threshold >= loglik(V, prior_weights, compute_lbf_params):
        V=0
    return V
    
    
    
def compute_lbf_1pop(V, X_std, Y, X_l2,
        residual_variance,
        return_moments=False,
        verbose=False,
        float_dtype = np.float64
        ):
    
    Xty = Y.dot(X_std)
    betahat = np.zeros(Xty.shape[0], dtype=float_dtype)
    shat2 = np.zeros(Xty.shape[0], dtype=float_dtype)
    lbf = np.zeros(Xty.shape[0], dtype=float_dtype)
    nz = (X_l2!=0)
    betahat[nz] = Xty[nz] / X_l2[nz]
    shat2[nz] = residual_variance / X_l2[nz]
    lbf[nz] = stats.norm(0, np.sqrt(V+shat2[nz])).logpdf(betahat[nz]) - stats.norm(0, np.sqrt(shat2[nz])).logpdf(betahat[nz]) #equation A.3
    if not return_moments: return lbf
    
    post_var = np.zeros((1, Xty.shape[0]), dtype=float_dtype)
    if not np.isclose(V, 0):
        post_var[0,nz] = 1.0 / (1.0/V + 1.0/shat2[nz]) # posterior variance
    post_mean = (1.0/residual_variance) * post_var * Xty
    post_mean2 = post_var + post_mean**2 # second moment

    return lbf, post_mean, post_mean2
    
    
    
def compute_lbf(V, Y_list, X_std_list, X_l2_arr,
        rho, inv_rho, logdet_rho,
        residual_variance,
        return_moments=False,
        verbose=False,
        float_dtype=np.float64
        ):

    #if there is only one population, use the fast computation or the original SuSiE code
    num_pops = len(Y_list)
    if num_pops==1:
        return compute_lbf_1pop(V, X_std_list[0], Y_list[0], X_l2_arr[0], residual_variance[0], return_moments, verbose, float_dtype)
    
    
    num_variables = X_std_list[0].shape[1]
    #ll = np.zeros(num_variables)
    lbf = np.zeros(num_variables, dtype=float_dtype)
    
    if return_moments:
        post_mean = np.zeros((num_pops, num_variables), dtype=float_dtype)
        post_mean2 = np.zeros((num_pops, num_variables), dtype=float_dtype)
        
    if not np.isclose(V,0):

        #compute YT_invD_Z = Y.T * inv(D) * Z - i.e., the inner product of Y on all X in each population, scaled by 1/sigma2 for that population
        YT_invD_Z = np.array([Y.dot(X)/sigma2 for Y,X,sigma2 in zip(Y_list, X_std_list, residual_variance)], dtype=float_dtype)
    
        #compute A (the effects covariance matrix) and its inverse and log-determinant
        A = rho*V
        inv_A = inv_rho / V
        logdetA = logdet_rho + num_pops*np.log(V)
    
        for i in range(num_variables):
        
            #compute the diagonal of Q = Z.T * inv(D) * Z (this is a diagonal matrix)
            Q_diag = X_l2_arr[:,i] / residual_variance

            #compute log-determinent for inv(A)+Q
            Ainv_plus_Q = inv_A + np.diag(Q_diag)
            logdet_Ainv_plus_Q_sign, logdet_Ainv_plus_Q = np.linalg.slogdet(Ainv_plus_Q)
            assert logdet_Ainv_plus_Q_sign>0
        
            #compute inv_Ainv_plus_Q_times_ZT_invD_Y
            inv_Ainv_plus_Q_times_ZT_invD_Y = np.linalg.solve(Ainv_plus_Q, YT_invD_Z[:,i])

            #compute log-BF for this variable
            lbf_1 = 0.5 * YT_invD_Z[:,i].dot(inv_Ainv_plus_Q_times_ZT_invD_Y)
            lbf_2 = -0.5 * (logdetA + logdet_Ainv_plus_Q)
            lbf[i] = lbf_1 + lbf_2
        
            #compute posterior moments for this variable
            if return_moments:
                AQ = A*Q_diag
                post_mean[:,i] = A.dot(YT_invD_Z[:,i]) - AQ.dot(inv_Ainv_plus_Q_times_ZT_invD_Y)
                post_covar_i = A - AQ.dot(A) + AQ.dot(np.linalg.solve(Ainv_plus_Q, AQ.T)) # AQ is symmetric...
                post_mean2[:,i] = post_mean[:,i]**2 + np.diag(post_covar_i)
        
    if return_moments:
        return lbf, post_mean, post_mean2
    else:
        return lbf
        
        
        
def single_effect_regression(Y_list, X_std_list, V, X_l2_arr, w_pop,
        rho, inv_rho, logdet_rho,
        residual_variance, 
        prior_weights=None,
        optimize_V=None,
        check_null_threshold=0,
        verbose=False,
        float_dtype = np.float64
        ):

    
    #optimize V if needed (V is sigma_0^2 in the paper)
    compute_lbf_params = (Y_list, X_std_list, X_l2_arr, rho, inv_rho, logdet_rho, residual_variance, False, verbose, float_dtype)
    if optimize_V not in ['EM', None]:
        V = optimize_prior_variance(optimize_V, prior_weights, compute_lbf_params=compute_lbf_params, alpha=None, post_mean2=None, w_pop=w_pop, check_null_threshold=check_null_threshold, float_dtype = float_dtype)
        
    #compute lbf (log Bayes-factors)
    lbf, post_mean, post_mean2 = compute_lbf(V, Y_list, X_std_list, X_l2_arr, rho, inv_rho, logdet_rho, residual_variance, return_moments=True, verbose=verbose)
    
    #compute alpha as defined in Appendix A.2
    maxlbf = np.max(lbf)
    w = np.exp(lbf - maxlbf)
    w_weighted = w * prior_weights
    weighted_sum_w = w_weighted.sum()
    alpha = w_weighted / weighted_sum_w
    
    #compute log-likelihood (equation A.5)
    lbf_model = maxlbf + np.log(weighted_sum_w)
    loglik = lbf_model + np.sum([np.sum(stats.norm(0, np.sqrt(sigma2)).logpdf(Y)) for sigma2,Y in zip(residual_variance, Y_list)])
        
    if optimize_V == 'EM':
        V = optimize_prior_variance(optimize_V, prior_weights, compute_lbf_params=compute_lbf_params, alpha=alpha, post_mean2=post_mean2, w_pop=w_pop, check_null_threshold=check_null_threshold, float_dtype=float_dtype)
        
    res = SER_RESULTS(alpha=alpha, mu=post_mean, mu2=post_mean2, lbf=lbf, lbf_model=lbf_model, V=V, loglik=loglik)
    return res
    
    

def update_each_effect(X_std_list, Y_list, s, X_l2_arr, w_pop,
        rho, inv_rho, logdet_rho,
        estimate_prior_variance=False, 
        estimate_prior_method='optim',
        check_null_threshold=0.0,
        verbose=False,
        float_dtype = np.float64
        ):

        
    if not estimate_prior_variance:
        estimate_prior_method = None
    L = s.alpha.shape[0]
    
    tqdm_L = tqdm(range(L), disable=not verbose, file=sys.stdout)
    for l in tqdm_L:
        tqdm_L.set_description('csnp %d/%d'%(l+1, L))
        R_list = []
        for k in range(len(X_std_list)):
            s.Xr_list[k] -= X_std_list[k].dot(s.alpha[l] * s.mu[k,l])
            R_list.append(Y_list[k] - s.Xr_list[k])
        res = single_effect_regression(R_list, X_std_list, s.V[l], X_l2_arr, w_pop,
              rho, inv_rho, logdet_rho,
              residual_variance=s.sigma2, prior_weights=s.pi,
              optimize_V=estimate_prior_method,
              check_null_threshold=check_null_threshold,
              verbose=verbose,
              float_dtype = float_dtype
              )
              
        # Update the variational estimate of the posterior mean.
        s.mu[:,l,:] = res.mu
        s.alpha[l,:] = res.alpha
        s.mu2[:,l,:] = res.mu2
        s.V[l] = res.V
        s.lbf[l] = res.lbf_model
        s.KL[l] = -res.loglik + SER_posterior_e_loglik(X_std_list, R_list, s.sigma2, res.mu * res.alpha, res.mu2 * res.alpha, X_l2_arr, s.n)
        for k in range(len(X_std_list)):
            s.Xr_list[k] += X_std_list[k].dot(s.alpha[l] * s.mu[k,l])

        
    return s

        
def susie_get_pip(s, prior_tol=1e-9):
    if s.has_null_index: s.alpha = s.alpha[:, :-1]
    include_idx = s.V > prior_tol
    if not np.any(include_idx):
        return np.zeros(s.alpha.shape[1])
    res = s.alpha[include_idx, :]
    pips = 1 - np.prod(1-res, axis=0)
    return pips
        


def susie_multi(X_list, Y_list, rho, L,
         scaled_prior_variance=0.00001,
         residual_variance=None, 
         prior_weights=None,
         null_weight = None,
         standardize=True,
         intercept=True,
         estimate_residual_variance=True,
         estimate_prior_variance=True,
         estimate_prior_method='EM',
         prior_tol=1e-9,
         residual_variance_upperbound=np.inf,
         residual_variance_lowerbound = 0,
         max_iter=100,
         tol=1e-3,
         check_null_threshold = 0,
         verbose=False,
         overwrite_X=False,
         float_dtype = np.float64
         ):

    t0 = time.time()
    #check input
    assert len(X_list) == len(Y_list)
    assert np.all([X.shape[1] == X_list[0].shape[1] for X in X_list])
    assert np.all([X.shape[0] == Y.shape[0] for X,Y in zip(X_list, Y_list)])
    if prior_weights is not None:
        prior_weights = prior_weights.astype(float_dtype)
    
    if null_weight is not None:
        assert 0<=null_weight<1
        if null_weight==0:
            null_weight = None
        else:
            if prior_weights is None:
                prior_weights = (1-null_weight)/X_list[0].shape[1] + np.zeros(X_list[0].shape[1]+1, dtype=float_dtype)
                prior_weights[-1] = null_weight
            else:
                prior_weights = prior_weights * (1-null_weight)
                prior_weights = np.concatenate((prior_weights, [null_weight]))
            X_list = [np.concatenate((X, np.zeros((X.shape[0], 1), dtype=float_dtype)), axis=1) for X in X_list]
    assert not np.any([np.any(np.isnan(X)) for X in X_list])
    
    #remove missing individuals
    for i in range(len(X_list)):
        if np.any(np.isnan(Y_list[i])):
            X_list[i] = X_list[i][~np.isnan(Y_list[i])]
            Y_list[i] = Y_list[i][~np.isnan(Y_list[i])]
        
    #center and scale Y
    if intercept:
        mean_y_list = [Y.mean() for Y in Y_list]
        Y_list = [Y-mean_Y for Y,mean_Y in zip(Y_list, mean_y_list)]
        
    #compute w_pop (the relative size of each population)
    n_arr = np.array([X.shape[0] for X in X_list], dtype=int)
    w_pop = (n_arr / n_arr.sum()).astype(float_dtype)

    #compute X mean and std
    if intercept:
        cm_arr = np.array([X.mean(axis=0) for X in X_list], dtype=float_dtype)
    else:
        cm_arr = np.zeros((len(X_list), X_list[0].shape[1]), dtype=float_dtype)
    if standardize:
        csd_arr = np.array([X.std(axis=0, ddof=1) for X in X_list], dtype=float_dtype)
        csd = np.sum(csd_arr * w_pop[:, np.newaxis], axis=0)
    else:
        csd = np.ones(X_list[0].shape[1], dtype=float_dtype)

        
    #Standardize X
    is_constant_column = np.isclose(csd, 0.0)
    csd[is_constant_column] = 1.0
    if overwrite_X:
        X_std_list = X_list
        for pop_i in range(len(X_std_list)):
            X_std_list[pop_i] -= cm_arr[pop_i]
            X_std_list[pop_i] /= csd
    else:
        X_std_list = [(X-cm)/csd for X,cm in zip(X_list, cm_arr)]
    
    #explicitly mark that constant columns are zero
    if np.any(is_constant_column):
        for i in range(len(X_std_list)):
            X_std_list[i][:, is_constant_column] = 0.0
        
    
    #create a C-contiguous version of X
    assert np.all([X.flags['F_CONTIGUOUS'] == X_std_list[0].flags['F_CONTIGUOUS'] for X in X_std_list])
    if X_std_list[0].flags['F_CONTIGUOUS']:
        for i in range(len(X_std_list)):
            X_std_list[i] = np.ascontiguousarray(X_std_list[i])
        
    
    X_l2_arr = np.array([np.einsum('ij,ij->j', X_std, X_std) for X_std in X_std_list], dtype=float_dtype)
    
    
    #compute rho properties
    rho = rho.astype(float_dtype)
    inv_rho = la.inv(rho).astype(float_dtype)
    logdet_rho_sign, logdet_rho = np.linalg.slogdet(rho)
    assert logdet_rho_sign>0
    
    #init setup
    p = X_list[0].shape[1]
    varY = np.concatenate(Y_list).var(ddof=1)
    varY_list = [Y.var(ddof=1) for Y in Y_list]
    if np.isscalar(scaled_prior_variance):
        assert 0 < scaled_prior_variance <= 1
    if residual_variance is None:
        residual_variance = np.array(varY_list, dtype=float_dtype)
    if prior_weights is None:
        prior_weights = np.zeros(p, dtype=float_dtype) + 1.0/p
    else:
        prior_weights = (prior_weights / np.sum(prior_weights)).astype(float_dtype)
    assert prior_weights.shape[0] == p
    if p<L: L=p
    s = S(X_std_list, L, scaled_prior_variance, residual_variance, varY, prior_weights, null_weight, float_dtype = float_dtype)
    elbo = np.zeros(max_iter+1) + np.nan
    elbo[0] = -np.inf
    
    
    ### start iterations ###
    tqdm_iter = tqdm(list(range(max_iter)), disable=not verbose, file=sys.stdout)
    for i in tqdm_iter:
        tqdm_iter.set_description('iteration %d/%d'%(i+1, max_iter))
        s = update_each_effect(X_std_list, Y_list, s, X_l2_arr, w_pop,
        rho, inv_rho, logdet_rho,
        estimate_prior_variance, estimate_prior_method,
        verbose=verbose,
        float_dtype = float_dtype,
        check_null_threshold = check_null_threshold
        )

        #compute objective before updating residual variance
        #because part of the objective s.kl has already been computed
        #under the residual variance before the update
        elbo[i+1] = get_objective(X_std_list, Y_list, s, X_l2_arr)
        if verbose:
            logging.info('objective: %s'%(elbo[i+1]))
            print('objective: %s'%(elbo[i+1]))
        
        if (elbo[i+1] - elbo[i]) < tol:
            s.converged = True
            tqdm_iter.close()
            break
        
        if estimate_residual_variance:
            s.sigma2 = estimate_residual_variance_func(X_std_list, Y_list, s, X_l2_arr, float_dtype)
            s.sigma2 = np.minimum(s.sigma2, residual_variance_upperbound).astype(float_dtype)
            s.sigma2 = np.maximum(s.sigma2, residual_variance_lowerbound).astype(float_dtype)
        if verbose:
            logging.info('objective after updating sigma2: %s'%(get_objective(X_std_list, Y_list, s, X_l2_arr)))
    
    elbo = elbo[1:i+2] # Remove first (infinite) entry, and trailing NAs.
    s.elbo = elbo
    s.niter = i+1

    if verbose: 
        logging.info('done in %0.2f seconds'%(time.time() - t0))
        logging.info(f'elbo: {elbo[~np.isnan(elbo)]}')
    if not s.converged:
        logging.info('IBSS algorithm did not converge in %d iterations'%(max_iter))
    else:
        logging.info('IBSS algorithm converged in %d iterations'%(i))
        
    #zero out everything related to constant variables, just to be on the safe side
    if np.any(is_constant_column):
        s.mu[:, :, is_constant_column] = 0.0
        s.mu2[:, :, is_constant_column] = 0.0
        s.alpha[:, is_constant_column] = 0.0
        
    if intercept:
        s.intercept = np.array(mean_y_list)
        s.fitted = []
        for k in range(len(X_list)):
            s.intercept[k] -= cm_arr[k].dot(np.sum(s.alpha*s.mu[k] / csd, axis=0))
            s.fitted.append(s.Xr_list[k] + mean_y_list[k])
    else:
        s.intercept = np.zeros(len(X_list))
        s.fitted = s.Xr_list
        
    s.pip = susie_get_pip(s, prior_tol=prior_tol)    
    s.X_column_scale_factors = csd.copy()
    s.X_column_scale_factors[is_constant_column] = 0.0
    s.coef = np.array([np.sum(s.mu[k] * s.alpha, axis=0) / csd for k in range(len(Y_list))])
    s.coef_sd = np.array([(np.sqrt(np.sum(s.alpha * s.mu2[k] - (s.alpha*s.mu[k])**2, axis=0)) / csd) for k in range(len(Y_list))])
    
    return s
        
        
        
#def susie(X, Y, **args):
#         
#     susie_obj = susie_multi(X_list=[X], Y_list=[Y], rho=np.ones((1,1)), **args)
#     susie_obj.coef = susie_obj.coef[0]
#     susie_obj.coef_sd = susie_obj.coef_sd[0]
#     return susie_obj
#    

#def generate_data(n_arr, h2_arr, p, L, rho, n_train=10000):
#
#    s = np.exp(np.abs(np.random.randn(p)))  #X_scale
#    b = np.exp(np.abs(np.random.randn(p)))  #X_bias
#    c = 1.0  #Y_scale
#    
#    R = la.cholesky(rho, lower=True)
#    assert np.allclose(R.dot(R.T), rho)
#    beta = np.zeros((len(n_arr), L))
#    for beta_i in range(L):
#        beta[:, beta_i] = np.random.randn(len(n_arr))
#        beta[:, beta_i] = R.dot(beta[:, beta_i])
#    
#    X_train_list = []
#    Y_train_list = []
#    X_test_list = []
#    Y_test_list = []
#    for pop_i in range(len(n_arr)):
#        X = np.random.randn(n_arr[pop_i],p).astype(float_dtype)
#        X = X*s + b
#        g = X[:,:L].dot(beta[pop_i])
#        g_std = g * np.sqrt(h2_arr[pop_i]) / g.std()
#        Y = g_std + np.random.randn(n_arr[pop_i])*np.sqrt(1-h2_arr[pop_i])
#        Y *= c
#        X_train_list.append(X)
#        Y_train_list.append(Y)
#        
#        X_test = np.random.randn(n_train, p).astype(float_dtype)
#        X_test = X_test*s + b
#        g_test = X_test[:,:L].dot(beta[pop_i])
#        g_test_std = g_test * np.sqrt(h2_arr[pop_i]) / g.std()
#        Y_test = g_test_std + np.random.randn(n_train)*np.sqrt(1-h2_arr[pop_i])
#        Y_test *= c
#        X_test_list.append(X_test)
#        Y_test_list.append(Y_test)
#        beta[pop_i] *= c * np.sqrt(h2_arr[pop_i]) / g.std()
#        assert np.allclose(g_std*c, X[:,:L].dot(beta[pop_i]))
        

    # # ##np.savetxt('X.txt', X, delimiter='\t')
    # # ##np.savetxt('Y.txt', Y, delimiter='\t')
    
    return beta, X_train_list, Y_train_list, X_test_list, Y_test_list
        
        
#def main():
#    configure_logger()
#
#    # X = np.loadtxt('X.txt', delimiter='\t')
#    # Y = np.loadtxt('Y.txt', delimiter='\t')
#    
#    #prior_weights = 1.0/X.shape[1] + np.zeros(X.shape[1])
#    #prior_weights[0] = 1.0
#        
#    #rho = np.array([[1,0.8], [0.8,1]])
#    #beta, X_train_list, Y_train_list, X_test_list, Y_test_list = generate_data(n_arr=[50000, 10000], h2_arr=[0.005,0.01], p=10, L=2, rho=rho)
#
#    rho = np.ones((1,1))
#    beta, X_train_list, Y_train_list, X_test_list, Y_test_list = generate_data(n_arr=[10000], h2_arr=[0.01], p=10, L=2, rho=rho)
#
#    
#    
#    susie_obj = susie_multi(X_train_list, Y_train_list, rho, L=10, estimate_prior_variance=True, verbose=True, prior_weights=None, estimate_prior_method='EM')
#    #susie_obj_em = susie_multi(X_train_list, Y_train_list, rho, L=10, estimate_prior_variance=True, verbose=True, prior_weights=None, estimate_prior_method='EM')
#
#    Y_hat = np.array([susie_obj.intercept[k] + X_test_list[k].dot(susie_obj.coef[k]) for k in range(len(X_test_list))])
#    import ipdb; ipdb.set_trace()    
#        
#        
#        
#if __name__ == '__main__':
#    main()

    
    
    
        
        
        
        
        
