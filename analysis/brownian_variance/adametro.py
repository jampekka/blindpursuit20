import numpy as np
import numba

jit = numba.njit

def metropolis_iter(f, init, cov=None):
    if cov is None:
        cov = np.eye(len(init))
    
    x = np.array(init)
    zero = np.zeros_like(x)
    l = f(x)
    while True:
        prop = x + np.random.multivariate_normal(zero, cov)
        prop_l = f(prop)
        ratio = np.exp(prop_l - l)
        accept = ratio > 1 or np.random.rand() < ratio
        if accept:
            x = prop
            l = prop_l
        yield x, l, accept

@jit
def multivariate_normal_random(m, cov):
    hack = (cov + cov.T)/2
    A = np.linalg.cholesky(hack + np.eye(len(m))*1e-9)

    return m + A@np.random.randn(len(m))

@jit
def adaptive_metropolis_iter(f, init, cov, slack=1e-9, adaptation_start=10):
    x = init
    zero = np.zeros_like(x)
    l = f(x)
    i = 0
    n_accepted = 0
    est_cov = cov.copy()
    est_mean = x.copy()
    mm = est_mean.reshape(-1, 1)@est_mean.reshape(1, -1)
    sd = (2.4)**2/len(x)
    slacker = np.eye(len(x))*slack
    while True:
        prop = x + multivariate_normal_random(zero, (cov + slacker)*sd)
        prop_l = f(prop)
        ratio = np.exp(prop_l - l)
        accept = ratio > 1 or np.random.rand() < ratio
        if accept:
            x = prop
            l = prop_l
            n_accepted += 1
        
        i += 1
        
        delta0 = (x - est_mean)
        est_mean += delta0/i
        delta1 = (x - est_mean)

        D = delta0.reshape(-1, 1)@(delta1/i).reshape(1, -1)
        est_cov *= (i - 1)/i
        est_cov += D

        if n_accepted >= adaptation_start:
            cov[:] = est_cov
        
        meta = {
            'loglik': l,
            'sampleno': i,
            'accepted': accept,
            'cov': cov,
            'proposal': prop,
            'proposal_loglik': prop_l
                }
        yield x, meta

@jit
def adaptive_metropolis(f, init, cov, n_samples=2000, slack=1e-9, adaptation_start=3):
    sampler = adaptive_metropolis_iter(f, init, cov, slack=slack, adaptation_start=adaptation_start)
    output = np.zeros((n_samples, len(init)))
    for i, (sample, meta) in zip(range(n_samples), sampler):
        output[i] = sample
    return output
