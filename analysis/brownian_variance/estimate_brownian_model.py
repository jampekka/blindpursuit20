import numpy as np
import numba
import scipy.optimize

jit = numba.njit

@jit
def normlogpdf(x, m, v):
    return (-np.log(v*np.pi*2) - (m - x)**2/(v))/2

@jit
def trial_loglikelihood(trial, m_0, m_t, v_0, v_t, v_n):
    total = 0.0

    m = m_0
    v = v_0
    t = 0
    for i in range(len(trial)):
        nt = trial[i,0]
        z = trial[i,1]
#    for nt, z in trial: # for some reason not working in numba; replaced by the above
        dt = nt - t
        t = nt
        m_pred = m + m_t*dt
        v_pred = v + v_t*dt
        z_lik = normlogpdf(z, m_pred, v_pred + v_n)
        total += z_lik
        K = (1/v_n)/(1/v_n + 1/v_pred)
        m = K*z + (1 - K)*m_pred
        v = K**2*v_n + (1 - K)**2*v_pred
    
    return total


def estimate_brownian_model(trials):
    """Estimate "brownian" model parameters
        NOTE: Assumes that timestamps start from 0 (but don't have to include samples at 0)
    """
    trials = tuple(trials) # Required for numba
    @jit
    def loglikelihood(m_0, m_t, v_0, v_t, v_n):
        total = 0.0

        for trial in trials:
            total += trial_loglikelihood(trial, m_0, m_t, v_0, v_t, v_n)
        return total


    fit = scipy.optimize.minimize(
            lambda x: -loglikelihood(*x[:2], *np.exp(x[2:])),
            [0.0, 0.0, np.log(1.0), np.log(1.0), np.log(1.0)])
    m_0, m_t = fit.x[:2]
    v_0, v_t, v_n = np.exp(fit.x[2:])

    return m_0, m_t, v_0, v_t, v_n


def sample_brownian_model_params(trials):
    from adametro import adaptive_metropolis
    trials = tuple(trials) # Required for numba
    @jit
    def loglikelihood(params):
        m_0, m_t, v_0, v_t, v_n = params
        total = 0.0
        for trial in trials:
            total += trial_loglikelihood(trial, m_0, m_t, v_0, v_t, v_n)

        return total
    
    initial = [0.0, 0.0, np.log(1.0), np.log(1.0), np.log(1.0)]

    mangler = lambda x: loglikelihood(np.concatenate((x[:2], np.exp(x[2:]))))
    mangler = numba.njit(mangler)
    maxlik = scipy.optimize.minimize(lambda x: -mangler(x), np.array(initial))
    
    samples = adaptive_metropolis(mangler, maxlik.x.copy(), np.eye(len(initial))*0.001, n_samples=2000, )
    samples[:,2:] = np.exp(samples[:,2:])
    samples = samples[1000:]
    return samples


