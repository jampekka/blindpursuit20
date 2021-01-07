import numpy as np
import matplotlib.pyplot as plt

def trial_mean_and_variance(ts, m_0, m_t, v_0, v_t, v_n):
    total = 0.0

    m = m_0
    v = v_0
    t = 0
    for i in range(ts):
        nt = ts[i]
        dt = nt - t
        t = nt
        m_pred = m + m_t*dt
        v_pred = v + v_t*dt
        z_lik = normlogpdf(z, m_pred, v_pred + v_n)
        total += z_lik
        if v_n == 0:
            K = 1.0
        elif v_pred == 0:
            K = 0.0
        else:
            K = (1/v_n)/(1/v_n + 1/v_pred)
        m = K*z + (1 - K)*m_pred
        v = K**2*v_n + (1 - K)**2*v_pred
    
    return total

# The model assumes that this is the distribution of the internal
# state + measurement noise at the given time.
def distribution_at_time(ts, m0, mt, v0, vt, vn):
    m = m0 + mt*ts
    v = v0 + vt*ts + vn
    return m, v

# Plot the model mean and +-2 sigma
def plot_brownian_model(ts, m0, mt, v0, vt, vn, **kwargs):
    m, v = distribution_at_time(ts, m0, mt, v0, vt, vn)
    s = np.sqrt(v)
    plt.plot(ts, m, **kwargs)
    plt.fill_between(ts, m - 2*s, m + 2*s,  alpha=0.5, **kwargs,)
  
# Just plot some random parameters. They should be replaced with the
# fitted ones.
def demo():
    import scipy.stats
    m0_true = 0.0
    mt_true = 0.5
    v0_true = 0.1**2
    vt_true = 1.0**2
    # If this goes too small, the fitting becomes problematic. May be a numerical issue
    vn_true = 0.1**2

    fixation_dur = 1/3
    duration_max = 3

    def simulate_saccade_phase_errors():
        time = 0.0
        value = m0_true + np.random.randn()*np.sqrt(v0_true)
        duration = np.random.rand()*duration_max
        observation = value + np.random.randn()*np.sqrt(vn_true)
        yield time, observation

        while True:
            dt = np.random.exponential(fixation_dur)
            time += dt
            if time > duration:
                break
            value += mt_true*dt + np.random.randn()*np.sqrt(vt_true*dt)
            observation = value + np.random.randn()*np.sqrt(vn_true)
            yield time, observation
    
    ts = np.linspace(0, duration_max, 1000)
    plot_brownian_model(ts, m0_true, mt_true, v0_true, vt_true, vn_true)
    n = 30
    data = [np.array(list(simulate_saccade_phase_errors())) for i in range(n)]
    data = [d for d in data if len(d)]
    

    n_samples = 0
    samples_over_sigma = 0
    for trial in data:
        t, phase = trial.T
        plt.plot(t, phase, '.', color='black', alpha=0.5)
        m, v = distribution_at_time(t, m0_true, mt_true, v0_true, vt_true, vn_true)
        n_samples += len(phase)
        samples_over_sigma += np.sum(np.abs(m - phase) > np.sqrt(v))
    
    # If the math is correct, these should be (statistically) same. With small n
    # the variation in the empirical estimate will be big. For large n (like 10000)
    # they start to get close. This is probably not the best way to test though...
    print(f"Empirical vs theoretical > 1 sigma share")
    print(samples_over_sigma/n_samples, "vs", scipy.stats.norm(0,1).cdf(-1)*2)
    plt.show()

if __name__ == '__main__':
    demo()
