import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.optimize
from estimate_brownian_model import estimate_brownian_model, sample_brownian_model_params

#np.random.seed(2)
# True values we're trying to estimate from simulated data
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


def simulate_fit():
    # Simulate the "saccade phase errors" for n trials
    n = 30
    data = [np.array(list(simulate_saccade_phase_errors())) for i in range(n)]
    data = [d for d in data if len(d)]

    # Estimate model parameters
    return estimate_brownian_model(data)

n = 100
data = [np.array(list(simulate_saccade_phase_errors())) for i in range(n)]
data = [d for d in data if len(d)]

"""
for d in data:
    plt.plot(*d.T, '.-', color='black', alpha=0.1)
plt.show()
"""

opt = estimate_brownian_model(data)
#print(opt)
samples = sample_brownian_model_params(data)
#fits = [simulate_fit() for i in range(100)]
names = "m0 mt v0 vt vn".split()
samples = np.rec.fromrecords(samples, names=names)

N = len(names)
fig, plots = plt.subplots(N, N, sharex='col')

opts = dict(zip(names, opt))
for row, yvar in zip(plots, names):
    for ax, xvar in zip(row, names):
        # WHAT THE HACK :D
        xtrue = locals()[xvar+"_true"]
        ytrue = locals()[yvar+"_true"]
        if xvar == yvar:
            ax.hist(samples[xvar], density=True, bins='scott')
            ax.axvline(xtrue, color='black')
            #ax.axvline(opts[xvar], color='red')
        else:
            ax.plot(samples[xvar], samples[yvar], '.')
            ax.axvline(xtrue, color='black')
            ax.axhline(ytrue, color='black')
        if ax.is_first_col():
            ax.set_ylabel(yvar)
        if ax.is_last_row():
            ax.set_xlabel(xvar)
plt.show()

#plt.figure("m0")
#plt.hist(fits['m0'])
#plt.axvline(fits[
