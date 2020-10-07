import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.optimize
from estimate_brownian_model import estimate_brownian_model
from sklearn.linear_model import LinearRegression
import random
import pandas as pd

duration_max = 3

# %%
# simulate samples of 100 trials based on parameters from observed data
def simulate_fit():
    # Simulate the "saccade phase errors" for n trials
    n = 100
    data = [np.array(list(simulate_saccade_phase_errors())) for i in range(n)]
    data = [d for d in data if len(d)]

    # Estimate model parameters
    return estimate_brownian_model(data)

fits = [simulate_fit() for i in range(100)]
names = "m0 v0 mt vt vn".split()
fits = np.rec.fromrecords(fits, names=names)


N = len(names)
fig, plots = plt.subplots(N, N, sharex='col')

for row, yvar in zip(plots, names):
    for ax, xvar in zip(row, names):
        # WHAT THE HACK :D
        xtrue = locals()[xvar+"_true"]
        ytrue = locals()[yvar+"_true"]
        if xvar == yvar:
            ax.hist(fits[xvar], density=True)
            ax.axvline(xtrue, color='black')
        else:
            ax.plot(fits[xvar], fits[yvar], '.')
            ax.axvline(xtrue, color='black')
            ax.axhline(ytrue, color='black')
        if ax.is_first_col():
            ax.set_ylabel(yvar)
        if ax.is_last_row():
            ax.set_xlabel(xvar)
plt.show()



# %%

def find_variance(sh, timept):
    
    # these need to be inside the function, otherwise they will use wrong params
    def simulate_saccade_phase_errors():
        time = 0
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
    

    p = str(sh.name)
    
    # add phase at disappearance
    sh['timept_mod'] = np.nan
    sh['timept_mod'][(sh.trial_ts == 0) | (sh.timept == 'begin')] = 'begin'
    

    trials = sh[['trial_number','trial_ts','phase_error']][(sh.timept_mod == timept)]
    tdata = []
    for i in trials.trial_number.unique():
        td = trials[(trials.trial_number == i) & (trials.phase_error > -3)].iloc[:,1:]
        td = np.array(td)
        if len(td) > 1:
            td = np.delete(td, 1, axis=0) # removes the second row = 1st saccade
        tdata.append(td)
        
    m0_true, v0_true, mt_true, vt_true, vn_true = estimate_brownian_model(tdata) # "true" here used to denote params from observed data
    
    
    # get fixation length
    dts = trials.groupby('trial_number').trial_ts.diff()
    fixation_dur = dts.mean()
    print(fixation_dur)
    print(len(tdata))
    
    #simulate ONE sample of 100 trials (-> time series plot)
    n = 100 
    data = [np.array(list(simulate_saccade_phase_errors())) for i in range(n)]
    data = [d for d in data if len(d)]

    # # Estimate model parameters
    m0, v0, mt, vt, vn = estimate_brownian_model(data)
    
    
    f,  (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(13,9)) 
    f.suptitle('Participant ' + p)
    

    for trial in tdata:
        ts, ps = trial.T
        ax1.plot(ts, ps, '.', alpha=0.3, color='black')
    ax1.set_title('observed')
    ax1.plot([0, 3], [0, m0_true + mt_true*3])
    ax1.text(2.7, .7, 
             'm0 = ' + str(round(m0_true,3)) +
             '\nv0 = ' + str(round(v0_true,3)) +
             '\nmt = ' + str(round(mt_true,3)) +
             '\nvt = ' + str(round(vt_true,3)) +
             '\nvn = ' + str(round(vn_true,3)))
    ax1.set_ylabel('Phase error')
    

    for trial in data:
        ts, ps = trial.T
        ax2.plot(ts, ps, '.', alpha=0.3, color='black')
    ax2.set_title('simulated')
    ax2.set_ylim(-3,3)
    ax2.plot([0, 3], [0, m0 + mt*3])
    ax2.text(2.7, .7, 
             '\nm0 = ' + str(round(m0,3)) +
             '\nv0 = ' + str(round(v0,3)) +
             '\nmt = ' + str(round(mt,3)) +
             '\nvt = ' + str(round(vt,3)) +
             '\nvn = ' + str(round(vn,3)))    
    ax2.set_xlabel('Time from disappearance (s)')
    ax2.set_ylabel('Phase error')

    plt.show()
    
    return f #np.array([m0_true, v0_true, mt_true, vt_true, vn_true])

#rp_hid.to_pickle('data/hidden_saccades.pickle')
#rp_hid = pd.read_pickle('data/hidden_saccades.pickle')

figs = rp_hid.groupby('participant').apply(find_variance, 'begin')

# %%

pp = PdfPages('phase_error_estimates.pdf')
for i in range(1,11):
    pp.savefig(figs[i])
pp.close()


# %%
# bootstrapping participant parameter estimates

def bootest(sh, timept):
    
    def sample_fit():
    # take samples of 300 from observed data (n 80-90 - does it make any sense)
        n = 100
        data = random.choices(tdata, k=n)
        data = [d for d in data if len(d)]
        # Estimate model parameters
        return estimate_brownian_model(data)


    p = str(sh.name)
    
    # add phase at disappearance
    sh['timept_mod'] = np.nan
    sh['timept_mod'][(sh.trial_ts == 0) | (sh.timept == 'begin')] = 'begin'
    

    trials = sh[['trial_number','trial_ts','phase_error']][(sh.timept_mod == timept)]
    tdata = []
    for i in trials.trial_number.unique():
        td = trials[(trials.trial_number == i) & (trials.phase_error > -3)].iloc[:,1:]
        td = np.array(td)
        if len(td) > 1:
            td = np.delete(td, 1, axis=0) # removes the second row = 1st saccade
        tdata.append(td)
    
    # 100 times of samples of 100
    fits = [sample_fit() for i in range(100)]
    names = "m0 v0 mt vt vn".split()
    fits = np.rec.fromrecords(fits, names=names)
    
    N = len(names)
    
    fig, plots = plt.subplots(N, N, sharex='col', figsize = (12,10))
    fig.suptitle('Participant ' + p)
    
    for row, yvar in zip(plots, names):
        for ax, xvar in zip(row, names):
            # WHAT THE HACK :D
            if xvar == yvar:
                ax.hist(fits[xvar], density=True)
            else:
                ax.plot(fits[xvar], fits[yvar], '.')
            if ax.is_first_col():
                ax.set_ylabel(yvar)
            elif ((xvar != yvar)):
                ax.get_yaxis().set_ticks([])
            if ax.is_last_row():
                ax.set_xlabel(xvar)
    plt.show()

    
    return fig #np.array([m0_true, v0_true, mt_true, vt_true, vn_true])

#rp_hid.to_pickle('data/hidden_saccades.pickle')
#rp_hid = pd.read_pickle('data/hidden_saccades.pickle')

boot_figs = rp_hid.groupby('participant').apply(bootest, 'begin')


# %%

pp = PdfPages('participant_estimates_bootstrapped.pdf')
for i in range(1,11):
    pp.savefig(boot_figs[i])
pp.close()

# %%
#participant bootstrapped estimate distributions in one fig
def boot_distr(sh, timept):
    
    def sample_fit():
    # take samples of 300 from observed data 
        n = 100
        data = random.choices(tdata, k=n)
        data = [d for d in data if len(d)]
        # Estimate model parameters
        return estimate_brownian_model(data)
    
    # add phase at disappearance
    sh['timept_mod'] = np.nan
    sh['timept_mod'][(sh.trial_ts == 0) | (sh.timept == 'begin')] = 'begin'
    

    trials = sh[['trial_number','trial_ts','phase_error']][(sh.timept_mod == timept)]
    tdata = []
    for i in trials.trial_number.unique():
        td = trials[(trials.trial_number == i) & (trials.phase_error > -3)].iloc[:,1:]
        td = np.array(td)
        if len(td) > 1:
            td = np.delete(td, 1, axis=0) # removes the second row = 1st saccade
        tdata.append(td)
    
    # 100 times of samples of 100
    fits = [sample_fit() for i in range(100)]
    names = "m0 v0 mt vt vn".split()
    fits = np.rec.fromrecords(fits, names=names)

    return fits

distr = rp_hid.groupby('participant').apply(boot_distr, 'begin')

ddf = pd.DataFrame(distr[1])
for i in range(2,11):
    ddf = pd.concat([ddf, pd.DataFrame(distr[i])])
    
ddf['participant'] = rp_hid.participant.unique().repeat(100).copy()

hist_figs = []
for c in "m0 v0 mt vt vn".split():
    
def plothist(c):
    p = ddf.hist(column=c, by = 'participant', sharex=True)
    plt.suptitle(c)
    plt.show()
    pp.savefig()
    return p

histfigs = pd.Series("m0 v0 mt vt vn".split()).apply(plothist)

# %%

pp = PdfPages('participant_hist.pdf')
for i in range(1,6):
    pp.savefig(histfigs[i])
pp.close()

# %%

#param = saccades_hidden.groupby('participant').apply(find_variance, 'begin')

param = pd.DataFrame.from_records(param.values,index=param.index, columns = ['m0', 'v0', 'mt', 'vt', 'vn'])
param.plot(kind='bar')


N = len(param.columns)
fig, plots = plt.subplots(N, N, sharex='col')

for row, yvar in zip(plots, names):
    for ax, xvar in zip(row, names):
        # WHAT THE HACK :D
        if xvar == yvar:
            ax.hist(param[xvar], density=True)
        else:
            ax.plot(param[xvar], param[yvar], '.')
        if ax.is_first_col():
            ax.set_ylabel(yvar)
        if ax.is_last_row():
            ax.set_xlabel(xvar)
plt.show()