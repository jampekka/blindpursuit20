import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import interp1d
import sys

data = pd.read_csv('../../data/saccades_hidden.csv')

data['gaze_error'] = np.linalg.norm(data[['target_x', 'target_y']].values - data[['gazeX_s', 'gazeY_s']].values, axis=1)
data['target_r'] = np.linalg.norm(data[['target_x', 'target_y']], axis=1)
data['gaze_r'] = np.linalg.norm(data[['gazeX_s', 'gazeY_s']], axis=1)
data['radius_error'] = r_e = data['gaze_r'] - data['target_r']

data = data[data.timept.isin(('launch',))]
#data = data[~data.first_saccade]

def localregr(ts, xs, scale=1.0):
    def reg(t):
        dt = (ts - t)/scale
        ws = np.exp(-dt**2)
        ws /= np.sum(ws)
        fit = Polynomial.fit(ts, xs, deg=1, w=ws)
        return fit(t), ws
    ms = np.array([reg(t)[0] for t in ts])
    ss = (xs - ms)**2
    
    @np.vectorize
    def reg_and_std(t):
        x, ws = reg(t)
        s = np.sqrt(np.dot(ss, ws))
        return x, s
    return reg_and_std

ts = np.linspace(0, 3, 500)

fig, (rax, pax) = plt.subplots(ncols=2, constrained_layout=True)
pax.yaxis.tick_right()
pax.yaxis.set_label_position("right")

alls = []
dump = []
for part, sd in data.groupby('participant'):
    vals = []
    for tr, td in sd.groupby('trial_number'):
        if len(td) < 2: continue
        xs = td.occl_ts.values
        
        # Rewrap as some seem to start at |angles| > pi:
        ps = td.phase_diff_unwrap.values
        ps = (ps + np.pi) % (2 * np.pi) - np.pi
        ps = np.unwrap(ps)
    
        rs = td.radius_error.values
        #ys = td.radius_error.values
        
        ys = np.c_[rs, ps]

        vals.append(interp1d(xs, ys, axis=0, bounds_error=False)(ts))
    
    vals = np.array(vals)
    m = np.nanmean(vals, axis=0)
    s = np.nanstd(vals, axis=0)
    n = np.sum(np.isfinite(vals[:,:,0]), axis=0)
    #ys = data.phase_diff_unwrap.values
    ys = data.radius_error.values
    #ms, ss = localregr(xs, ys, scale=0.2)(ts)
    valid = n > 5
    vs = s.copy()
    vs[~valid] = np.nan
    alls.append(vs)

    d = pd.DataFrame({'participant': part, 'ts': ts, 'n_valid': n, 'radius_std': s[:,0], 'phase_std': s[:,1], 'participant': part})
    
    dump.append(d)
    rax.plot(ts[valid], s[valid,0], color=f"C{part}")
    rax.set_xlabel("Time hidden (seconds)")
    rax.set_ylabel("Radius error standard deviation (degrees)")
    
   
    pax.plot(ts[valid], s[valid,1]/(2*np.pi), color=f"C{part}")
    pax.set_ylabel("Phase error standard deviation (turns)")
    pax.set_xlabel("Time hidden (seconds)")

alls = np.array(alls)
gm = np.nanmedian(alls, axis=0)
rax.plot(ts, gm[:,0], color='black', lw=3)
pax.plot(ts, gm[:,1]/(2*np.pi), color='black', lw=3)
#plt.plot(xs, ys, 'k.')
#plt.plot(ts, ms)
#plt.plot(ts, ms + 2*ss)

dump = pd.concat(dump)
dump.to_csv(sys.stdout, index=False)

plt.show()
