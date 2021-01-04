import numpy as np
import pandas as pd
import scipy.special
import scipy.optimize
import matplotlib.pyplot as plt

def get_detection_data():
    data = pd.read_parquet('../preprocessing/gaze_and_target.parquet')

    detections = []
    for (p, t), td in data.groupby(['participant_id', 'trial_number']):
        failure = False
        success = False
        
        #target_idxs = np.flatnonzero((td.signtype.values == 'target') & (td.is_visible))
        target_idxs = np.flatnonzero((td.signtype.values == 'mask') & (td.is_visible))
        #target_idxs2 = np.flatnonzero((td.signtype.values == 'target') & (td.is_visible))
        
        if len(target_idxs) == 0:
            continue

        # Using just the first moment of target occurence, could use
        # average of all frames with the target present, but this is such
        # a brief moment, it probably makes no difference
        target_i = target_idxs[0]

        signs = td.signtype.unique()
        if 'success' in signs:
            success = True
        if 'failure' in signs:
            failure = True

        assert not (success and failure)
        if not (success or failure): continue
        
        row = td.iloc[target_i].copy()
        row['success'] = success
        detections.append(row)
        
    dets = pd.DataFrame.from_records(detections)
    screen_extent = np.array([1920, 1080]).reshape(1, -1)
    dets['target_radius'] = np.linalg.norm(dets[['target_x', 'target_y']].values - screen_extent/2, axis=1)
    dets['target_gaze_distance'] = np.linalg.norm(dets[['target_x', 'target_y']].values - dets[['gaze_screen_x', 'gaze_screen_y']].values, axis=1)

    return dets

def mafc_logit(m):
    def func(x):
        c = (x*m - 1)/(m - 1)
        return scipy.special.logit(c)
    return func

class LogisticDetection:
    def __init__(self, slope, intercept, p0=1/4):
        self.slope = slope
        self.intercept = intercept
        self.p0 = p0

    def __call__(self, x):
        pred = x*self.slope + self.intercept
        p = scipy.special.expit(pred)
        return p + self.p0*(1 - p)

    def logpdf(self, x, success):
        p = self(x)
        return np.log(p**success * (1-p)**(1-success))

def fit_detection(x, success):
    # Just a simple logistic regression
    def loss(param):
        return -np.sum(LogisticDetection(*param).logpdf(x, success))
    
    # Nelder-Mead is good enough for lme4, so we can blame them if
    # this gives bad results.
    fit = scipy.optimize.minimize(loss, [0.0, 0.0], method='Nelder-Mead')
    fit.model = LogisticDetection(*fit.x)

    return fit

def plot_detection_data():
    # TODO: Do this in (approx) view degrees!
    dets = get_detection_data()
    
    dets = dets.query("scenario == 'peripheralVisionTest'")
    #dets = dets.query("scenario == 'swing'")
    #dets = dets.query("confidence > 0.9")
    
    for part, dets in dets.groupby('participant_id'):
    #for dets in [dets]:
        #print(dets.target_radius.unique())
        binshares = []
        for bin, bind in dets.groupby(pd.qcut((dets.target_gaze_distance), 5)):
        #for bin, bind in dets.groupby(dets.target_radius.round()):
            #print(bin)
            binshares.append((bin.mid, bind.success.mean()))
        
        binshares = np.array(binshares)
        plt.plot(binshares[:,0], binshares[:,1], 'o-', label=part)
    plt.axhline(0.25, color='black', label="Chance")
    plt.title("Peripheral vision test")
    plt.xlabel("Target displacement from gaze")
    plt.ylabel("Success rate")
    plt.legend()
    plt.show()

def fit_detection_data():
    # TODO: Do this in (approx) view degrees!
    dets = get_detection_data()
    
    #dets = dets.query("scenario == 'peripheralVisionTest'")
    dets = dets.query("scenario == 'swing'")

    total = 0.0
    
    rng = np.linspace(0, 1000, 100)[1:]

    for part, partdets in dets.groupby("participant_id"):
        x = (partdets.target_gaze_distance.values)
        success = partdets.success.values
        fit = fit_detection(x, success)
        total -= fit.fun

        plt.plot(rng, fit.model(rng), color=f"C{part}", label=part)
        print(fit.x)
    print("Total logpdf", total)
    plt.axhline(1/4, color='black', linestyle='dashed')
    plt.legend()
    plt.ylabel("Detection probability")
    plt.xlabel("Gaze error (pixels)")
    plt.show()

if __name__ == '__main__':
    #plot_detection_data()
    fit_detection_data()
