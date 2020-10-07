# a script for merging gaze and target data, fixing irregularities (in timestamps), 
# and separating conditions and trials

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import scipy.ndimage


target = pd.read_csv('data/position_data.csv')

gaze = pd.read_csv('data/gaze_data.csv')

gaze.rename(columns={'pupilTime': 'ts'}, inplace=True)

def find_conditions(sd):
    ts = sd.ts.copy()
    
    # first, detect "chunks", separated by time gaps
    deltas = ts.diff()
    is_gap = deltas > 1
    block = is_gap.cumsum()
    
    sd['target_x'] = sd['x'].copy()
    sd['target_y'] = sd['y'].copy()
    
    tx = sd.groupby(block).target_x.bfill()
    
    cond_gap = is_gap & ((round(tx,2) == 324.00) | (round(tx,2) == 960.00))   
 
    sd['condition'] = cond_gap.cumsum()
    
    groups = sd.groupby('condition')

    xs = groups['target_x'].max().round()
    ys = groups['target_y'].max().round()

    is_constant = list(xs.index[(ys == 540) & (xs != 960)])
    is_circular = list(xs.index[(1280 < xs) & (xs < 1290)])
    is_ballistic = list(xs.index[(1600 < xs) & (xs < 1770)])
    
    cond = dict({'constant': is_constant, 'circular': is_circular, 'ballistic': is_ballistic})
    m2 = {v: k for k,vv in cond.items() for v in vv}
    sd['condition'] = sd.condition.map(m2).copy()
    
    # #and check
    # groups = sd.groupby('condition')
    # fig, ax = plt.subplots()
    # for name, group in groups:
    #     ax.plot(group.ts, group.target_x, label=name)
    # ax.legend()
    # plt.show()
    
    return sd


# find trials 
def find_trials(sd):
    
    sd['target_x'] = sd['x'].copy()
    sd['target_y'] = sd['y'].copy()


    #separate into trials 
    
    def find_last_point(boolseries):
        boolseries_last = (((boolseries != boolseries.shift(1)) & (~boolseries))).shift(-1)
        boolseries_last.iloc[-1] = False
        return boolseries_last
    
    # mark trial ends: end of target presentation
    is_trial_bound = (sd.condition.notnull()) & (sd.is_target == True)
    is_target_last = find_last_point(is_trial_bound)

    sd['trial_number'] = is_target_last.cumsum()
    
    is_cue_visible = (sd.is_visible==1) & (sd.is_cue==1)
    sd['trial_first'] = ((is_cue_visible != is_cue_visible.shift(1)) & (is_cue_visible)) 
    
    
    #works except for ballistic; need to define still points
    is_ballistic_still = (sd.condition == 'ballistic') & (round(sd.y) == 1026)
    ballistic_first = find_last_point(is_ballistic_still)
    #shift a bit to center the starting point (similar to constant condition)
    ballistic_first = ballistic_first.shift(-50)
    
    sd.trial_first[sd.condition == 'ballistic'] = ballistic_first[sd.condition == 'ballistic'].copy()
    sd['trial_first'] = sd.trial_first.fillna(False)
    
    # each trial now ends in target. now remove "beginning" of trial ( = tail of last trial)
    # i.e. stuff before trial_first
    sd.trial_first[sd.trial_first != 1] = np.nan
    sd['is_trial'] = sd.groupby(['condition','trial_number']).trial_first.ffill()
    sd['is_trial'] = sd.is_trial.fillna(False)
    
    #plt.scatter(sd.ts[sd.is_trial], sd.target_x[sd.is_trial])
    #choose only conditions
    sd = sd[sd.condition.isin(['constant','circular','ballistic'])].copy()    
    
    sd.trial_number[sd.is_trial != 1] = np.nan
    
    #remove "leaking" trials
    l = sd.groupby(['condition','trial_number']).is_target.last()
    dur = sd.groupby(['condition', 'trial_number']).ts.agg(np.ptp)
    
    remove = l[l!=1].reset_index().loc[:,'condition':'trial_number'].copy()
    remove_trials = pd.concat([remove, dur[((dur > 7) | (dur < .5))].reset_index().loc[:,'condition':'trial_number']]).copy()
    
    for cond, trn in remove_trials.values:
        sd.is_trial[(sd.condition == cond) & (sd.trial_number == trn)] = False
        
    sd.trial_number[sd.is_trial != 1] = np.nan
    
    sd.groupby('condition')['trial_number'].nunique()
    
    # fig, ax = plt.subplots()
    # for name, group in sd.groupby('trial_number'):
    #     ax.scatter(group.ts, group.target_x, label=name, alpha=.8)
    #     ax.scatter(group.ts[group.is_trial==1], group.target_x[group.is_trial==1], color="black", alpha=.2)
    # ax.legend()
    # fig.show()


    sd['full_trial'] = sd.trial_number.ffill()
    sd['success_trial'] = sd.groupby('full_trial').is_success.transform(lambda x: x.eq(1).any())


    #select columns
    df = sd.loc[:,'participant':'is_target'].join(sd[['condition','trial_number','full_trial','success_trial','is_trial']]).copy()
    
    return df



# clean and sync data
def sync_data(sd):
   
    subj = sd.name
    
    # Get the participant's target data'
    sdt = target_trials.query("participant == @subj")
    
    diff = sdt.ts.diff()    
    sdt['is_gap'] = diff > 1
    sdt.loc[sdt.is_gap,'x':'success_trial'] = np.nan # mark first point after gap as nan, to avoid spilling
    
    def resample(field, **kwargs):
        return interp1d(sdt['ts'].values, sdt[field].values, axis=0, bounds_error=False, **kwargs)(sd['ts'].values)
    # Resample position using the default (linear) interpolation
    sd['target_x'] = resample('x')
    sd['target_y'] = resample('y')


    flagfields = 'is_visible', 'is_cue', 'is_target', 'success_trial', 'is_gap', 'condition', 'trial_number'
    match = sdt.ts.values.searchsorted(sd.ts.values) - 1
    for f in flagfields:
        sd[f] = sdt[f].values[match]
        
    sd = sd[sd.target_x.notnull()].copy()

    degs_per_pix = 80/1920
    sd['target_x'] -= 1920/2; sd['target_x'] *= degs_per_pix
    sd['gazeX'] -= 1920/2; sd['gazeX'] *= degs_per_pix
    sd['target_y'] -= 1080/2; sd['target_y'] *= degs_per_pix
    sd['gazeY'] -= 1080/2; sd['gazeY'] *= degs_per_pix
   
    # Save the original just for comparison
    sd['ogazeX'] = sd.gazeX
    sd['ogazeY'] = sd.gazeY
   
    # Get low confidence samples
    badones = (sd.conf < 0.8).values
    badones = scipy.ndimage.morphology.binary_dilation(badones, iterations=5)
   
    sd = sd[~badones].copy()
    
    #now clean target data
    sd['xspeed'] = sd.target_x.diff()/sd.ts.diff() 

    badonest = (np.abs(sd.xspeed) > 20000).values
    badonest = scipy.ndimage.morphology.binary_dilation(badonest, iterations=5)

    # plt.scatter(sd.ts, sd.target_x, alpha=.3)
    # plt.scatter(sd.ts[badonest], sd.target_x[badonest], alpha=.8)    
    # plt.title('bad segments of target data')
    # plt.show()
    
    df = sd[~badonest].copy() 
    
    return df




### first find conditions and trials, then sync
target_cond = target.groupby('participant').apply(find_conditions)
target_trials = target_cond.groupby('participant').apply(find_trials)

#target_trials.to_csv('data/trial_data.csv')

synced = gaze.groupby('participant').apply(sync_data)

#synced.to_pickle('data/trial_gaze_data.pickle')
