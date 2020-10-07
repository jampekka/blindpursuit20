import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import nslr_hmm
from matplotlib.backends.backend_pdf import PdfPages

dfr = pd.read_pickle('data/trial_gaze_data.pickle') # from gaze_target_prepro.py
dfr = dfr[(dfr.trial_number.notnull()) & (dfr.condition == 'circular')]
dfr = dfr.iloc[:,1:]

def segment_hmm(df):

      # HMM    
    sample_class, segmentation, seg_class = nslr_hmm.classify_gaze(df.ts.values, df[['gazeX', 'gazeY']].values,
              structural_error=0.2, optimize_noise=False # Assume 0.2 degree noise
              )
      
    # recreate a new signal using the segmented results.
    gaze_interp = interp1d(segmentation.t, segmentation.x, axis=0, bounds_error=False)
    df['gazeX_s'], df['gazeY_s'] = gaze_interp(df.ts).T.copy()

        
    # COLORS = {
    #       nslr_hmm.FIXATION: 'grey',
    #       nslr_hmm.SACCADE: 'blue',
    #       nslr_hmm.SMOOTH_PURSUIT: 'grey',
    #       nslr_hmm.PSO: 'grey',
    #       }
    
    # f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(13,9))
    
    # hidtimes = df.ts.copy()
    # hidtimes[df.is_visible==1] = np.nan
    # ax1.plot(df.ts, df.target_x)
    # ax1.plot(hidtimes, df.target_x, label = 'hidden')
    # ax1.plot(df.ts[df.is_target==1], df.target_x[df.is_target==1], '.', markersize=3, label = 'target')
    # ax1.legend()
    # ax1.set_ylabel('target x (degrees)')
        

    # ax2.set_ylabel('x (degrees)')
    # for i, seg in enumerate(segmentation.segments):
    #       cls = seg_class[i]
    #       ax2.plot(seg.t, np.array(seg.x)[:,0], color=COLORS[cls], alpha=.6)
    
    # handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in ['grey','blue']]
    # ax2.legend(handlelist,['other','saccade'])
    
    # ax3.set_ylabel('y (degrees)')
    # for i, seg in enumerate(segmentation.segments):
    #       cls = seg_class[i]
    #       ax3.plot(seg.t, np.array(seg.x)[:,1], color=COLORS[cls], alpha=.6)
    
    # handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in ['grey','blue']]
    # ax3.legend(handlelist,['other','saccade'])
    
    # f.show()
    
    times = []
    xcoords = []
    ycoords = []
    for seg in segmentation.segments:
        times.append(seg.t)
        xcoords.append(np.array(seg.x)[:,0])
        ycoords.append(np.array(seg.x)[:,1])
    xcoords = pd.DataFrame(xcoords, columns = ['x_begin', 'x_end'])
    ycoords = pd.DataFrame(ycoords, columns = ['y_begin', 'y_end'])
    times = pd.DataFrame(times, columns = ['ts_begin', 'ts_end'])
    segdf = pd.concat([xcoords,ycoords,times],axis=1).reset_index()
    segdf['sclass'] = seg_class
    
    segdf_long = pd.melt(segdf, id_vars = ['index', 'sclass'])
    segdf_long['var'], segdf_long['timept'] = segdf_long['variable'].str.split('_', 1).str
    segdf_long = segdf_long[['index', 'sclass','value', 'var', 'timept']].copy()
    
    segdf_long = segdf_long.pivot_table(index=['index', 'sclass', 'timept'], columns=['var'], values='value').reset_index().copy()
    
    saccades = segdf_long[segdf_long.sclass == 2]
    saccades = saccades.rename(columns={"index": "segment_index"}).reset_index()
    saccades = saccades[['timept', 'ts', 'x', 'y', 'segment_index']].copy()

    def merge_saccades(ds):

        #remove duplicate ts --> end/begin that have the same ts.
        remove_border = ds.duplicated(subset=['ts'], keep=False)
        ds = ds[~remove_border]
        # recalculate segment index
        ds['segment_index_old'] = ds['segment_index'].copy()
        ds['segment_index'] = ds.groupby('timept').cumcount()+1
        return ds
    
    saccades_merged = merge_saccades(saccades)    
    
    df = df.reset_index(drop=True).merge(saccades_merged, how = 'left').copy()

    return df


def calc_radius_phase(data):

    #calculate radius of gaze
    t = data[['target_x', 'target_y']].values
    data['t_dist'] = np.linalg.norm(t, axis=1)
    
    g = data[['gazeX_s', 'gazeY_s']].values
    data['g_dist'] = np.linalg.norm(g, axis=1)
    data['gaze_error'] = np.linalg.norm(g - t, axis=1) 
    data['radius_error'] = data.g_dist - data.t_dist
    
    data['trialbegin_ts'] = data.groupby('trial_number')['ts'].transform(lambda x: (x-x.min()))
    

    #make trial ts (= time from disappearance)
    data['trial_ts'] = data.groupby(['trial_number', 'is_visible'])['ts'].transform(lambda x: (x-x.min()))
    data['trial_ts'][data.is_visible != 0] = np.nan
    

    def get_angles(td):
         target = td[['target_x', 'target_y']].values
         gaze = td[['gazeX_s', 'gazeY_s']].values
         startangle = np.arctan2(*target[0])
         a = -startangle 
         c, s = np.cos(a), np.sin(a)
         R = np.array([[c, -s], [s, c]])
         target = np.einsum("ik,kl->il", target, R)
         gaze = np.einsum("ik,kl->il", gaze, R)
       
         td['target_angles'] = np.unwrap(np.arctan2(target[:,0], target[:,1]))
         td['gaze_angles'] = np.unwrap(np.arctan2(gaze[:,0], gaze[:,1]))
         return td
    
    dfangles = data.groupby('trial_number').apply(get_angles)
    dfangles['phase_error'] = dfangles.gaze_angles - dfangles.target_angles
    
    return dfangles




def plot_saccades_hidden_magn(f1):
    
    p = str(f1.name)
        
    f, (ax1, ax3) = plt.subplots(2, 1, sharex=True, figsize=(16,9))
    f.suptitle('Participant ' + p + ' saccades (hidden target)')
    
  #  ax1.scatter(f1.trial_ts[f1.timept == 'begin'], f1.radius_error[f1.timept == 'begin'], color='green', alpha=.3, label = 'launch')
    ax1.plot(f1.trial_ts[f1.timept == 'begin'], f1.radius_error[f1.timept == 'begin'], '.', color='green', alpha=.5, label = 'launch', markersize=4)
    for i, d in f1.groupby('segment_index'):
        ax1.plot(d.trial_ts, d.radius_error, color='black', alpha=.1)
    ax1.legend()
    ax1.axhline(y=0, color='black', alpha=.5)
    ax1.set_ylabel('Radius error')     
    ax1.set_ylim(-15, 15)
   
 #    ax2.axhline(y=0, color='black', alpha=.5)
 #  #  ax2.scatter(f1.trial_ts[f1.timept == 'begin'], f1.gaze_error[f1.timept == 'begin'], color='green', alpha=.3, label = 'launch')
 #    ax2.plot(f1.trial_ts[f1.timept == 'end'], f1.gaze_error[f1.timept == 'end'], '.', color='green', alpha=.8, label = 'landing', markersize=4)
 #    for i, d in f1.groupby('segment_index'):
 #        ax2.plot(d.trial_ts, d.gaze_error, color='black', alpha=.2)
 #    ax2.set_ylabel('Error magniturde')
 # #   ax2.set_ylim(-10, 10)

  #  ax3.scatter(f1.trial_ts[f1.timept == 'begin'], f1.phase_error[f1.timept == 'begin'], color='green', alpha=.3, label = 'launch')
    ax3.plot(f1.trial_ts[f1.timept == 'begin'], f1.phase_error[f1.timept == 'begin'], '.', color='green', alpha=.5, label = 'launch', markersize=4)
    for i, d in f1.groupby('segment_index'):
        ax3.plot(d.trial_ts, d.phase_error, color='black', alpha=.1)
    ax3.axhline(y=0, color='black', alpha=.5)
    ax3.set_ylabel('Phase error')
    ax3.set_ylim(-2,2)
    ax3.set_xlabel('Time from disappearance (s)')
    
    return f



def plot_saccades_vis_magn(f1):
    
    p = str(f1.name)
        
    f, (ax1, ax3) = plt.subplots(2, 1, sharex=True, figsize=(16,9))
    f.suptitle('Participant ' + p + ' saccades (visible target)')
    
  #  ax1.scatter(f1.trialbegin_ts[f1.timept == 'begin'], f1.radius_error[f1.timept == 'begin'], color='green', alpha=.3, label = 'launch')
    ax1.plot(f1.trialbegin_ts[f1.timept == 'begin'], f1.radius_error[f1.timept == 'begin'], '.', color='green', alpha=.5, label = 'launch', markersize=4)
    for i, d in f1.groupby('segment_index'):
        ax1.plot(d.trialbegin_ts, d.radius_error, color='black', alpha=.1)
    ax1.legend()
    ax1.axhline(y=0, color='black', alpha=.5)
    ax1.set_ylabel('Radius error')     
    ax1.set_ylim(-15, 15)
   
 #    ax2.axhline(y=0, color='black', alpha=.5)
 #  #  ax2.scatter(f1.trialbegin_ts[f1.timept == 'begin'], f1.gaze_error[f1.timept == 'begin'], color='green', alpha=.3, label = 'launch')
 #    ax2.plot(f1.trialbegin_ts[f1.timept == 'end'], f1.gaze_error[f1.timept == 'end'], '.', color='green', alpha=.8, label = 'landing', markersize=4)
 #    for i, d in f1.groupby('segment_index'):
 #        ax2.plot(d.trialbegin_ts, d.gaze_error, color='black', alpha=.2)
 #    ax2.set_ylabel('Error magniturde')
 # #   ax2.set_ylim(-10, 10)

  #  ax3.scatter(f1.trialbegin_ts[f1.timept == 'begin'], f1.phase_error[f1.timept == 'begin'], color='green', alpha=.3, label = 'launch')
    ax3.plot(f1.trialbegin_ts[f1.timept == 'begin'], f1.phase_error[f1.timept == 'begin'], '.', color='green', alpha=.5, label = 'launch', markersize=4)
    for i, d in f1.groupby('segment_index'):
        ax3.plot(d.trialbegin_ts, d.phase_error, color='black', alpha=.1)
    ax3.axhline(y=0, color='black', alpha=.5)
    ax3.set_ylabel('Phase error')
    ax3.set_ylim(-2,2)
    ax3.set_xlabel('Time (s)')
    
    return f


dfs = dfr.reset_index(drop=True).groupby('participant').apply(segment_hmm)

#dfs.to_pickle('data/segment_saccades.pickle')
#dfs = pd.read_pickle('data/segment_saccades.pickle')

# calc_radius_phase for visible and hidden separately 

df_hid = dfs[(dfs.is_visible==0)] # excludes target
df_vis = dfs[(dfs.is_visible==1) & (dfs.is_cue == 1)]

rp_hid = df_hid.reset_index(drop=True).groupby('participant').apply(calc_radius_phase)
rp_vis = df_vis.reset_index(drop=True).groupby('participant').apply(calc_radius_phase)


hid_saccades_both_plots = rp_hid[rp_hid.phase_error > -3].groupby('participant').apply(plot_saccades_hidden_magn)
pp = PdfPages('saccades_hidden.pdf')
for i in range(1,11):
    pp.savefig(hid_saccades_both_plots[i])
pp.close()    

# TODO: why starting radius negative? "trial" starts too soon after feedback?
# note some trials are 4 s because they're fully visible (1st block)
vis_saccades_both_plots = rp_vis.groupby('participant').apply(plot_saccades_vis_magn)
pp = PdfPages('saccades_visible.pdf')
for i in range(1,11):
    pp.savefig(vis_saccades_both_plots[i])
pp.close()      

        