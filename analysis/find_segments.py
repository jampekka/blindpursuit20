import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import scipy.ndimage
import nslr_hmm


dfr = pd.read_parquet('preprocessing/gaze_and_target.parquet')

# some renaming
dfr.rename(columns={"participant_id": "participant", "gaze_timestamp": "ts", 
                    "confidence": "conf", "gaze_screen_x": "gazeX", "gaze_screen_y": "gazeY"},
           inplace=True)

# Get low confidence samples
badones = (dfr.conf < 0.8).values
badones = scipy.ndimage.morphology.binary_dilation(badones, iterations=5)
   

dfr.loc[badones,['gazeX', 'gazeY']] = np.nan

#now clean target data
xspeed = dfr.target_x.diff()/dfr.ts.diff() 

badonest = (np.abs(xspeed) > 20000).values
badonest = scipy.ndimage.morphology.binary_dilation(badonest, iterations=5)

dfr.loc[badonest,['target_x', 'target_y']] = np.nan



def segment_hmm(df):

    # remove duplicated timestamps
    df = df.drop_duplicates(subset=['ts'], keep='first')    

    # remove NANs from gaze
    df = df[~((df.gazeX.isnull()) | (df.gazeY.isnull()))]
    

      # HMM    
    sample_class, segmentation, seg_class = nslr_hmm.classify_gaze(df.ts.values, df[['gazeX', 'gazeY']].values,
              structural_error=0.2, optimize_noise=False # Assume 0.2 degree noise
              )
      
    
    # recreate a new signal using the segmented results.
    gaze_interp = interp1d(segmentation.t, segmentation.x, axis=0, bounds_error=False)
    df['gazeX_s'], df['gazeY_s'] = gaze_interp(df.ts).T.copy()

        
    
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
    segdf['segment_class'] = seg_class
    
    segdf_long = pd.melt(segdf[['index', 'ts_begin', 'ts_end', 'segment_class']], id_vars = ['index', 'segment_class'])
    segdf_long['var'], segdf_long['timept'] = segdf_long['variable'].str.split('_', 1).str    
    segdf_long = segdf_long[['index', 'segment_class','value', 'var', 'timept']].copy()    
    segdf_long = segdf_long.pivot_table(index=['index', 'segment_class', 'timept'], columns=['var'], values='value').reset_index().copy()
  
    # merge consecutive segments of same class
    remove_border = segdf_long.duplicated(subset=['ts', 'segment_class'], keep=False)
    segdf_long = segdf_long[~remove_border]
    
    # recalculate segment index
    segdf_long['segment_index_old'] = segdf_long['index'].copy()
    segdf_long['segment_index'] = segdf_long.groupby('timept').cumcount()+1
    
    segdf_long['segment_index'][segdf_long.timept == 'end'] = np.nan # leave segment indices at begin
    segdf_long['segment_index'] = segdf_long.segment_index.bfill()
    
    segdf_long = segdf_long.pivot_table(index=['ts', 'segment_index'], columns=['timept'], values='segment_class').reset_index().copy()
    segdf_long.rename(columns={"segment_index": "begin_segment_index", "begin": "begin_seg_class", "end": "end_seg_class"}, inplace=True)
    

    df = df.reset_index(drop=True).merge(segdf_long, how = 'left').copy()

    return df



segmented_df = dfr.reset_index(drop=True).groupby('participant').apply(segment_hmm)

#segmented_df.to_pickle('segment_data_all.pickle')

#select only circular
segmented_df_circ = segmented_df[segmented_df.scenario == "swing"]

#segmented data to R
segmented_df_circ.reset_index(drop=True).to_feather('data/segments.feather')




        