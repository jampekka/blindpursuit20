import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tyxplot import Tyxplot

def plot_sessions():
    data = pd.read_parquet('../preprocessing/gaze_and_target.parquet')
    #data = pd.read_parquet('../preprocessing/test.parquet')
    
    data = data.query('scenario == "swing"')

    for (p, t), sd in data.groupby(['participant_id', 'trial_number']):
        end = np.flatnonzero(sd.signtype.values != 'cue')[0]
        sd = sd.iloc[:end]
        if len(sd) == 0: continue
        ts = sd.gaze_timestamp - sd.gaze_timestamp.iloc[0]
        if np.all(sd.is_visible): continue
        center_x = 1920/2
        center_y = 1080/2

        gts = np.where(sd.confidence > 0.7, ts.values, np.nan)

        gaze_phase = np.unwrap(np.arctan2(sd.gaze_screen_x - center_x, sd.gaze_screen_y - center_y))
        
        target_phase = np.unwrap(np.arctan2(sd.target_x - center_x, sd.target_y - center_y))
        
        phase_error = gaze_phase - target_phase
        #plt.plot(ts, target_phase, color='green')
        #plt.plot(ts, gaze_phase, color='black')
        #plt.figure()
        
        tyx = Tyxplot()
        plt.title(f"{p}, {t}, new")

        tyx.plot(gts, sd.head_screen_x, sd.head_screen_y, label='Head', color='blue')
        
        tyx.plot(gts, sd.gaze_screen_x, sd.gaze_screen_y, label='Gaze', color='black')
        #plt.plot(gts, np.degrees(phase_error), label='Gaze', color='black')
        vts = np.where(sd.is_visible.values, ts.values, np.nan)
        tyx.plot(vts, sd.target_x, sd.target_y, color='green', label='Visible')
        #plt.plot(vts, np.degrees(target_phase)*0, label='Target visible', color='green')
        vts = np.where(sd.is_visible.values, np.nan, ts.values)
        tyx.plot(vts, sd.target_x, sd.target_y, color='red', label='Hidden')
        #plt.plot(vts, np.degrees(target_phase)*0, label='Target hidden', color='red')
        plt.show()
 
if __name__ == '__main__':
    plot_sessions()
