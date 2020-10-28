import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_sessions():
    data = pd.read_parquet('../preprocessing/gaze_and_target.parquet')
    
    data.query('scenario == "swing"', inplace=True)
    for (p, t), sd in data.groupby(['participant_id', 'trial_number']):
        end = np.flatnonzero(sd.signtype.values == 'target')[0]
        sd = sd.iloc[:end]
        ts = sd.gaze_timestamp - sd.gaze_timestamp.iloc[0]

        plt.plot(ts, sd.gaze_screen_x, label='Gaze', color='black')
        
        vts = np.where(sd.is_visible.values, ts.values, np.nan)
        plt.plot(vts, sd.target_x, label='Target visible', color='green')
        vts = np.where(sd.is_visible.values, np.nan, ts.values)
        plt.plot(vts, sd.target_x, label='Target hidden', color='red')
        
        plt.ylim(0, 1920)
        plt.show()
 
if __name__ == '__main__':
    plot_sessions()
