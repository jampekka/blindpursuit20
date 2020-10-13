import numpy as np
import pandas as pd
from file_methods import save_object
import matplotlib.pyplot as plt

# gaze_positions to same structure as pupil_data['gaze_positions']
#  {'norm_pos': [XX XX], 'confidence' : X, 'timestamp' : X, 'base_data' : 
#  [{'timestamp': X, 'id': 0}, {'timestamp': X, 'id': 1}]}


path = 'D:/BlindPursuit/pupil/06/000/'

gaze_positions_new = pd.read_csv(path + 'exports/001/gaze_positions.csv', sep=',')

tss = gaze_positions_new.groupby(['gaze_timestamp']).size()
plt.plot(tss) # nice plot
(np.max(gaze_positions_new['gaze_timestamp']) - np.min(gaze_positions_new['gaze_timestamp']))/60
# length of recording

#remove duplicate timestamps
gaze_positions_new = gaze_positions_new.drop_duplicates(subset=['gaze_timestamp'], keep='first')

#make normpos lists
normposx = np.array(gaze_positions_new['norm_pos_x'])
normposy = np.array(gaze_positions_new['norm_pos_y'])
norm_positions = [[normposx[i], normposy[i]] for i in range(len(normposx))]

#make dict
q = list(zip(norm_positions, np.array(gaze_positions_new['confidence']), np.array(gaze_positions_new['gaze_timestamp'])))
qdict = [{'norm_pos' : x,'confidence' : y, 'timestamp' : z} for x, y, z in q]

save_object(qdict, path + 'pupil_data_exp')


