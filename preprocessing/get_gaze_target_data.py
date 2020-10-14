import json
import numpy as np
import pandas as pd
import os.path

def load_jsons(lines):
    for line in lines:
        yield json.loads(line)

def safeget(p, *keys):
    for key in keys[:-1]:
        p = p.get(key, {})
    return p.get(keys[-1], None)

def dotget(p, path):
    return safeget(p, *path.split('.'))

def get_target_pos(row):
    m = dotget(row, 'data.pdBase.platform.elements')
    if m is None:
        return None
    # Bizarre...
    m = np.array([m[str(i)] for i in range(16)])
    m = m.reshape(4, 4).T
    return m[0:2,-1]

def load_log(rows):
    target_pos = np.nan
    target_positions = []
    is_visible = True
    for row in rows:
        p = get_target_pos(row)
        if p is not None:
            t = row['recv_ts_mono']
            target = dotget(row, 'data.pdBase.targetSign') 
            is_cue = target == 'cue'
            is_success = target == 'success'
            is_failure = target == 'failure'
            is_target = target == 'target'
            is_visible = dotget(row, 'data.pdBase.targetVisible')
            target_positions.append((t, *p, is_visible, is_cue, is_target, is_success, is_failure))

    return np.rec.fromrecords(target_positions, names='ts,x,y,is_visible,is_cue,is_target,is_success,is_failure')


def get_pupil_sync(rows):
    rowlist = []
    sys = []
    pupil = []
    try:
        for row in rows:
            rowlist.append(row)
            if row['topic'] not in ('pupil.0', 'pupil.1'): continue
            if 'timestamp' not in row['data']: continue
            sys.append(row['recv_ts_mono'])
            pupil.append(row['data']['timestamp'])
    except EOFError:
        pass
    fit = np.poly1d(np.polyfit(pupil, sys, 1))
    return fit

#%%

def main():

    kh = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    path = 'C:/Users/tuisku.tammi/Documents/PhD/pursuit_analysis/'

    paths = []
    for i in kh:
        pupil_path = os.path.join(path, 'pupil', i)
        log = os.path.join(path, 'behav', i, 'trajlog.jsons')
        paths.append([pupil_path, log])
    
    
    position_list = []
    gaze_data_list = []
    for pupil_path, log in paths:    
        positions = load_log(load_jsons(open(log))) 
        gaze = pd.read_csv(os.path.join(pupil_path, 'screen_coords.csv'), sep=',', names = ['pupilTime','index','conf','gazeX','gazeY'])
        positions['x'] = positions.x/2.0*1080 + 1920/2
        positions['y'] = positions.y/2.0*1080 + 1080/2
        
        positions = pd.DataFrame(positions)
        positions['participant'] = os.path.split(pupil_path)[1]
        positions.set_index('participant', inplace=True)
        gaze['participant'] = os.path.split(pupil_path)[1]
        gaze.set_index('participant', inplace=True)
    
        position_list.append(positions)
        gaze_data_list.append(gaze)
    
    position_data = pd.concat(position_list)
    gaze_data = pd.concat(gaze_data_list)

    position_data.to_csv('position_data.csv')
    gaze_data.to_csv('gaze_data.csv')

if __name__ == "__main__":
    main()
