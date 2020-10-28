# Attempt at syncing and merging all (relevant) data into
# one data frame in one pipeline.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import cv2
import pickle
import get_markers
from scipy.interpolate import interp1d
import time
import gzip
import json
from pprint import pprint
import get_markers
import extract_log

camera_spec = "sdcalib.rmap.full.camera.pickle"
#camera = pickle.load(open(camera_spec, 'rb'), encoding='bytes')

session_index = json.load(open("session_index.json"))

def get_marker_positions(pupil_base, camera_spec=camera_spec):
    """Compute marker positions or load them from cache"""
    pupil_base = Path(pupil_base)
    output = pupil_base / "markers_new.npy"
    if output.exists():
        return np.load(output, allow_pickle=True)
    
    return get_markers.marker_positions(str(camera_spec), str(pupil_base / "world.mp4"), str(output))

def denormalize(x, y, width, height, flip_y=False):
    x *= width
    if flip_y:
        y = 1-y
    y *= height
    return x,y

def marker_corners(x,y,l):
    c0 = [x,y]
    c1 = [x + l, y]
    c2 = [x + l, y + l]
    c3 = [x, y + l]
    return [c0, c1, c2, c3]

def map_to_screen_coords(camera, gaze_data, markerss, marker_times):
    width, height = camera[b'resolution']

    cam_m = camera[b'camera_matrix']
    cam_dist = camera[b'dist_coefs']
    assert camera[b'distortion_model'] == b'fisheye'
    newcamera = camera[b'rect_camera_matrix']
    #gaze_x, gaze_y = gaze_data[['norm_pos_x', 'norm_pos_y']].T.copy()
    #gaze_x, gaze_y = denormalize(x, y, width, height, flip_y=True)
    
    h, w = 1080, 1920    
    size = 0.1
    margin_scale = 75/512
    x = h*size*margin_scale
    y = h*size*margin_scale
    l = (1 - 2*margin_scale) * h*size

    id0 = marker_corners(x, h - y - l, l)
    id1 = marker_corners(w - x - l, h - y - l, l)
    id2 = marker_corners(x,y,l)
    id3 = marker_corners(w - x - l, y, l)
    marker_dict = {"0": id0, "1": id1, "2": id2, "3": id3}
    
    # This is probably not the nicest way to handle
    # missing markers
    marker_data = {k: [] for k in marker_dict}
    for i, markers in enumerate(markerss):
        markers = {str(m['id']): m for m in markers['markers']}
        for k in marker_dict:
            if k not in markers:
                marker_data[k].append(np.zeros((4, 2)) + np.nan)
                continue
            marker_data[k].append(np.array(markers[k]['verts']).reshape(4, 2))

    marker_interps = {
            k: interp1d(marker_times, np.array(marker_data[k]), axis=0, bounds_error=False)
            for k in marker_dict
            }

    gazes = []
    heads = []

    for ts, x, y in gaze_data[['gaze_timestamp', 'norm_pos_x', 'norm_pos_y']].values:
        #ts = row.gaze_timestamp
        #x = row.norm_pos_x
        #y = row.norm_pos_y
        camera_coords = []
        screen_coords = []
        for k, interp in marker_interps.items():
            verts = interp(ts)
            if np.any(np.isnan(verts)): continue
            camera_coords.extend(verts)
            screen_coords.extend(marker_dict[k])

        camera_coords = np.float32(camera_coords).reshape(-1, 1, 2)
        screen_coords = np.float32(screen_coords).reshape(-1, 1, 2)


        if len(camera_coords) < 8:
            gazes.append([np.nan, np.nan])
            heads.append([np.nan, np.nan])
            continue

        M, mask = cv2.findHomography(camera_coords, screen_coords)
        
        

        gaze = np.array(denormalize(x, y, width, height, flip_y=True))
        gaze = cv2.fisheye.undistortPoints(gaze.reshape(-1, 1, 2), cam_m, cam_dist, P=newcamera)
        
        head = np.array(denormalize(0.5, 0.5, width, height, flip_y=True))
        head = cv2.fisheye.undistortPoints(head.reshape(-1, 1, 2), cam_m, cam_dist, P=newcamera)
        
        gaze = cv2.perspectiveTransform(gaze, M).reshape(2)
        head = cv2.perspectiveTransform(head, M).reshape(2)
        
        gazes.append(gaze)
        heads.append(head)
    
    gaze_data['gaze_screen_x'], gaze_data['gaze_screen_y'] = np.array(gazes).T
    gaze_data['head_screen_x'], gaze_data['head_screen_y'] = np.array(heads).T
    
    return gaze_data

def load_experiment(experiment_log):
    log = extract_log.load_data((json.loads(r) for r in open(experiment_log)))
    return log

def sync_and_merge_session(gaze_export, pupil_base, experiment_base):
    #camera_spec = "sdcalib.rmap.full.camera.pickle"
    camera = pickle.load(open(camera_spec, 'rb'), encoding='bytes')
    
    gaze_data = pd.read_csv(gaze_export)
    
    marker_times = np.load(pupil_base / "world_timestamps.npy")
    markerss = get_marker_positions(pupil_base)
    gaze_data = map_to_screen_coords(camera, gaze_data, markerss, marker_times)

    target = load_experiment(experiment_base / "trajlog.jsons")
    
    # TODO: Better sync!!
    # TODO: Refactor outta here!
    system_to_recv = np.polynomial.Polynomial.fit(target.system_ts.values, target.recv_ts_mono, 1)
    target['pupil_ts'] = system_to_recv(target.system_ts.values)

    data = gaze_data
    def resample(field, **kwargs):
        return interp1d(target['pupil_ts'].values, target[field].values, axis=0, bounds_error=False, **kwargs)(data['gaze_timestamp'].values)
    
    # Resample position using the default (linear) interpolation
    data['target_x'] = resample('x')
    data['target_y'] = resample('y')
    
    # Resample flags using nearest 
    flagfields = 'is_visible', 'scenario', 'signtype', 'trial_number'
    match = interp1d(target.pupil_ts.values, np.arange(len(target)), kind='nearest', bounds_error=False, fill_value=(0, -1))(data.gaze_timestamp.values).astype(int)
    data['log_match_ts'] = target.pupil_ts.values[match]
    for f in flagfields:
        data[f] = target[f].values[match]

    return data

def onlyone(itr):
    lst = list(itr)
    assert len(lst) == 1, "Can handle only exactly one item!"
    return lst[0]

def sync_and_merge_participant(participant_id, data_dir='../data'):
    # Find export file

    session = session_index[str(participant_id)]
    data_dir = Path(data_dir)
    
    participant_id = int(participant_id)
    ppstr = f"{participant_id:02d}"
    export_file = onlyone(data_dir.glob(f"**/{ppstr}/*/exports/*/gaze_positions.csv"))
    
    # Find pupil dir
    pupil_base = onlyone(data_dir.glob("**/"+session['pupil_folder']))
    
    # Find experiment log
    experiment_base = onlyone(data_dir.glob(f"**/"+session['log_folder']))

    #print(experiment_log)
    #print(pupil_base)
    #print(export_file)
    #
    return sync_and_merge_session(export_file, pupil_base, experiment_base)

def sync_and_merge(output):
    data = []
    for pid in session_index:
        print("Processing participant", pid)
        d = sync_and_merge_participant(pid)
        d['participant_id'] = int(pid)
        data.append(d)

    data = pd.concat(data)
    data.to_parquet(output)

if __name__ == '__main__':
    import argh
    import sys
    #argh.dispatch_command(sync_and_merge_participant)
    argh.dispatch_command(sync_and_merge)
    #argh.dispatch_command(get_marker_positions)
    #get_marker_positions(sys.argv[1])
