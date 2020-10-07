
#%%
import numpy as np
import cv2
import pickle
from scipy.interpolate import interp1d
from file_methods import load_object



def denormalize(pos, width, height, flip_y=False):
    x = pos[0]
    y = pos[1]
    x *= width
    if flip_y:
        y = 1-y
    y *= height
    return x,y

def ref_surface_to_img(pos,m_to_screen):
    shape = pos.shape
    pos.shape = (-1,1,2)
    new_pos = cv2.perspectiveTransform(pos,m_to_screen )
    new_pos.shape = shape
    return new_pos

def get_marker_positions_pixels(srf_data_file):
    corners = [[0,0],[0,1],[1,1],[1,0]]
    data = []
    for d,i in zip(srf_data_file,range(len(srf_data_file))):
        if d is not None:
            data.append([i,[denormalize(ref_surface_to_img(np.array(c,dtype=np.float32),d['m_to_screen']),(1280,720)) for c in corners]])
        else:
            data.append([i,None])    
    return data

def screen_coords(pos, M):
        pos = np.float32([pos])
        shape = pos.shape
        pos.shape = (-1,1,2)
        new_pos = cv2.perspectiveTransform(pos,M)
        new_pos.shape = shape
        new_pos = new_pos[0]
        new_pos = (new_pos[0], new_pos[1])
        return new_pos

def camera(camera_path):
    h, w = 720, 1280

    camera = pickle.load(open(camera_path, 'rb'), encoding='bytes')
    image_resolution = camera[b'resolution']
    
    if b'rect_map' not in camera:
        camera_matrix = camera[b'camera_matrix']
        camera_distortion = camera[b'dist_coefs']
        rect_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, camera_distortion, image_resolution, 0.0)
        rmap = cv2.initUndistortRectifyMap(
            camera_matrix, camera_distortion, None, rect_camera_matrix, image_resolution,
            cv2.CV_32FC1)
    else:
        rmap = camera[b'rect_map']
        rect_camera_matrix = camera[b'rect_camera_matrix']

    cam_m, cam_dist, image_resolution, newcamera = camera[b'camera_matrix'], camera[b'dist_coefs'], camera[b'resolution'], rect_camera_matrix
    
    return cam_m, cam_dist, newcamera, rmap

def marker_corners(x,y,l):
    c0 = [x,y]
    c1 = [x + l, y]
    c2 = [x + l, y + l]
    c3 = [x, y + l]
    return [c0, c1, c2, c3]

def correlate_data(data,timestamps):
    timestamps = list(timestamps)
    data_by_frame = [[] for i in timestamps]

    frame_idx = 0
    data_index = 0

    while True:
        try:
            datum = data[data_index]
            ts = ( timestamps[frame_idx]+timestamps[frame_idx+1] ) / 2.
        except IndexError:
            break

        if datum['timestamp'] <= ts:
            datum['index'] = frame_idx
            data_by_frame[frame_idx].append(datum)
            data_index +=1
        else:
            frame_idx+=1
    return data_by_frame

#%%

#path = 'D:/BlindPursuit/pupil/10/000/'
path = sys.argv[1]
gaze_list = load_object(path + "pupil_data_exp")
timestamps = np.load(path + "world_timestamps.npy")


# %%
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

synchedGaze = correlate_data(gaze_list, timestamps) 

#hack
# save np.load
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
marker_list = np.load(path + "markers_new.npy")
# restore np.load for future normal usage
np.load = np_load_old


camera_path = 'sdcalib.rmap.full.camera.pickle'
cam_m, cam_dist, newcamera, rmap = camera(camera_path)
marker_data = {"0": [], "1": [], "2": [], "3": []}
marker_timestamps = {"0": [], "1": [], "2": [], "3": []}
markers_in_frames = []
      
for i in range(len(marker_list)): 
    markers = marker_list[i]
    ts = timestamps[i]
    found = []
    markers = markers['markers']
    for marker in markers:
        if str(marker['id']) in marker_dict.keys():
            found.append(marker['id'])
            marker_verts = np.array(marker['verts']).reshape(-1,2)
            marker_data[str(marker['id'])].append(marker_verts)
            marker_timestamps[str(marker['id'])].append(ts)
    markers_in_frames.append(found) 

marker_interpolators = {}
for key in marker_data.keys():
    marker_interp = interp1d(marker_timestamps[key], marker_data[key], axis=0, bounds_error=False)
    marker_interpolators[key] = marker_interp

#%%
###### craniotopic coordinates
#orig_gaze_data = []
#for frame in synchedGaze:
#    for g in frame:
#        ts = g['timestamp']
#        index = g['index']
#        conf = g['confidence']
#        if len(markers_in_frames[index]) > 1:
#            gaze = np.float32(g['norm_pos'])
#            orig_gaze_data.append([ts, index, conf, gaze[0], gaze[1]])
#orig_gaze_data = np.array(orig_gaze_data)
#print(orig_gaze_data.shape)
#np.savetxt(path + "orig_coords.csv", orig_gaze_data, delimiter=",")

#%%

screen_gaze_data = []

cap = cv2.VideoCapture(path + "world.mp4")
visualize = False
for frame in synchedGaze:
    for g in frame:
        ts = g['timestamp']
        index = g['index']
        conf = g['confidence']
        if len(markers_in_frames[index]) > 1:
            screen_positions = []
            marker_positions = []
            for m in markers_in_frames[index]:
                marker_verts = marker_interpolators[str(m)](ts)
                screen_verts = np.float32(marker_dict[str(m)])
       #         np.append(screen_positions, screen_verts)
       #         np.append(marker_positions, marker_verts)
                screen_positions.append(screen_verts)
                marker_positions.append(marker_verts)
                
            marker_positions = np.float32(marker_positions).reshape(-1,1,2)
            screen_positions = np.float32(screen_positions).reshape(-1,1,2)
                
            M, mask = cv2.findHomography(marker_positions,screen_positions, 0)

            gaze = denormalize(g['norm_pos'],1280, 720,flip_y=True) 
            gaze = np.float32(gaze).reshape(-1,1,2)
            gaze = cv2.fisheye.undistortPoints(gaze,cam_m,cam_dist, P=newcamera)
            
            head = denormalize([0.5, 0.5], 1280, 720,flip_y=True) 
            head = np.float32(head).reshape(-1, 1, 2)
            head = cv2.fisheye.undistortPoints(head ,cam_m,cam_dist, P=newcamera)
            screen_pos = tuple(cv2.perspectiveTransform(gaze, M).reshape(2))
            screen_pos_head = tuple(cv2.perspectiveTransform(head, M).reshape(2))
            screen_gaze_data.append([ts, index, conf, *screen_pos, *screen_pos_head])
    if visualize and len(markers_in_frames[index]) > 1:
        cap.set(1, index)
        ret, oframe = cap.read()

        frame = cv2.remap(oframe, rmap[0], rmap[1], cv2.INTER_LINEAR)     
        frame = cv2.warpPerspective(frame,M,(1920,1080))

        cv2.circle(frame, (screen_pos[0], screen_pos[1]), 10, (255,0, 0), thickness = -1)
        cv2.circle(frame, (screen_pos_head[0], screen_pos_head[1]), 10, (0,255, 0), thickness = -1)
        frame=cv2.flip(frame,0)
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) 
        cv2.imshow('undistorted',frame)
        cv2.waitKey(1)
screen_gaze_data = np.array(screen_gaze_data)
print(screen_gaze_data.shape)
np.savetxt(path + "screen_coords.csv", screen_gaze_data, delimiter=",")

