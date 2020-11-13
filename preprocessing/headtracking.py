import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import cv2
from marker_spec import get_marker_spec
from scipy.interpolate import interp1d
from filterpy.kalman import UnscentedKalmanFilter, JulierSigmaPoints, unscented_transform

def project_plane_markers(markers, state):
    x_w, y_w = markers.T
    x, y, z = state[0]
    heading, pitch, roll = state[1]
    # TODO: SIMPLIFY!!
    sh, ch = np.sin(heading), np.cos(heading)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sr, cr = np.sin(roll), np.cos(roll)

    px = (x*sh*sp*sr + x*ch*cr - x_w*(sh*sp*sr + ch*cr) + y*sh*sp*cr - y*sr*ch - y_w*(sh*sp*cr - sr*ch) + z*sh*cp)/(-x*sh*cr + x*sp*sr*ch + x_w*(sh*cr - sp*sr*ch) + y*sh*sr + y*sp*ch*cr - y_w*(sh*sr + sp*ch*cr) + z*ch*cp)
    py = (x*sr*cp - x_w*sr*cp + y*cp*cr - y_w*cp*cr - z*sp)/(-x*sh*cr + x*sp*sr*ch + x_w*(sh*cr - sp*sr*ch) + y*sh*sr + y*sp*ch*cr - y_w*(sh*sr + sp*ch*cr) + z*ch*cp)
    return np.array([px, py]).T

def project_to_zero_plane(state, gaze_heading, gaze_pitch):
    x, y, z, heading, pitch, roll = state[:6]
    x_proj = x - z*np.sin(roll)*np.tan(gaze_pitch + pitch) + z*np.cos(roll)*np.tan(gaze_heading + heading)/np.cos(gaze_pitch + pitch)
    y_proj = y - z*np.sin(roll)*np.tan(gaze_heading + heading)/np.cos(gaze_pitch + pitch) - z*np.cos(roll)*np.tan(gaze_pitch + pitch)
    return np.array([x_proj, y_proj])

def predict_state(state, dt):
    # TODO: Avoid blowing up the state variance when there
    # are no measurement. E.g. decay to x0, P0
    state[2] -= state[2]*0.1*dt
    state[3] -= state[3]*0.1*dt
    state[0] += state[2]*dt
    state[1] += state[3]*dt

    return state

def confidence_ellipse(mean, cov, ax=None, n_std=1.0, facecolor='none', **kwargs):
    if ax is None:
        ax = plt.gca()
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    
    mean_x, mean_y = mean
    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def framer(camera, videofile):
    video = cv2.VideoCapture(videofile)
    rect_camera_matrix = camera[b'rect_camera_matrix']
    K, D, resolution, cm = camera[b'camera_matrix'], camera[b'dist_coefs'], camera[b'resolution'], rect_camera_matrix
    rmap = camera[b'rect_map']
    def get_frame(frameno):
        video.set(cv2.CAP_PROP_POS_FRAMES, frameno)
        ret, frame = video.read()
        frame = cv2.remap(frame, rmap[0], rmap[1], cv2.INTER_LINEAR)
        return frame
    return get_frame

def denormalize_camera_points(points, M):
    f_x, c_x = M[0, 0], M[0, -1]
    f_y, c_y = M[1, 1], M[1, -1]

    points[:,0] *= f_x
    points[:,0] += c_x

    points[:,1] *= -1
    points[:,1] *= f_y
    points[:,1] += c_y

    return points

def normalize_camera_points(points, M):
    f_x, c_x = M[0, 0], M[0, -1]
    f_y, c_y = M[1, 1], M[1, -1]

    points[:,0] -= c_x
    points[:,0] /= f_x

    points[:,1] -= c_y
    points[:,1] /= f_y
    points[:,1] *= -1

    return points


def poser(dt, M, markerss, world_markers):
    tracker = HeadTracker(dt)
    ts, ms, cs = [], [], []
    
    N = len(markerss)
    for i, row in enumerate(markerss):
        #print(i/N*100)
        #if i >= N: break
        markers = row['markers']
        
        world_positions = []
        screen_positions = []
        for m in markers:
            try:
                world = world_markers[str(m['id'])]
            except KeyError:
                continue
            
            if m['id_confidence'] < 0.7: continue
            for w, s in zip(world, m['verts']):
                world_positions.append(w)
                screen_positions.append(s)
        
        world_positions = np.array(world_positions).reshape(-1, 2)
        screen_positions = np.array(screen_positions).reshape(-1, 2)
        screen_positions = normalize_camera_points(screen_positions, M)
        
        m, c = tracker(world_positions, screen_positions)
        ms.append(m)
        cs.append(c)
    ms = np.array(ms)
    cs = np.array(cs)
    m, c, _ = tracker.kf.rts_smoother(ms, cs)
    rng = np.arange(len(ms))
    return interp1d(rng, ms, axis=0, bounds_error=False), interp1d(rng, cs, axis=0, bounds_error=False)


def plot_tracking(camera_spec, markerfile):
    camera = pickle.load(open(camera_spec, 'rb'), encoding='bytes')
    M = camera[b'rect_camera_matrix']
    cam_m = camera[b'camera_matrix']
    cam_dist = camera[b'dist_coefs']
    assert camera[b'distortion_model'] == b'fisheye'
    
    f_x, c_x = M[0, 0], M[0, -1]
    f_y, c_y = M[1, 1], M[1, -1]
    h = 1
    w = 16/9

    screen_corners = np.array([
        [w/2,h/2],
        [-w/2,h/2],
        [-w/2,-h/2],
        [w/2,-h/2],
        ])

    screen_corners_px = screen_corners.copy()
    screen_corners_px[:,0] += w/2
    screen_corners_px[:,1] += h/2
    screen_corners_px *= 1080
    

    markerss = np.load(markerfile, allow_pickle=True)
    world_markers = get_marker_spec()
    #poses = track_head(M, markerss, world_markers)
    head_m, head_c = poser(1/30, M, markerss, world_markers)
    print("Got poser")
    #dt = 1/30 # TODO: Use the actual timestamps
    #pos = np.array([0.0, 0.0, -1.0])
    #rot = np.array([0.0, 0.0, 0.0])
    #dpos = np.array([0.0, 0.0, 0.0])
    #drot = np.array([0.0, 0.0, 0.0])

    initialized = False

    all_world_positions = []
    for w in world_markers.values():
        for v in w:
            all_world_positions.append(v)
    all_world_positions = np.array(all_world_positions)

    
    # TODO: Tmp hack, no need for video
    from pathlib import Path
    get_frame = framer(camera, str(Path(markerfile).parent/'world.mp4'))
    
    video_ts = np.load(Path(markerfile).parent/'world_timestamps.npy')
    import msgpack
    gaze_data = msgpack.load(open(Path(markerfile).parent/'pupil_data', 'rb'))['gaze_positions']
    ts, gaze = zip(*((g['timestamp'], g['norm_pos']) for g in gaze_data))
    from scipy.interpolate import interp1d
    gaze = interp1d(ts, gaze, axis=0)
    tracker = HeadTracker(1/30)

    for i, row in enumerate(markerss):
        if i < 60000: continue
        markers = row['markers']
        pupil_ts = video_ts[i]
        if not markers: continue
        # Hack for now
        #if len(markers) < 2: continue
        
        world_positions = []
        screen_positions = []
        for m in markers:
            try:
                world = world_markers[str(m['id'])]
            except KeyError:
                continue
            
            if m['id_confidence'] < 0.7: continue
            for w, s in zip(world, m['verts']):
                world_positions.append(w)
                screen_positions.append(s)
        
        world_positions = np.array(world_positions).reshape(-1, 2)
        initialized = initialized or len(world_positions) >= 4*4
        #if not initialized:
        #    continue
        screen_positions = np.array(screen_positions).reshape(-1, 2)
        screen_positions = normalize_camera_points(screen_positions, M)
        
        #x, c = tracker(world_positions, screen_positions)
        
        x = head_m(i)
        c = head_c(i)


        #yield row['ts']
        #pred_pos = project_plane_markers(all_world_positions, kf.x.reshape(state_shape))

        #plt.plot(screen_positions[:,0], screen_positions[:,1], 'go')
        #plt.plot(pred_pos[:,0], pred_pos[:,1], 'ro')
        
        """
        sigmas_h = []
        for s in tracker.kf.sigmas_f:
            sigmas_h.append(project_plane_markers(np.array([[0.0,0.0]]), s.reshape(2, 3)))
        sigmas_h = np.array(sigmas_h).reshape(-1, 2)
        
        zp, S = unscented_transform(sigmas_h, tracker.kf.Wm, tracker.kf.Wc, 0)
        """
        #confidence_ellipse(zp, S, edgecolor='black')
        #plt.plot(*sigmas_h.T, 'ko')
        #print(S)
        #plt.plot(sigmas_h[:,0], sigmas_h[:,1], 'k.')
        
        #plt.xlim(-1, 1)
        #plt.ylim(-1, 1)
        #plt.pause(0.01)
        #plt.cla()
        #continue
        
        g = gaze(pupil_ts)
        
        proj_corners = project_plane_markers(screen_corners, x.reshape(2, 3))
        proj_corners = denormalize_camera_points(proj_corners, M)

        frame = get_frame(i)
        H = cv2.getPerspectiveTransform(proj_corners.astype(np.float32), screen_corners_px.astype(np.float32))
        screen = cv2.warpPerspective(frame, H, (1920, 1080))
        
        #cv2.circle(frame, tuple(g.astype(int)), 10, (255,0,0), thickness = -1)
        #cv2.polylines(frame, [proj_corners.astype(np.int32)], True, (255, 0, 0))
        #cv2.imshow("frame", frame)
        
        g[0] *= 1280
        g[1] = 1 - g[1]
        g[1] *= 720
        g = cv2.fisheye.undistortPoints(g.reshape(-1, 1, 2), cam_m, cam_dist).reshape(-1)
        #g = normalize_camera_points(g.reshape(1, 2), M).reshape(2)
        
        #gh, gp = np.arctan(g)
        gs = []
        gs = project_to_zero_plane(x, *np.arctan(-g)).reshape(1, 2)
        gs = np.array(gs).T
        gs[0] *= 1080
        gs[0] += 1920/2
        gs[1] *= 1080
        gs[1] += 1080/2
        for g in gs.T:
            cv2.circle(screen, tuple(g.astype(int)), 5, (255,0,0), thickness = -1)
        cv2.imshow("screen", screen[::-1])
        cv2.waitKey(1)
        #plt.xlim(-1, 1)
        #plt.ylim(-1, 1)
        #plt.pause(0.01)
        #plt.cla()

class HeadTracker:
    def __init__(self, dt, state=np.array([0.0, 0.0, -2.0, 0.0, 0.0, 0.0])):
        M = len(state)
        points = JulierSigmaPoints(M)
        def project(state, world_positions):
            return project_plane_markers(world_positions, state.reshape(2, 3)).reshape(-1)

        self.kf = kf = UnscentedKalmanFilter(dt=dt, dim_x=M, dim_z=1, points=points,
            #fx=lambda X, dt: predict_state(X.reshape(4, -1), dt).reshape(-1),
            fx=lambda x, dt: x,
            hx=project,
            )
        z_dim = 2 # This changes according to the measurement
        kf.P = np.eye(M)*0.3
        kf.x = state.reshape(-1).copy() # Initial state guess
        self.z_var = 0.05**2
        #kf.R = z_var # Observation variance
        dpos_var = 0.01**2*dt
        drot_var = np.radians(1.0)**2*dt
        #kf.Q = np.diag([0]*3 + [0]*3 + [dpos_var]*3 + [drot_var]*3)
        kf.Q = np.diag([dpos_var]*3 + [drot_var]*3)
    
    def __call__(self, world_positions, screen_positions):
        self.kf.predict()
        if len(world_positions):
            measurement = screen_positions.reshape(-1)
            self.kf.update(measurement, R=np.diag([self.z_var]*len(measurement)),
                    world_positions=world_positions)
        return self.kf.x, self.kf.P


#def track_head(camera_spec, markerfile):
def track_head(camera_matrix, markerss, world_markers):
    M = camera_matrix
    
    f_x, c_x = M[0, 0], M[0, -1]
    f_y, c_y = M[1, 1], M[1, -1]
    h = 1
    w = 16/9

    dt = 1/30 # TODO: Use the actual timestamps
    
    pos = np.array([0.0, 0.0, -2.0])
    rot = np.array([0.0, 0.0, 0.0])
    dpos = np.array([0.0, 0.0, 0.0])
    drot = np.array([0.0, 0.0, 0.0])

    state = np.array([
        pos, rot,
        #dpos, drot
        ])

    state_shape = state.shape
    
    M = np.prod(state_shape)
    points = JulierSigmaPoints(M)

    def project(state, world_positions):
        return project_plane_markers(world_positions, state.reshape(state_shape)).reshape(-1)

    kf = UnscentedKalmanFilter(dt=dt, dim_x=M, dim_z=len(world_markers)*2, points=points,
            #fx=lambda X, dt: predict_state(X.reshape(4, -1), dt).reshape(-1),
            fx=lambda x, dt: x,
            hx=project,
            )
    
    z_dim = 2 # This changes according to the measurement
    kf.P = np.eye(M)*0.3
    kf.x = state.reshape(-1).copy() # Initial state guess
    z_var = 0.05**2
    kf.R = z_var # Observation variance
    dpos_var = 0.01**2*dt
    drot_var = np.radians(1.0)**2*dt
    #kf.Q = np.diag([0]*3 + [0]*3 + [dpos_var]*3 + [drot_var]*3)
    kf.Q = np.diag([dpos_var]*3 + [drot_var]*3)
    
    all_world_positions = []
    for w in world_markers.values():
        for v in w:
            all_world_positions.append(v)
    all_world_positions = np.array(all_world_positions)
    
    for row in markerss:
        markers = row['markers']
        
        world_positions = []
        screen_positions = []
        for m in markers:
            try:
                world = world_markers[str(m['id'])]
            except KeyError:
                continue
            
            if m['id_confidence'] < 0.7: continue
            for w, s in zip(world, m['verts']):
                world_positions.append(w)
                screen_positions.append(s)
        
        world_positions = np.array(world_positions).reshape(-1, 2)
        screen_positions = np.array(screen_positions).reshape(-1, 2)
        screen_positions[:,0] -= c_x
        screen_positions[:,0] /= f_x
        screen_positions[:,1] -= c_y
        screen_positions[:,1] /= -f_y
        
        kf.predict()

        if len(world_positions) >= 0:
            measurement = screen_positions.reshape(-1)
            kf.update(measurement, R=np.diag([z_var]*len(measurement)), world_positions=world_positions)
        yield kf.x.copy(), kf.P.copy()



if __name__ == '__main__':
    plot_tracking(*sys.argv[1:])

