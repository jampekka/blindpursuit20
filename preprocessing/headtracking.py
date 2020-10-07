import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import cv2
from marker_spec import get_marker_spec


def track_head(camera_spec, markerfile):
    camera = pickle.load(open(camera_spec, 'rb'), encoding='bytes')
    rect_M = camera[b'rect_camera_matrix']
    markerss = np.load(markerfile, allow_pickle=True)
    world_markers = get_marker_spec()
    for row in markerss:
        markers = row['markers']
        if not markers: continue
        if len(markers) < 4: continue
        
        world_positions = []
        screen_positions = []
        for m in markers:
            try:
                world = world_markers[str(m['id'])]
            except KeyError:
                continue

            for w, s in zip(world, m['verts']):
                world_positions.append(w)
                screen_positions.append(s)
        
        world_positions = np.array(world_positions, dtype=np.float).reshape(-1, 2)
        world_positions = np.concatenate((world_positions, np.zeros(len(world_positions)).reshape(-1, 1)), axis=1)
        screen_positions = np.array(screen_positions, dtype=np.float).reshape(-1, 2)
        
        retval, rvec, tvec = cv2.solvePnP(world_positions, screen_positions, rect_M, None)
        M = np.zeros((4, 4))
        M[-1,-1] = 1
        M[:-1,-1] = tvec.reshape(-1)
        M[:3,:3] = cv2.Rodrigues(rvec)[0]
        M = np.linalg.inv(M)
        
        # TODO: Is this really the coordinate system? Verify
        direction = np.array([0, 1, 0])@M[:3,:3]
        origin = M[:-1,-1]
        dist = -origin[-1]/direction[-1]
        pos = origin + dist*direction
        yield row['ts'], pos, M
    

if __name__ == '__main__':
    import itertools
    data = track_head(*sys.argv[1:])
    #data = itertools.islice(data, 100)
    ts, pos, M = zip(*data)
    pos = np.array(pos)
    plt.plot(ts, pos[:,0])
    plt.ylim(-0.5, 0.5)
    plt.plot(ts, pos[:,1], color='C1')
    plt.ylim(-0.5, 0.5)
    plt.show()
