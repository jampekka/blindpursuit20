def marker_corners(x,y,l):
    c0 = [x,y]
    c1 = [x + l, y]
    c2 = [x + l, y + l]
    c3 = [x, y + l]
    return [c0, c1, c2, c3]

def marker_corners(x,y,l):
    r = l/2
    left = x - r
    right = x + r
    top = y + r
    bottom = y - r

    return [
        [left, bottom],
        [right, bottom],
        [right, top],
        [left, top]
            ]

def get_marker_spec():
    #h, w = 1080, 1920
    h = 1
    w = 16/9
    marker_scale = 0.1
    
    marker_size = h*marker_scale
    marker_margin_scale = 74/512
    marker_margin = marker_margin_scale*marker_size

    marker_extent = marker_size - marker_margin*2

    #left = marker_margin
    #top = marker_margin
    #right = w - marker_size + marker_margin
    #bottom = h - marker_size + marker_margin
    
    left = -w/2 + marker_size/2
    right = w/2 - marker_size/2
    top = h/2 - marker_size/2
    bottom = -h/2 + marker_size/2

    markers = {
            "0": marker_corners(left, top, marker_extent),
            "1": marker_corners(right, top, marker_extent),
            "2": marker_corners(right, bottom, marker_extent),
            "3": marker_corners(left, bottom, marker_extent),
            }

    return markers
    margin = marker_margin_scale
    x = h*marker_size*margin
    y = h*marker_size*margin
    l = (1 - 2*margin)*h*marker_size

    id0 = marker_corners(x, h - y - l, l)
    id1 = marker_corners(w - x - l, h - y - l, l)
    id2 = marker_corners(x,y,l)
    id3 = marker_corners(w - x - l, y, l)
    marker_dict = {"0": id0, "1": id1, "2": id2, "3": id3}

    return marker_dict
