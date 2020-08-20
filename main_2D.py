import numpy as np

from capture import Capture, CaptureSet, NormalizedCaptureSet
import utils, interpolation

#cap_set = CaptureSet("../../data/captures/meetingRoom_360/raw")
#A = cap_set.get_capture(0).pos
#B = cap_set.get_capture(1).pos
#C = cap_set.get_capture(3).pos
#D = utils.get_point_on_plane(A, B, C)#, 0.5, 0.5)

#cap_set.draw_scene()
#cap_set.draw_scene(indices=[22, 24, 27], s_points=np.array([D]))

cap_set_norm = NormalizedCaptureSet("../../data/captures/meetingRoom_360/normalized")

i = [44, 23, 56]
caps = cap_set_norm.get_captures([i[0], i[1], i[2]])
cap_set_norm.draw_scene(indices=[i[0], i[1], i[2]])
for dist in np.linspace(0,1,10):
    D_pos = utils.get_point_on_plane(caps[i[0]].pos, caps[i[1]].pos, caps[i[2]].pos, dist1=0.0, dist2=dist)
    inset = range(cap_set_norm.get_size())
    interpolation.interpolate_nD(cap_set_norm, inset, [D_pos], hack=dist)

#inset = list(range(22)) + list(range(23,cap_set_norm.get_size()))
#inset = [22, 24, 27, 55]



