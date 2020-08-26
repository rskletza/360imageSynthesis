import numpy as np

from capture import Capture, CaptureSet, NormalizedCaptureSet
import utils, interpolation

"""
#cap_set = CaptureSet("../../data/captures/meetingRoom_360/raw")
cap_set = CaptureSet("../../data/captures/synthesized_checkersphere/second", radius=12)
i = [6, 5, 4]
"""
cap_set = NormalizedCaptureSet("../../data/captures/meetingRoom_360/normalized")
i = [22, 23, 27]

caps = cap_set.get_captures([i[0], i[1], i[2]])
#cap_set.draw_scene(indices=[i[0], i[1], i[2]])
#cap_set.draw_scene()
inset = list(range(1,cap_set.get_size()))
#interpolation.interpolate_nD(cap_set, [1], np.array([[-1.5, 0, 0], [-1, 0, 0], [-0.5, 0, 0], [0, 0, 0], [0.5,0,0], [1,0,0], [1.5,0,0]]))
#interpolation.interpolate_nD(cap_set, inset, np.array([[0, 0, 0]]))
interpolation.interpolate_nD(cap_set, inset, np.array([cap_set.get_position(0)]))

"""
#synthesize 10 points on a line between point i[0] and i[2]
for dist in np.round(np.linspace(0,1,10),2):
    D_pos = utils.get_point_on_plane(caps[i[0]].pos, caps[i[1]].pos, caps[i[2]].pos, dist1=0.5, dist2=dist)
#    inset = list(range(i)) + list(range(i+1,cap_set.get_size()))
    inset = i
    interpolation.interpolate_nD(cap_set, inset, [D_pos], hack=dist)

#synthesize each existing viewpoint from all N-1 points (excluding point to be synthesized
for i in range(cap_set.get_size()):
#    D_pos = utils.get_point_on_plane(caps[i[0]].pos, caps[i[1]].pos, caps[i[2]].pos, dist1=0.0, dist2=dist)
    D_pos = cap_set.get_position(i)
    inset = list(range(i)) + list(range(i+1,cap_set.get_size()))
    interpolation.interpolate_nD(cap_set, inset, [D_pos], hack=i)
"""

