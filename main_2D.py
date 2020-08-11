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
#cap_set_norm.draw_scene()

caps = cap_set_norm.get_captures([22, 24, 27])
D_pos = utils.get_point_on_plane(caps[22].pos, caps[24].pos, caps[27].pos, dist1=0, dist2=1)
#cap_set_norm.draw_scene(indices=[22], s_points=np.array([D_pos]))
#print(D_pos)
interpolation.interpolate_nD(cap_set_norm, [22], [D_pos])



