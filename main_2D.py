import numpy as np

from capture import Capture, CaptureSet
import utils

cap_set = CaptureSet("../../data/captures/meetingRoom_360")
A = cap_set.get_capture(22).pos
B = cap_set.get_capture(24).pos
C = cap_set.get_capture(27).pos
D = utils.get_point_on_plane(A, B, C)#, 0.5, 0.5)

#cap_set.draw_scene()
cap_set.draw_scene(indices=[22, 24, 27], s_points=np.array([D]))


