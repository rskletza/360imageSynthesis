import numpy as np

from capture import Capture, CaptureSet
import utils, interpolation, optical_flow
from envmap import EnvironmentMap

"""
#cap_set = CaptureSet("../../data/captures/meetingRoom_360/raw")
cap_set = CaptureSet("../../data/captures/synthesized_checkersphere/second", radius=12)
i = [6, 5, 4]
cap_set = CaptureSet("../../data/captures/meetingRoom_360/normalized")
i = [22, 23, 27]
cap_set = CaptureSet("../../data/captures/synthesized_room/square_room_1res", in_place=True)
i = [0,1,2]
"""

cap_set = CaptureSet("../../data/captures/synthesized_room/square_room_textured_brick_small", blenderfile="../../data/captures/synthesized_room/square_room_textured_brick_small/room_square_textured_brick_small.blend")
i = [15,16,31]

caps = cap_set.get_captures([i[0], i[1], i[2]])
p = 11
point = cap_set.get_position(p)
#inset = list(range(p)) + list(range(p+1,cap_set.get_size()))
#inset = [15,16,17,22,24,29,30,31] #23
inset = [10,12,17,18,19] #11
#inset = [10,11,12,17,18,19] #4
#cap_set.draw_scene()
cap_set.draw_scene(indices=inset, s_points=np.array([point]))
interpolator = interpolation.Interpolator3D(cap_set)
"""
interpolator.interpolate(inset, point, knn=2, flow=True)
interpolator.visualize_all(p)
#interpolator.show_point_influences(400, 200, show_inset=True, sphere=True, best_arrows=True)

"""
#synthesize 10 points on a line between point i[0] and i[2]
#for dist in np.round(np.linspace(0,1,11),2):
for dist in np.round(np.linspace(0,1,6),2):
    D_pos = utils.get_point_on_plane(caps[i[0]].pos, caps[i[1]].pos, caps[i[2]].pos, dist1=0, dist2=dist)
#    inset = list(range(i)) + list(range(i+1,cap_set.get_size()))
#    inset = [i[0], i[2]]
#    inset = list(range(cap_set.get_size()))
    inset = [15,16,17,22,24,29,30,31]
    interpolator.interpolate(inset, D_pos, knn=2, flow=True)
    interpolator.visualize_all(dist)
    '''
    #write out results as cubemaps
    flowout = interpolator.flow_blend
    regout = interpolator.reg_blend
    utils.cvwrite(EnvironmentMap(flowout, 'latlong').convertTo('cube').data, "out_flow"+str(dist)+".jpg")
    utils.cvwrite(EnvironmentMap(regout, 'latlong').convertTo('cube').data, "out_reg_"+str(dist)+".jpg")
    '''

"""
#synthesize each existing viewpoint from all N-1 points (excluding point to be synthesized
for i in range(cap_set.get_size()):
#for i in [0, 10, 22, 50]:
#    D_pos = utils.get_point_on_plane(caps[i[0]].pos, caps[i[1]].pos, caps[i[2]].pos, dist1=0.0, dist2=dist)
    D_pos = cap_set.get_position(i)
    inset = list(range(i)) + list(range(i+1,cap_set.get_size()))
    out = interpolator.interpolate(inset, D_pos, knn=2)
    utils.cvwrite(out, "out_" + str(i) + ".jpg")
#    avg = interpolator.visualize_best_deviation()
    avg = interpolator.visualize_best_deviation(utils.OUT+'dev_angles_'+str(i)+".jpg")
"""
"""

#reproject input points to target point (1 to 1) without blending to debug reprojection
point = cap_set.get_position(17)
for viewpointnum in range(cap_set.get_size()):
    out = interpolator.interpolate([viewpointnum], point)
    utils.cvwrite(out, "reproj_" + str(point) + "_from_" + str(cap_set.get_position(viewpointnum)) + ".jpg")

#reproject input point to target points(1 to 1) without blending to debug reprojection
for viewpointnum in range(cap_set.get_size()):
    point = cap_set.get_position(viewpointnum)
    out = interpolator.interpolate([0], point)
#    utils.cvwrite(out, str(viewpointnum) + "reproj_" + str(point) + "_from_" + str(cap_set.get_position(0)) + ".jpg")

"""
