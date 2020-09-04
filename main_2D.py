import numpy as np

from capture import Capture, CaptureSet, NormalizedCaptureSet
import utils, interpolation

#cap_set = CaptureSet("../../data/captures/meetingRoom_360/raw")
cap_set = CaptureSet("../../data/captures/synthesized_checkersphere/second", radius=12)
i = [6, 5, 4]
"""
cap_set = NormalizedCaptureSet("../../data/captures/meetingRoom_360/normalized")
i = [22, 23, 27]
"""

caps = cap_set.get_captures([i[0], i[1], i[2]])
#cap_set.draw_scene(indices=[i[0], i[1], i[2]])
#cap_set.draw_scene(indices=[xmax, xmin, ymax, ymin, zmax, zmin])
#cap_set.draw_scene()
#inset = list(range(1,cap_set.get_size()))
p = 1
point = cap_set.get_position(p)
inset = list(range(p)) + list(range(p+1,cap_set.get_size()))
#inset = list(range(p)) + list(range(p+1,10))
#inset = [8, 9, 34, 35, 31, 6, 5, 4, 0]
interpolator = interpolation.Interpolator3D(cap_set)
#out = interpolator.interpolate(inset, point, knn=3)
#utils.cvwrite(out, "out_" +str(p)+".jpg")
#avg = interpolator.visualize_best_deviation(utils.OUT+'dev_angles_'+str(p)+".jpg")
#avg = interpolator.visualize_best_deviation()
#h,w,_ = cap_set.get_capture(0).img.shape
#h = int(h/2)
#w = int(w/2)
#h = 400
#w = 200
#interpolator.show_point_influences(h, w, show_inset=True, sphere=False, best_arrows=True)

"""
#synthesize 10 points on a line between point i[0] and i[2]
for dist in np.round(np.linspace(0,1,11),2):
    D_pos = utils.get_point_on_plane(caps[i[0]].pos, caps[i[1]].pos, caps[i[2]].pos, dist1=0, dist2=dist)
#    inset = list(range(i)) + list(range(i+1,cap_set.get_size()))
#    inset = i
#    inset = [i[0], i[2]]
#    inset = list(range(cap_set.get_size()))
    out = interpolator.interpolate(inset, D_pos, knn=3)
    utils.cvwrite(out, "out_" +str(dist)+".jpg")

"""
#synthesize each existing viewpoint from all N-1 points (excluding point to be synthesized
for i in range(cap_set.get_size()):
#    D_pos = utils.get_point_on_plane(caps[i[0]].pos, caps[i[1]].pos, caps[i[2]].pos, dist1=0.0, dist2=dist)
    D_pos = cap_set.get_position(i)
    inset = list(range(i)) + list(range(i+1,cap_set.get_size()))
    out = interpolator.interpolate(inset, D_pos, knn=3)
    utils.cvwrite(out, "out_" + str(i) + ".jpg")
    avg = interpolator.visualize_best_deviation(utils.OUT+'dev_angles_'+str(i)+".jpg")
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
