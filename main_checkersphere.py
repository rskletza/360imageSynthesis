import numpy as np
import string

from capture import Capture, CaptureSet
import utils, interpolation, optical_flow
from envmap import EnvironmentMap
from preproc import parse_metadata

#cap_set = CaptureSet("../../data/captures/thesis_selection/checkersphere_6vps_250x500/", radius=2.1213)#, blenderfile="checkersphere.blend")
cap_set = CaptureSet("../../data/captures/thesis_selection/square_synth_room/6x6/", radius=2.47, in_place=True)#, blenderfile="checkersphere.blend")

s_points, _ = parse_metadata(cap_set.location + 'gt_metadata.txt')

cap_set.draw_scene(s_points=s_points, twoD=True)

print(cap_set.positions)
print(getout)

#s_points = np.array([
#    [-1, -1, 0], [-1, -0.5, 0], [-1, 0, 0], [-1, 0.5, 0], [-1, 1, 0],
#    [-0.5, -1, 0], [-0.5, -0.5, 0], [-0.5, 0, 0], [-0.5, 0.5, 0], [-0.5, 1, 0],
#    [0, -1, 0], [0, -0.5, 0], [0, 0, 0], [0, 0.5, 0], [0, 1, 0],
#    [0.5, -1, 0], [0.5, -0.5, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0.5, 1, 0],
#    [1, -1, 0], [1, -0.5, 0], [1, 0, 0], [1, 0.5, 0], [1, 1, 0],
#]) #gt points

ids = list(string.ascii_uppercase)

cap_set.draw_scene(s_points=s_points, twoD=True, saveas='checkersphere_scene.jpg')

interpolator = interpolation.Interpolator3D(cap_set)

def interpolate_maxmin(i):
    s_point = s_points[i]

    #with max viewpoints
    inset = cap_set.get_vps_in_radius(s_point, radius=(cap_set.radius/2))
    interpolator.interpolate(inset, s_point, knn=2, blend=False)
    interpolator.blend_image(knn=2)
    interpolator.visualize_interpolation(str(ids[i])+"_max_vps", type="reg")

    inset = cap_set.get_closest_vps(s_point, n=4)
    interpolator.interpolate(inset, s_point, blend=False, flow=True)
    interpolator.flow_blend_image()
    interpolator.visualize_interpolation(str(ids[i])+"_max_vps", type="flow")

    #with min viewpoints
    inset = cap_set.get_closest_vps(s_point, 2)
    interpolator.interpolate(inset, s_point, knn=2, blend=True, flow=True)
    interpolator.visualize_interpolation(str(ids[i])+"_min_vps")

for i in range(0,s_points.shape[0]):
    interpolate_maxmin(i)
