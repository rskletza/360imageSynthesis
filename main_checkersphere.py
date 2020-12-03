import numpy as np
import string

from capture import Capture, CaptureSet
import utils, interpolation, optical_flow
from envmap import EnvironmentMap

cap_set = CaptureSet("../../data/captures/evaluation/checkersphere250x500/", blenderfile="checkersphere.blend", radius=2.1213)

s_points = np.array([
    [-1, -1, 0], [-1, -0.5, 0], [-1, 0, 0], [-1, 0.5, 0], [-1, 1, 0],
    [-0.5, -1, 0], [-0.5, -0.5, 0], [-0.5, 0, 0], [-0.5, 0.5, 0], [-0.5, 1, 0],
    [0, -1, 0], [0, -0.5, 0], [0, 0, 0], [0, 0.5, 0], [0, 1, 0],
    [0.5, -1, 0], [0.5, -0.5, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0.5, 1, 0],
    [1, -1, 0], [1, -0.5, 0], [1, 0, 0], [1, 0.5, 0], [1, 1, 0],
]) #gt points

ids = list(string.ascii_uppercase)

cap_set.draw_scene(s_points=s_points, twoD=True, saveas='checkersphere_scene.jpg')

interpolator = interpolation.Interpolator3D(cap_set)

for i in range(1,s_points.shape[0]):
    s_point = s_points[i]
    inset = cap_set.get_vps_in_radius(s_point, radius=(cap_set.radius/2))
#    inset = cap_set.get_2_closest_vps(s_point)

    out = interpolator.interpolate(inset, s_point, knn=2, flow=True)
    interpolator.visualize_all(ids[i])
