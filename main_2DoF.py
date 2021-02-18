import numpy as np
import string

from capture import Capture, CaptureSet
from preproc import parse_metadata
import utils, synthesis, optical_flow
from envmap import EnvironmentMap

#load the capture set
cap_set = CaptureSet("./testdata/VirtualRoom_CaptureSet/", radius=2.47, in_place=True)

############## choose the points to synthesize ############## 

#Option 1: parse the locations of the viewpoints to be synthesized (ground truth data exists for these points)
s_points, _ = parse_metadata(cap_set.location + 'gt_metadata.txt')

'''
#Option 2: synthesize 10 points on a line between point A and B
A = cap_set.get_position(1)
B = cap_set.get_position(18)
s_points = []
for dist in np.round(np.linspace(0,1,11),2):
    D_pos = A + dist * (B - A)
    s_points.append(D_pos)
s_points = np.array(s_points)
'''

#names for the synthesized viewpoints: if there are fewer than 26, use letters, otherwise use numbers
if len(s_points) <= 26:
    ids = list(string.ascii_uppercase)
else:
    ids = list(range(len(s_points)))

############## display or save the scene visualization: contains the synthesized and the captured viewpoints ############## 

#display the points
cap_set.draw_spoints(s_points, ids, slabel=True, ilabel=False, sphere=True)
#or save the point visualization
#cap_set.draw_spoints(s_points, ids, slabel=True, ilabel=False, sphere=True, saveas=utils.OUT + "scene.jpg")

############## synthesis of the chosen points ############## 

synthesizer = synthesis.Synthesizer2DoF(cap_set)

for i in range(s_points.shape[0]):
    print("id " + ids[i])
    s_point = s_points[i]

    #get the closest four captured viewpoints to use as input
    inset = cap_set.get_closest_vps(s_point, n=4)
    print(inset)

    #synthesis with regular blending
    synthesizer.synthesize(inset, s_point, flow=False, visualize=(False, ids[i], utils.OUT))

    #synthesis with flow-based blending
    synthesizer.synthesize(inset, s_point, flow=True, flow_precision=1, visualize=(False, ids[i], utils.OUT))

