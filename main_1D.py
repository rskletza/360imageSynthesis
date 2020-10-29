import cv2
import numpy as np

import utils, optical_flow
from envmap import EnvironmentMap
from interpolation import Interpolator1D, invert_flow
from cubemapping import ExtendedCubeMap

'''
### planar interpolation ###
imgAfront = cv2.cvtColor(cv2.imread("../../data/1D_testsets/01_exterior_stadium/front_A.jpg", 1), cv2.COLOR_BGR2RGB)
imgBfront = cv2.cvtColor(cv2.imread("../../data/1D_testsets/01_exterior_stadium/front_B.jpg", 1), cv2.COLOR_BGR2RGB)
interpolator_p = Interpolator1D(imgAfront, imgBfront, "planar")
out_p = interpolator_p.interpolate(0.5)
flow_p = interpolator_p.get_flow_visualization()
utils.cvshow(out_p, 'planar_front.jpg')
utils.cvshow(flow_p, 'planar_front_flow.jpg')
'''

### panoramic interpolation ###
#imgA = cv2.cvtColor(cv2.imread("../../data/1D_testsets/01_exterior_stadium/03.JPG", 1), cv2.COLOR_BGR2RGB)
#imgB = cv2.cvtColor(cv2.imread("../../data/1D_testsets/01_exterior_stadium/02.JPG", 1), cv2.COLOR_BGR2RGB)

#imgA = cv2.cvtColor(cv2.imread("../../data/captures/meetingRoom_360/normalized/images/4.jpg", 1), cv2.COLOR_BGR2RGB)
#imgB = cv2.cvtColor(cv2.imread("../../data/captures/meetingRoom_360/normalized/images/6.jpg", 1), cv2.COLOR_BGR2RGB)

#imgA = cv2.cvtColor(cv2.imread("../../data/captures/synthesized_room/square_room_0.5res/images/0.jpg", 1), cv2.COLOR_BGR2RGB)
#imgB = cv2.cvtColor(cv2.imread("../../data/captures/synthesized_room/square_room_0.5res/images/1.jpg", 1), cv2.COLOR_BGR2RGB)

#imgA = cv2.cvtColor(cv2.imread("../../data/1D_testsets/03_virtual_room/15_high_res.jpg", 1), cv2.COLOR_BGR2RGB)
#imgB = cv2.cvtColor(cv2.imread("../../data/1D_testsets/03_virtual_room/16_high_res.jpg", 1), cv2.COLOR_BGR2RGB)

#imgA = cv2.cvtColor(cv2.imread("../../data/1D_testsets/03_virtual_room/0055.jpg", 1), cv2.COLOR_BGR2RGB)
#imgB = cv2.cvtColor(cv2.imread("../../data/1D_testsets/03_virtual_room/0058.jpg", 1), cv2.COLOR_BGR2RGB)

indices = [16,30]
imgA = cv2.cvtColor(cv2.imread("../../data/captures/synthesized_room/square_room_textured_brick/images/"+str(indices[0])+".jpg", 1), cv2.COLOR_BGR2RGB)
imgB = cv2.cvtColor(cv2.imread("../../data/captures/synthesized_room/square_room_textured_brick/images/"+str(indices[1])+".jpg", 1), cv2.COLOR_BGR2RGB)
path = "../../data/captures/synthesized_room/square_room_textured_brick/optical_flow/"+str(indices[0])+"-"+str(indices[1])+".npy"
path_inverse = "../../data/captures/synthesized_room/square_room_textured_brick/optical_flow/"+str(indices[1])+"-"+str(indices[0])+".npy"
with open(path, 'rb') as f:
    flow = np.load(f)
with open(path_inverse, 'rb') as f:
    inverse_flow = np.load(f)

interpolator = Interpolator1D(imgA, imgB, "latlong", flow=(flow, inverse_flow))

'''set optical flow parameters, if necessary (when not using precalculated flow)'''
#utils.build_params(p=0.5, l=5, w=20, i=20, path="../../data/1D_testsets/03_virtual_room")
#interpolator = Interpolator1D(test, imgB, "latlong", param_path="../../data/1D_testsets/03_virtual_room")

#utils.cvwrite(interpolator.A.extended["back"], "test_original_back.jpg")
#visualize the flow
flowcube, inv_flowcube, flow_arrows, inv_flow_arrows = interpolator.get_flow_visualization()
utils.cvwrite(flowcube, 'flow_cube.jpg')
utils.cvwrite(inv_flowcube, 'flow_cube_inverse.jpg')
utils.cvwrite(flow_arrows, 'flow_cube_arrows.jpg')
utils.cvwrite(inv_flow_arrows, 'flow_cube_inverse_arrows.jpg')

''' visualize the difference between the original cube and the extended cube '''
#utils.cvwrite(interpolator.A.calc_clipped_cube(), '01_imgA_original.jpg')
#utils.cvwrite(interpolator.A.get_Xcube(), 'extended_01_imgA.jpg')
#utils.cvwrite(interpolator.B.calc_clipped_cube(), '03_imgB_original.jpg')
#utils.cvwrite(interpolator.B.get_Xcube(), 'extended_03_imgB.jpg')

'''interpolate a specific position'''
#out = interpolator.interpolate(0.5)
#utils.cvwrite(out, '02_imgInterpolated_clipped_0.5.jpg')
#utils.cvwrite(interpolator.out.get_Xcube(), 'extended_02_imgInterpolated_0.5.jpg')

'''interpolate on the line between the two viewpoints'''
for d in np.around(np.linspace(0,1,11), 1):
    out = interpolator.interpolate(d)
#    out = EnvironmentMap(out, "cube").convertTo("latlong").data
    utils.cvwrite(out, '02_imgInterpolated_' + str(d) + '.jpg')

#visualize the difference between the original (clipped_in), the extended, and the clipped cube (At the moment clipped_in (original) and clipped_out (output of calc_clipped_cube) are not identical. This needs to be fixed)
#utils.cvshow(interpolator.A.calc_clipped_cube(), 'clipped_in')
#utils.cvshow(utils.build_cube(interpolator.A.extended), 'clipped_extended')
#utils.cvshow(interpolator.out.calc_clipped_cube(), 'clipped_out')

