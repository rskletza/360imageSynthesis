import cv2
import numpy as np

import utils
from synthesis import Interpolator1DoF

'''
### planar interpolation ###
imgA = utils.load_img("./testdata/1DoF_testdata/front_A.jpg")
imgB = utils.load_img("./testdata/1DoF_testdata/front_B.jpg")
interpolator_p = Interpolator1DoF(imgA, imgB, "planar")
out_planar = interpolator_p.interpolate(0.5)
flow_cube, _ = interpolator_p.get_flow_visualization()
utils.cvshow(out_planar, 'planar_front.jpg')
utils.cvshow(flow_cube, 'planar_front_flow.jpg')
'''

### panoramic interpolation ###
imgA = utils.load_img("./testdata/VirtualRoom_CaptureSet/images/0.jpg")
imgB = utils.load_img("./testdata/VirtualRoom_CaptureSet/images/1.jpg")

interpolator = Interpolator1DoF(imgA, imgB, "latlong")

'''set optical flow parameters, if necessary (when not using precalculated flow)'''
#utils.build_params(p=0.5, l=5, w=20, i=20, path="./")
#interpolator = Interpolator1DoF(imgA, imgB, "latlong", param_path="./")

#visualize the flow
flowcube, flow_arrows= interpolator.get_flow_visualization()
utils.cvwrite(flowcube, 'flow_cube.jpg')
utils.cvwrite(flow_arrows, 'flow_cube_arrows.jpg')

''' visualize the difference between the original cube and the extended cube '''
#utils.cvwrite(interpolator.A.calc_clipped_cube(), '01_imgA_original.jpg')
#utils.cvwrite(interpolator.A.get_Xcube(), 'extended_01_imgA.jpg')
#utils.cvwrite(interpolator.B.calc_clipped_cube(), '03_imgB_original.jpg')
#utils.cvwrite(interpolator.B.get_Xcube(), 'extended_03_imgB.jpg')

'''interpolate a specific position'''
out = interpolator.interpolate(0.5)
utils.cvwrite(out, '02_imgInterpolated_clipped_0.5.jpg')
#utils.cvwrite(interpolator.out.get_Xcube(), 'extended_02_imgInterpolated_0.5.jpg')

'''interpolate on the line between the two viewpoints'''
for d in np.around(np.linspace(0,1,11), 1):
    out = interpolator.interpolate(d)
    utils.cvwrite(out, '02_imgInterpolated_' + str(d) + '.jpg')
