import cv2
import numpy as np

import utils
from interpolation import Interpolator
from cubemapping import ExtendedCubeMap
from optical_flow import farneback_of


#cube1 = ExtendedCubeMap("../../data/panos_rosalie/3/R0010043_20200503102723.JPG", "latlong")
#cube2 = ExtendedCubeMap("../../data/panos_rosalie/3/R0010042_20200503102620.JPG", "latlong")
#cube1 = ExtendedCubeMap("../../data/1D_testsets/01_exterior_stadium/03.JPG", "latlong")
#cube2 = ExtendedCubeMap("../../data/1D_testsets/01_exterior_stadium/02.JPG", "latlong")
cube1 = ExtendedCubeMap("../../data/1D_testsets/02_meeting_room/20.jpg", "latlong")
cube2 = ExtendedCubeMap("../../data/1D_testsets/02_meeting_room/21.jpg", "latlong")
#cube1 = ExtendedCubeMap("../../data/panos_rosalie/3_reduced/01.JPG", "latlong")
#cube2 = ExtendedCubeMap("../../data/panos_rosalie/3_reduced/015.JPG", "latlong")

utils.cvwrite(cube1.calc_clipped_cube(), '01_imgA_original')
utils.cvwrite(cube1.get_Xcube(), '01_imgA_extended')
utils.cvwrite(cube2.calc_clipped_cube(), '03_imgB_original')
utils.cvwrite(cube2.get_Xcube(), '03_imgB_extended')

flow = cube1.optical_flow(cube2, farneback_of)
#flow = cube1.optical_flow_face("front", cube2, farneback_of)
interpolator = Interpolator("cube")
new = interpolator.flow(cube1, cube2, flow, 0.5)
#utils.cvshow(new.get_Xcube())
utils.cvwrite(new.calc_clipped_cube(), '02_imgInterpolated_0.5')

#utils.cvshow(cube1.calc_clipped_cube(), 'clipped_in')
#utils.cvshow(utils.build_cube(cube1.extended), 'clipped_extended')
#utils.cvshow(new.calc_clipped_cube(), 'clipped_out')

#for d in np.around(np.linspace(0,1,11), 1):
#    new = interpolator.flow(cube1, cube2, flow, d)
#    utils.cvwrite(new.calc_clipped_cube(), '02_imgInterpolated_' + str(d))

#utils.cvshow(cube1.extended["front"], '01_imgA')
#utils.cvshow(new.extended["front"], '02_imgInterpolated')
#utils.cvshow(cube2.extended["front"], '03_imgB')
"""
FACES = ['top', 'front', 'left', 'right', 'bottom', 'back']
FACES = ['top']
for face in FACES:
    print(face)
    face1 = cube1.extended[face]
    face2 = cube2.extended[face]
#face1 = cv2.cvtColor(cv2.imread("../../data/1D_testsets/01_exterior_stadium/front_A.jpg", 1), cv2.COLOR_BGR2RGB)
#face2 = cv2.cvtColor(cv2.imread("../../data/1D_testsets/01_exterior_stadium/front_B.jpg", 1), cv2.COLOR_BGR2RGB)
#    utils.cvshow(face1, "01_A" + face)
    flow, rgb = farneback_of(face1, face2)
    print("flow")
    utils.print_type(flow)
    utils.cvshow(rgb)
    interpolator = Interpolator("planar")
    new = interpolator.flow(face1, face2, flow, 0.5).astype(np.float32)
#    new = cv2.cvtColor(new.astype(np.float32), cv2.COLOR_BGR2GRAY)
    utils.print_type(new)
    utils.cvshow(new, "02_interpolated" + face)
#    utils.cvshow(face2, "03_B" + face)
"""

