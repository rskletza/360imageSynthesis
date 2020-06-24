import cv2
import numpy as np

import utils
from interpolation import Interpolator
from cubemapping import ExtendedCubeMap
from optical_flow import farneback_of


#cube1 = ExtendedCubeMap("../../data/test/0001.jpg", "latlong")
#cube2 = ExtendedCubeMap("../../data/test/0002.jpg", "latlong")
#cube1 = ExtendedCubeMap("../../data/panos_rosalie/3/R0010043_20200503102723.JPG", "latlong")
#cube2 = ExtendedCubeMap("../../data/panos_rosalie/3/R0010042_20200503102620.JPG", "latlong")
cube1 = ExtendedCubeMap("../../data/panos_rosalie/3_reduced/03.JPG", "latlong")
cube2 = ExtendedCubeMap("../../data/panos_rosalie/3_reduced/02.JPG", "latlong")
#cube1 = ExtendedCubeMap("../../data/panos_rosalie/3_reduced/01.JPG", "latlong")
#cube2 = ExtendedCubeMap("../../data/panos_rosalie/3_reduced/015.JPG", "latlong")
#utils.cvwrite(cube1.calc_clipped_cube(), '01_imgA')
#utils.cvwrite(cube2.calc_clipped_cube(), '03_imgB')

flow = cube1.optical_flow(cube2, farneback_of)
interpolator = Interpolator("cube")
new = interpolator.flow(cube1, cube2, flow, 0)
utils.cvshow(cube1.calc_clipped_cube(), 'clipped_in')
utils.cvshow(utils.build_cube(cube1.extended), 'clipped_extended')
utils.cvshow(new.calc_clipped_cube(), 'clipped_out')

#for d in np.around(np.linspace(0,1,11), 1):
#    new = interpolator.flow(cube1, cube2, flow, d)
#    utils.cvwrite(new.calc_clipped_cube(), '02_imgInterpolated_' + str(d))

#utils.cvshow(cube1.extended["front"], '01_imgA')
#utils.cvshow(new.extended["front"], '02_imgInterpolated')
#utils.cvshow(cube2.extended["front"], '03_imgB')
"""

face2 = cv2.cvtColor(cv2.imread("../../data/out/good/03_originalA.jpg", 1), cv2.COLOR_BGR2RGB)
face1 = cv2.cvtColor(cv2.imread("../../data/out/good/01_ground_truthB.jpg", 1), cv2.COLOR_BGR2RGB)
flow = optical_flow(face1, face2)
interpolator = Interpolator("planar")
new = interpolator.flow(face1, face2, flow, 0.5)
utils.cvshow(face1, "01_A")
utils.cvshow(new, "02_interpolated")
utils.cvshow(face2, "03_B")
"""

