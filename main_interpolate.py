import cv2
import numpy as np

import utils
from interpolation import Interpolator
from cubemapping import ExtendedCubeMap
from optical_flow import optical_flow


#cube1 = ExtendedCubeMap("../../data/test/0001.jpg", "latlong")
#cube2 = ExtendedCubeMap("../../data/test/0002.jpg", "latlong")
#cube1 = ExtendedCubeMap("../../data/panos_rosalie/3/R0010043_20200503102723.JPG", "latlong")
#cube2 = ExtendedCubeMap("../../data/panos_rosalie/3/R0010042_2020050310262.JPG", "latlong")
cube1 = ExtendedCubeMap("../../data/panos_rosalie/3_reduced/03.JPG", "latlong")
cube2 = ExtendedCubeMap("../../data/panos_rosalie/3_reduced/02.JPG", "latlong")
#cube1 = ExtendedCubeMap("../../data/panos_rosalie/3_reduced/01.JPG", "latlong")
#cube2 = ExtendedCubeMap("../../data/panos_rosalie/3_reduced/015.JPG", "latlong")

flow = cube1.optical_flow(cube2, optical_flow)
interpolator = Interpolator("cube")
new = ExtendedCubeMap(interpolator.flow(cube1, cube2, flow, 0.5), "Xcube")
#utils.cvshow(cube1.extended["front"], '01_imgA')
#utils.cvshow(new.extended["front"], '02_imgInterpolated')
#utils.cvshow(cube2.extended["front"], '03_imgB')
utils.cvshow(cube1.get_Xcube(), '01_imgA')
utils.cvshow(new.get_Xcube(), '02_imgInterpolated')
utils.cvshow(cube2.get_Xcube(), '03_imgB')
"""

face2 = cv2.cvtColor(cv2.imread("../../data/out/03_originalA.jpg", 1), cv2.COLOR_BGR2RGB)
face1 = cv2.cvtColor(cv2.imread("../../data/out/01_ground_truthB.jpg", 1), cv2.COLOR_BGR2RGB)
flow = optical_flow(face1, face2)
interpolator = Interpolator("planar")
new = interpolator.flow(face1, face2, flow, 0.5)
utils.cvshow(face1, "01_A")
utils.cvshow(new, "02_interpolated")
utils.cvshow(face2, "03_B")
"""

