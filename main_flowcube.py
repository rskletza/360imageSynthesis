import cv2
import numpy as np

import utils
from cubemapping import ExtendedCubeMap
from optical_flow import optical_flow

#cube1 = ExtendedCubeMap("../../data/panos_rosalie/3_reduced/01.JPG", "latlong", 0.2)
#cube1 = ExtendedCubeMap("../../data/panos_rosalie/3/R0010041_20200503102518.JPG", "latlong", 0.2)

#test = ExtendedCubeMap("../../data/panos_rosalie/3/R0010041_20200503102518.JPG", "latlong", 0.2)
#test = ExtendedCubeMap("../../data/panos_rosalie/3_reduced/01.JPG", "latlong", 0.2)
test = ExtendedCubeMap("../../data/test/face_test_s.jpg", "cube", 0.2)
#print(test.data.shape)

#flow
#cube1 = ExtendedCubeMap("../../data/panos_rosalie/3_reduced/01.JPG", "latlong")
#cube2 = ExtendedCubeMap("../../data/panos_rosalie/3_reduced/02.JPG", "latlong")
cube1 = ExtendedCubeMap("../../data/test/0001.jpg", "latlong")
cube2 = ExtendedCubeMap("../../data/test/0002.jpg", "latlong")
flow = cube1.optical_flow(cube2, optical_flow)

hsv = np.zeros((flow.shape[0], flow.shape[1], 3))
hsv[...,1] = 255
mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
hsv[...,0] = ang*180/np.pi/2
hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
rgb = cv2.cvtColor(hsv.astype(np.float32), cv2.COLOR_HSV2BGR)

utils.cvshow(rgb)
