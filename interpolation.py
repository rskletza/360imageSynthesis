import numpy as np
import cv2

import utils
from scipy.ndimage.interpolation import map_coordinates

#def shift_img(img, flow, alpha):
#    xx, yy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
#    xx_shifted = (xx - flow[:,:,0] * alpha).astype(np.uint8)
#    xx_shifted[xx_shifted >= img.shape[1]] = img.shape[1] - 1
#    yy_shifted = (yy - flow[:,:,1] * alpha).astype(np.uint8)
#    yy_shifted[yy_shifted >= img.shape[0]] = img.shape[0] - 1
#    print(xx.shape, xx)
#    img_shifted = np.zeros_like(img)
#    img_shifted = img[yy_shifted, xx_shifted, :]
#
#    return img_shifted

def shift_img(img, flow, alpha):
    xx, yy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))

    xx_shifted = (xx - (flow[:,:,0] * alpha)).astype(np.float32)
    yy_shifted = (yy - (flow[:,:,1] * alpha)).astype(np.float32)

#    shifted_img = cv2.remap(img, xx_shifted, yy_shifted, cv2.INTER_LINEAR)
    shifted_coords = np.array([yy_shifted.flatten(), xx_shifted.flatten()])
    print(img.shape)
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shifted_img = np.zeros_like(img)
    for d in range(img.shape[2]):
        shifted_img[:,:,d] = np.reshape(map_coordinates(img[:,:,d], shifted_coords), (img.shape[0], img.shape[1]))

    return shifted_img

types = ["cube", "planar"]
class Interpolator:
    def __init__(self, type):
        self.type = type
        pass

    def linear(self, A, B, alpha):
        out = (1-alpha) * A.get_Xcube() + alpha * B.get_Xcube()
        return out

    def flow(self, A, B, flow, alpha):
#        flow = np.full_like(flow, 20)
#        flow[:,:,0] = -10
        if self.type is "cube":
#            front = utils.split_cube(A.get_Xcube())["front"]
#            front_flow = utils.split_cube(flow)["front"]
#            middle = int(front_flow.shape[0]/2)
#            shifted_front = shift_img(front, -front_flow, alpha)
#            utils.cvshow(shifted_front)
            imgA = A.get_Xcube()
            imgB = B.get_Xcube()
        elif self.type is "planar":
            imgA = A
            imgB = B
        else:
            raise NotImplementedError

        shifted_A = shift_img(imgA, flow, alpha)
        shifted_B = shift_img(imgB, -flow, (1-alpha))


        out = alpha * shifted_A + (1-alpha) * shifted_B
        return out

        
