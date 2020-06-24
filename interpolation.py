import numpy as np
import cv2

import utils

from cubemapping import ExtendedCubeMap

types = ["cube", "planar"]
class Interpolator:
    """
    creates an interpolator for either planar or cube (panoramic) images
    """
    def __init__(self, type):
        self.type = type

    def linear(self, A, B, alpha):
        out = (1-alpha) * self.get_image(A) + alpha * self.get_image(B)
        return out

    def flow(self, A, B, flow, alpha):
        shifted_A = utils.shift_img(self.get_image(A), flow, alpha)
        shifted_B = utils.shift_img(self.get_image(B), -flow, (1-alpha))

        out = (1 - alpha) * shifted_A + alpha * shifted_B

        if self.type is "cube":
            return ExtendedCubeMap(out, "Xcube", fov=A.fov, w_original=A.w_original)
        else:
            return out

    def get_image(self, A):
        if self.type is "cube":
            return  A.get_Xcube()
        elif self.type is "planar":
            return A
        else:
            raise NotImplementedError

        
