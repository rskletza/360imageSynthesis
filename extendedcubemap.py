import cv2
import numpy as np
from scipy.ndimage.interpolation import map_coordinates

import utils
from envmap import EnvironmentMap, rotation_matrix

"""
sides of the cube:
         ___
        | T |
     ___|___|___
    | L | F | R |
    |___|___|___|
        |BO |
        |___|
        |BA |
        |___|

T: top
L: left
F: front
R: right
BO: bottom
BA: back
"""

class ExtendedCubeMap:
    """
    Creates an extended cube: each face is extended so that points moving across edges can be tracked correctly by optical flow algorithms
    Input can either be a panorama in the format latlong or cube or a synthesized extended cube
    If it is a regular panorama, it is extended and the extended faces are stored
    If it is an extended cube, it is not extended but stored as is
    """
    def __init__(self, imgpath, format, fov=150, w_original=None):
        self.fov = fov
        self.format = format

        f = "cube" if format is "Xcube" else format

        self._envMap = EnvironmentMap(imgpath, f)
        self.w = int(self._envMap.data.shape[0]/4) #width of cube face

        #Xcube -> already an extended cube
        if format is "Xcube":
            self.extended = utils.split_cube(imgpath)
            self.w_original = w_original
        else:
            self.extended = self.extend_projection(self.fov)
            self.w_original = self.w
            self.w = self.extended["front"].shape[0]
            if(format is not 'cube'):
                self._envMap.convertTo('cube')

    def get_Xcube(self):
#        return self._envMap.data
        return utils.build_cube(self.extended)

    def calc_clipped_cube(self):
        """
        Calculates the original, non-extended cube
        """
        faces = {}
        border_width = int(np.floor((self.w - self.w_original)/2))

        xx, yy = np.meshgrid(np.arange(self.w_original), np.arange(self.w_original))
        clipped_coords = np.array([yy.flatten(), xx.flatten()])
        depth = self._envMap.data.shape[2]

        for face in utils.FACES:
            clipped = self.extended[face][border_width:-border_width, border_width:-border_width, :]
            faces[face] = np.zeros((self.w_original, self.w_original, depth))
            for d in range(depth):
                faces[face][:,:,d] = np.reshape(map_coordinates(clipped[:,:,d], clipped_coords), (self.w_original, self.w_original))
        return utils.build_cube(faces)

    def extend_projection(self, fov):
        """
        Calculates a projection for each face of the cube with the given field of view 
        CAUTION: due to a problem in the skylibs library (in the project function), there is a pixel offset
        """
        # adjacent is 1 (unit sphere) --> opposite is tan(fov)
        norm_original_width = np.tan(np.deg2rad(90/2))*2 #original fov is 90
#        print("norm original", norm_original_width)
        norm_new_width = np.tan(np.deg2rad(fov/2))*2
#        print("norm new", norm_new_width)

        face_width = int(self.w * (norm_new_width/norm_original_width))

        rotations = {   "top": rotation_matrix(0, np.deg2rad(-90), 0),
                        "front": rotation_matrix(0, 0, 0),
                        "left": rotation_matrix(np.deg2rad(-90), 0, 0),
                        "right": rotation_matrix(np.deg2rad(90), 0, 0),
                        "bottom": rotation_matrix(0, np.deg2rad(90), 0),
                        "back": rotation_matrix(0, np.deg2rad(180), 0)
                    }
        faces = {}
        for face in utils.FACES:
            faces[face] = self._envMap.project(fov, rotations[face], 1., (face_width, face_width))
        return faces

    def optical_flow(self, other, flowfunc, params):
        """
        Applies an optical flow algorithm on each face of the extended cube
        
        other: other ExtendedCubeMap for flow calculation
        flowfunc: optical flow function returning a 2D array of vectors
        """
        flow = {}
        for face in utils.FACES:
            flow[face] = flowfunc((self.extended[face]*255).astype(np.uint8), (other.extended[face]*255).astype(np.uint8), params)

        flow_cube = utils.build_cube(flow)
        return flow_cube

    def apply_facewise(self, func, operands, params):
        """
        Applies a function to each face of the ExtendedCubeMap individually

        function: function to use on faces
        operands: cubemap-shaped values to use in function (e.g. flowcube for shifting)
        params: further parameters to use in function
        """
        ops = utils.split_cube(operands)
        res = {}
        for face in utils.FACES:
            res[face] = func(self.extended[face], ops[face], params)
        res = utils.build_cube(res)
        return res
