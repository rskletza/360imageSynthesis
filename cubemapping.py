import cv2
import numpy as np

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
FACES = ['top', 'front', 'left', 'right', 'bottom', 'back']

class ExtendedCubeMap:
    def __init__(self, imgpath, format, percent=1.2):

        #Xcube -> already an extended cube
        if format is "Xcube":
            self._envMap = EnvironmentMap(imgpath, "cube")
            self.w = int(self._envMap.data.shape[0]/4) #width of cube face
            self.extended = utils.split_cube(imgpath)
        else:
            self._envMap = EnvironmentMap(imgpath, format)
            self.w = int(self._envMap.data.shape[0]/4) #width of cube face

            #TODO link percent and fov, then pass percent
            self.extended = self.extend_projection(96)
            if(format is not 'cube'):
                self._envMap.convertTo('cube')

#        for face in FACES:
#            utils.cvshow(self.extended[face])
#        utils.cvshow(self.get_Xcube())

    def get_Xcube(self):
#        return self._envMap.data
        return utils.build_cube(self.extended)

    def get_cube(self):
        pass


    def extend_projection(self, fov):
        """
        fov: field of view to project with, should be > 90
        calculates a projection for each face of the cube with extra padding (doubled edge)
        """
        face_width = int(self.w * 1.5) #TODO make dependent on fov

        rotations = {   "top": rotation_matrix(0, np.deg2rad(-90), 0),
                        "front": rotation_matrix(0, 0, 0),
                        "left": rotation_matrix(np.deg2rad(-90), 0, 0),
                        "right": rotation_matrix(np.deg2rad(90), 0, 0),
                        "bottom": rotation_matrix(0, np.deg2rad(90), 0),
                        "back": rotation_matrix(0, np.deg2rad(180), 0)
                    }
        faces = {}
        for face in FACES:
            faces[face] = self._envMap.project(fov, rotations[face], 1., (face_width, face_width))
        return faces

    def optical_flow(self, other, flowfunc):
        """
        other: other ExtendedCubeMap for flow calculation
        flowfunc: optical flow function returning a 2D array of vectors
        """
        flow = {}
        bgr = {}
        for face in FACES:
            flow[face], bgr[face] = flowfunc(self.extended[face], other.extended[face])

        flow_cube = utils.build_cube(flow)
        bgr_cube = utils.build_cube(bgr)
#        utils.cvshow(bgr_cube, "flowcube")
        img = bgr_cube
        img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB)
        if(np.amax(img) > 1):
            out = img
        else:
            out = img*255
        cv2.imwrite("../../data/out/flow_cube.jpg", out)
        return flow_cube
