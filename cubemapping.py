import cv2
import numpy as np

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

#TODO should go in utils or something
def cvshow(img, filename=None):
    img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB)
    cv2.imshow(filename, img)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s') and filename is not None:
        out = img*255
        cv2.imwrite("../../data/out/" + filename + ".jpg", out)
    cv2.destroyAllWindows()

def split_cube(cube):
    w = int(cube.shape[0]/4)
    faces = {}
    faces["top"]       = cube[0:w, w:w*2, :]
    faces["left"]      = cube[w:2*w, 0:w, :]
    faces["front"]     = cube[w:2*w, w:2*w, :]
    faces["right"]     = cube[w:2*w, 2*w:3*w]
    faces["bottom"]    = cube[2*w:3*w, w:2*w, :]
    faces["back"]      = cube[3*w:4*w, w:2*w, :]
    return faces

def build_cube(faces):
    w = faces["top"].shape[0] #width of a single face
    if len(faces["top"].shape) is 3:
        cube = np.zeros((w*4, w*3, faces["top"].shape[2]))
        cube[0:w, w:w*2, :]     = faces["top"]
        cube[w:2*w, 0:w, :]     = faces["left"]
        cube[w:2*w, w:2*w, :]   = faces["front"]
        cube[w:2*w, 2*w:3*w, :] = faces["right"]
        cube[2*w:3*w, w:2*w, :] = faces["bottom"]
        cube[3*w:4*w, w:2*w, :] = faces["back"]
    else:
        cube = np.zeros_like((w*4, w*3))
        cube[0:w, w:w*2]     = faces["top"]
        cube[w:2*w, 0:w]     = faces["left"]
        cube[w:2*w, w:2*w]   = faces["front"]
        cube[w:2*w, 2*w:3*w]    = faces["right"]
        cube[2*w:3*w, w:2*w] = faces["bottom"]
        cube[3*w:4*w, w:2*w] = faces["back"]
    return cube

class ExtendedCubeMap:
    def __init__(self, imgpath, format, percent=1.2):
        self._envMap = EnvironmentMap(imgpath, format)

        self.w = int(self._envMap.data.shape[0]/4) #width of cube face
        print("width: ", self.w)
            
        #TODO link percent and fov, then pass percent
        self.extended = self.extend_projection(96)
        if(format is not 'cube'):
            self._envMap.convertTo('cube')
#
#        #extract the separate faces
#        self.faces = split_cube(self._envMap.data)
#        for face in FACES:
#            cvshow(self.extended[face], face)

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
        clips away the extension
        """
        flow = {}
        for face in FACES:
            extended_flow = flowfunc(self.extended[face], other.extended[face])
            #extract real faces from extended faces
            padding = int((extended_flow.shape[0] - self.w) / 2)
            flow[face] = extended_flow[padding:-padding, padding:-padding, :]
            print(flow[face].shape)

        flow_cube = build_cube(flow)
        return flow_cube


#    def get_neighbors(self, side):
#        if (side not in SIDES):
#            print("Incorrect side value: " + side + " . Value must be one of [top, left, front, right, bottom, back]" )
#            return
#        if (side == 'top'):
#            center = self.top
#            above = self.back
#            left = cv2.rotate(self.left, cv2.ROTATE_90_CLOCKWISE)
#            below = self.front
#            right = cv2.rotate(self.right, cv2.ROTATE_90_COUNTERCLOCKWISE)
#        elif (side == 'left'):
#            center = self.left
#            above = cv2.rotate(self.top, cv2.ROTATE_90_COUNTERCLOCKWISE)
#            left = cv2.rotate(self.back, cv2.ROTATE_180)
#            below = cv2.rotate(self.bottom, cv2.ROTATE_90_CLOCKWISE)
#            right = self.front
#        elif (side == 'front'):
#            center = self.front
#            above = self.top
#            left = self.left
#            below = self.bottom
#            right = self.right
#        elif (side == 'right'):
#            center = self.right
#            above = cv2.rotate(self.top, cv2.ROTATE_90_CLOCKWISE)
#            left = self.front
#            below = cv2.rotate(self.bottom, cv2.ROTATE_90_COUNTERCLOCKWISE)
#            right = cv2.rotate(self.back, cv2.ROTATE_180)
#        elif (side == 'bottom'):
#            center = self.bottom
#            above = self.front
#            left = cv2.rotate(self.left, cv2.ROTATE_90_COUNTERCLOCKWISE)
#            below = self.back
#            right = cv2.rotate(self.right, cv2.ROTATE_90_CLOCKWISE)
#        elif (side == 'back'):
#            center = self.back
#            above = self.bottom
#            left = cv2.rotate(self.left, cv2.ROTATE_180)
#            below = self.top
#            right = cv2.rotate(self.right, cv2.ROTATE_180)
##        return {'center':center, 'above':above, 'left':left, 'below':below, 'right':right}
#        return {'above':above, 'left':left, 'below':below, 'right':right}
