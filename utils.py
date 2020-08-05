import cv2
import numpy as np
import json
import random
from skimage import io
from os import listdir

#from capture import Capture

def print_type(array):
    print("min: " , np.amin(array), ", max: ", np.amax(array))
    print("shape: ", array.shape, ", type: ", array.dtype)

def cvshow(img, filename=None):
    if(np.floor(np.amax(img)) > 1):
        img = img.astype(np.uint8)

    if img.dtype is (np.dtype('float64') or np.dtype('float32')):
        img = (img * 255).astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#    img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB)
    cv2.imshow(filename, img)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s') and filename is not None:
#        if(np.amax(img) <= 1):
#            img = img*255
#        print(np.amax(img), img.dtype)
        cv2.imwrite("../../data/out/" + filename + ".jpg", img)
    cv2.destroyAllWindows()

def cvwrite(img, filename):
    if(np.floor(np.amax(img)) > 1):
        img = img.astype(np.uint8)

    if img.dtype is (np.dtype('float64') or np.dtype('float32')):
        img = (img * 255).astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("../../data/out/" + filename + ".jpg", img)

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

def build_params(p=0.5, l=5, w=13, i=15, poly_expansion=7, sd=1.5, path=".", store=True):
    params = {
        "pyr_scale": p,
        "levels": l,
        "winsize": w,
        "iters": i,
        "poly_expansion": poly_expansion,
        "sd": sd
    }
    
    if store:
        with open(path + '/ofparams.json', 'w', encoding='utf-8') as json_file:
            json.dump(params, json_file, ensure_ascii=False, indent=4)

    return params

def load_params(path='.'):
    try:
        with open(path + '/ofparams.json', 'r') as json_file:
            params = json.load(json_file)
    except FileNotFoundError:
        params = build_params(store=False)
    return params;

def parse_metadata(txt):
    """
    takes a text file as an input containing rotation and position metadata and returns the rotations and positions as np arrays
    input must have rotation as quaternion and position for each image on a separate line, in order, separated by commas
    the x/z plane is parallel to the floor
    """
    rotations = []
    positions = []
    with open(txt, 'r') as file:
        lines = file.readlines() #TODO why caps
        for line in lines:
            strs = line.split(',')
            if len(strs) is not 7:
                raise Exception("One or more of the lines in the input file have the wrong number of values")

            rot = np.array([strs[0], strs[1], strs[2], strs[3]]).astype(float)
            rotations.append(rot)

            pos = np.array([strs[4], strs[5], strs[6]]).astype(float)
            positions.append(pos)

    return np.array(positions), np.array(rotations)

def get_point_on_plane(A, B, C, dist1=None, dist2=None):
    """
    calculates a (random) point D on the plane spanned by A, B, C
    if dist1 and dist2 are None, a random value is used
    dist1 determines the distance of point AB along the vector between A and B
    dist2 determines the distance of point D along the vector between AB and C

    A    
    |       
    AB----D---C
    |
    |
    |
    B
    """
    random.seed()
    if dist1 is None:
        dist1 = random.random()
    if dist2 is None:
        dist2 = random.random()
    print(dist1, dist2)

    AB = A + dist1 * (B - A) 
    return AB + dist2 * (C - AB)

def center(points):
    """
    takes a point cloud of numpy arrays
    returns same point cloud centered around 0,0,0
    """
#   normalize
#    minima = np.amin(points)
#    points -= minima
#
#    maxima = np.amax(points)
#    points = ( points/maxima ) * suggested_diameter

    minima = np.amin(points, axis=0)
    maxima = np.amax(points, axis=0)
    center_point = minima + (maxima-minima) * 0.5
    shifted_points = points - center_point

    return shifted_points
