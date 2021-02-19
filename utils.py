"""
A collection of useful functions
"""

import cv2
import numpy as np
import json
import skimage
from skimage import io, transform
import matplotlib.pyplot as plt

from envmap import EnvironmentMap

#modify this to the desired output path
OUT = "./"

FACES = ['top', 'front', 'left', 'right', 'bottom', 'back']
DPI = 300

def print_type(array):
    """
    Prints the min and max values of an array and its shape
    """
    print("min: " , np.amin(array), ", max: ", np.amax(array))
    print("shape: ", array.shape, ", type: ", array.dtype)

def cvshow(img, filename=None):
    """
    Displays an image
    """
    io.imshow(img)
    io.show()

def cvwrite(img, filename=None, path=OUT):
    """
    Saves an image at the specified location
    """
    if filename is None:
        fullpath = path
    else:
        fullpath = path + filename
    img = np.clip(img, -1, 1)
    img = skimage.img_as_ubyte(img)
    io.imsave(fullpath, img)

def split_cube(cube):
    """
    Splits an image in cubemap representation into a python dict with the face names corresponding to the face images. Reverse of build_cube
    """
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
    """
    Builds an image in cubemap representation from a python dict created using split_cube
    """
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

def build_sideways_cube(faces):
    """
    Builds an image in sideways cubemap representation from a python dict created using split_cube (only for visualization!)
            top
             -
    left - center - right - back
             -
            bottom
    """
    w = faces["top"].shape[0] #width of a single face
    if len(faces["top"].shape) is 3:
        cube = np.ones((w*3, w*4, faces["top"].shape[2]))
        cube[0:w, w:w*2, :]     = faces["top"]
        cube[w:2*w, 0:w, :]     = faces["left"]
        cube[w:2*w, w:2*w, :]   = faces["front"]
        cube[w:2*w, 2*w:3*w, :] = faces["right"]
        cube[2*w:3*w, w:2*w, :] = faces["bottom"]
        cube[w:2*w, 3*w:4*w, :] = transform.rotate(faces["back"], 180)
    else:
        cube = np.ones_like((w*4, w*3))
        cube[0:w, w:w*2]     = faces["top"]
        cube[w:2*w, 0:w]     = faces["left"]
        cube[w:2*w, w:2*w]   = faces["front"]
        cube[w:2*w, 2*w:3*w]    = faces["right"]
        cube[2*w:3*w, w:2*w] = faces["bottom"]
        cube[w:2*w, 3*w:4*w] = transform.rotate(faces["back"], 180)
    return cube

def build_cube_strip(faces):
    """
    builds a strip of the cube: left - center - right - back
    for space-conserving visualization
    """
    w = faces["top"].shape[0] #width of a single face
    if len(faces["top"].shape) is 3:
        strip = np.zeros((w, w*4, faces["top"].shape[2]))
        strip[0:w, 0:w, :]     = faces["left"]
        strip[0:w, w:2*w, :]   = faces["front"]
        strip[0:w, 2*w:3*w, :] = faces["right"]
        strip[0:w, 3*w:4*w, :] = transform.rotate(faces["back"], 180)
    else:
        strip = np.zeros_like((w, w*4))
        strip[0:w, 0:w]     = faces["left"]
        strip[0:w, w:2*w]   = faces["front"]
        strip[0:w, 2*w:3*w] = faces["right"]
        strip[0:w, 3*w:4*w] = transform.rotate(faces["back"], 180)
    return strip

def build_cube_strip_with_bottom(faces):
    """
    builds a strip of the cube: left - center - right - back
                                          -
                                       bottom
    for space-conserving visualization
    """
    w = faces["top"].shape[0] #width of a single face
    if len(faces["top"].shape) is 3:
        strip = np.ones((w*2, w*4, faces["top"].shape[2]))
        strip[0:w, 0:w, :]     = faces["left"]
        strip[0:w, w:2*w, :]   = faces["front"]
        strip[0:w, 2*w:3*w, :] = faces["right"]
        strip[0:w, 3*w:4*w, :] = transform.rotate(faces["back"], 180)
        strip[w:2*w, w:2*w, :] = faces["bottom"]
    else:
        strip = np.ones_like((w*2, w*4))
        strip[0:w, 0:w]     = faces["left"]
        strip[0:w, w:2*w]   = faces["front"]
        strip[0:w, 2*w:3*w] = faces["right"]
        strip[0:w, 3*w:4*w] = transform.rotate(faces["back"], 180)
        strip[w:2*w, w:2*w] = faces["bottom"]
    return strip

def build_params(p=0.5, l=5, w=13, i=15, poly_expansion=7, sd=1.5, path=".", store=True):
    """
    Stores parameters for optical flow calculation
    """
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
    """
    Loads parameters for optical flow calculation; if none exist, uses default parameters given in build_params
    """
    try:
        with open(path + '/ofparams.json', 'r') as json_file:
            params = json.load(json_file)
    except FileNotFoundError:
        params = build_params(store=False)
    return params;

def sample_points(points, numpoints=200):
    """
    Samples points from a uniform distribution

    points: a 3d array, from which the samples are taken uniformly from the width and height
    numppoints: the number of points to sample
    """
    if len(points.shape) == 1:
        points = points[:,np.newaxis]

    if (points.shape[0] * points.shape[1]) > numpoints:
        #find the number of points to keep for width and height so that the total points will be numpoints
        height = int(np.sqrt(numpoints/2))
        dist_h = int(points.shape[0]/height)
        width = 2 * height
        dist_w = int(points.shape[1]/width)
        #slice the points
        return points[0:-1:dist_h, 0:-1:dist_w, :]
    else:
        return points

def is_cubemap(image):
    """
    Very rudimentary way to check if an image is in cubemap representation
    """
    return (image.shape[0] / 4 == image.shape[1] / 3)

def load_img(path):
    '''
    Loads an image at the specified path in rgb format in the range 0,1 in float32
    '''
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError("image at " + path + " not found")
    return (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255).astype(np.float32)

def latlong2cube(latlong):
    '''
    Converts an image in latlong format to an image in cubemap format
    '''
    return EnvironmentMap(latlong, "latlong").convertTo("cube").data
