import cv2
import numpy as np
import json
import random
from skimage import io
import matplotlib.pyplot as plt

def print_type(array):
    print("min: " , np.amin(array), ", max: ", np.amax(array))
    print("shape: ", array.shape, ", type: ", array.dtype)

def cvshow(img, filename=None):
    if(np.floor(np.amax(img)) > 2):
        img = img.astype(np.uint8)

    if img.dtype is (np.dtype('float64') or np.dtype('float32')):
        img = (img * 255).astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width = img.shape[:2]
    cv2.namedWindow(filename, cv2.WINDOW_NORMAL)
    if height > 1080 or width > 1920:
        cv2.resizeWindow(filename, (width//2), (height//2))

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

def cvwrite(img, filename=None, path="../../data/out/"):
    if(np.floor(np.amax(img)) > 1):
        img = img.astype(np.uint8)

    if img.dtype is (np.dtype('float64') or np.dtype('float32')):
        img = (img * 255).astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if filename is None:
        fullpath = path
    else:
        fullpath = path + filename
    cv2.imwrite(fullpath, img)

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

    AB = A + dist1 * (B - A) 
    return AB + dist2 * (C - AB)

def plot(points, points2=None, points3=None):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.scatter(points[:,:,0], points[:,:,1], points[:,:,2], color='blue')
    if points2 is not None:
        ax.scatter(points2[:,:,0], points2[:,:,1], points2[:,:,2], color='orange')
    if points3 is not None:
        ax.scatter(points3[:,:,0], points3[:,:,1], points3[:,:,2], color='purple')

    plt.show()
