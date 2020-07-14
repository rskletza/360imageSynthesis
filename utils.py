import cv2
import numpy as np
import json
from skimage import io

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
