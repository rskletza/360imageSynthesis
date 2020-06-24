import cv2
import numpy as np

from scipy.ndimage.interpolation import map_coordinates

def cvshow(img, filename=None):
    if(np.amax(img) > 1):
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
    if(np.amax(img) > 1):
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

    shifted_coords = np.array([yy_shifted.flatten(), xx_shifted.flatten()])
    shifted_img = np.zeros_like(img)
    for d in range(img.shape[2]):
        shifted_img[:,:,d] = np.reshape(map_coordinates(img[:,:,d], shifted_coords), (img.shape[0], img.shape[1]))

    return shifted_img


