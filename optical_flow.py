import cv2
import numpy as np

import utils

def farneback_of(imgA, imgB, param_path="."):
    """
    uses the OpenCV implementation of the Farneback algorithm for calculating optical flow
    input: two uint8 images that the optical flow should be calculated on, flow is calculated from A to B
    param_path: if specific optical flow parameters should be used, pass the file (which was created by utils.build_params)

    returns: a numpy array of flow vectors of the same width and height as the input images
    """
    if (imgA is None or imgB is None):
        raise Exception("Image(s) not loaded correctly. Aborting")

    if (imgA.dtype != np.uint8):
        raise Exception("Image A needs to be type uint8. It is currently " + str(imgA.dtype))

    if (imgB.dtype != np.uint8):
        raise Exception("Image B needs to be type uint8. It is currently " + str(imgB.dtype))

    #check whether greyscale, if not, convert
    if (len(imgA.shape) > 2):
        imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    if (len(imgB.shape) > 2):
        imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    #if file can't be loaded, uses default parameters
    params = utils.load_params(param_path)

    flow = cv2.calcOpticalFlowFarneback(imgA, imgB, None, params['pyr_scale'], params['levels'], params['winsize'], params['iters'], params['poly_expansion'], params['sd'], 0)

    return flow

def visualize_flow(flow):
    """
    Visualizes the optical flow field with colors
    based on https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
    """
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3))
    hsv[...,1] = 255
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)/255
    return bgr

def visualize_flow_arrows(img, flow, step=16):
    """
    Visualizes the flow field with arrows
    adapted from
    https://github.com/opencv/opencv/blob/master/samples/python/opt_flow.py
    """
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = img
#    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#    cv2.polylines(vis, lines, 0, (255, 0, 0), thickness=3)
    for (x1,y1), (x2, y2) in lines:
        cv2.arrowedLine(vis, (x1,y1), (x2,y2), color=(1, 0, 0), thickness=4, tipLength=0.2)
#    for (x1, y1), (_x2, _y2) in lines:
#        cv2.circle(vis, (x1, y1), 1, (255, 0, 0), -1)
    return vis #vis.astype(np.float32)/255
