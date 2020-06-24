import cv2
import numpy as np

from utils import cvshow

#based on https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

def farneback_of(imgA, imgB):
    """
    uses the OpenCV implementation of the Farneback algorithm for calculating optical flow
    input: two images that the optical flow should be calculated on, flow is calculated from A to B
    output: a numpy array of flow vectors of the same width and height as the input images
    TODO: at the moment, also returns the visualization of the flow
    """
    if (imgA is None or imgB is None):
        raise Exception("Image(s) not loaded correctly. Aborting")

    #TODO nicer
    if (np.amax(imgA) < 1):
        imgA = (imgA * 255).astype(np.uint8)
    if (np.amax(imgB) < 1):
        imgB = (imgB * 255).astype(np.uint8)

    if imgA.dtype is (np.dtype('float64') or np.dtype('float32')):
        imgA = (imgA * 255).astype(np.uint8)
    if imgB.dtype is (np.dtype('float64') or np.dtype('float32')):
        imgB = (imgB * 255).astype(np.uint8)

    #check whether greyscale, if not, convert
    if (len(imgA.shape) > 2):
        imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    if (len(imgB.shape) > 2):
        imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    hsv = np.zeros((imgA.shape[0], imgA.shape[1], 3))
    hsv[...,1] = 255

    pyr_scale = [0.5]
    levels = [3, 5]
    winsize = [1, 13, 15, 100]
    iters = [5, 10, 15]
    poly = [(7,1.5)]

    flow = cv2.calcOpticalFlowFarneback(imgA, imgB, None, pyr_scale[0], levels[1], winsize[1], iters[2], poly[0][0], poly[0][1], 0)


#    for pyr in pyr_scale:
#        for l in levels:
#            for w in winsize:
#                for i in iters:
#                    for p in poly:
#                        flow = cv2.calcOpticalFlowFarneback(imgA, imgB, None, pyr, l, w, i, p[0], p[1], 0)
#
#
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

#    cv2.imwrite("../../data/out/recover/flow_pyr" + str(pyr) + "l" + str(l) + "w" + str(w) + "i" + str(i) + "p" + str(p) + ".jpg", bgr)
#    cv2.imwrite("../../data/out/recover/flow_test.jpg", bgr)

    return flow, bgr

