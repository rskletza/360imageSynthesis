import cv2
import numpy as np

#based on https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

def optical_flow(imgA, imgB):
    if (imgA is None or imgB is None):
        raise Exception("Image(s) not loaded correctly. Aborting")

    #check whether greyscale, if not, convert
    if (len(imgA.shape) > 2):
        imgA = cv2.cvtColor(imgA.astype(np.float32), cv2.COLOR_BGR2GRAY)
    if (len(imgB.shape) > 2):
        imgB = cv2.cvtColor(imgB.astype(np.float32), cv2.COLOR_BGR2GRAY)


    hsv = np.zeros((imgA.shape[0], imgA.shape[1], 3))
    hsv[...,1] = 255

    flow = cv2.calcOpticalFlowFarneback(imgA, imgB, None, 0.5, 3, 15, 3, 5, 1.2, 0)

#    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
#    hsv[...,0] = ang*180/np.pi/2
#    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
#    rgb = cv2.cvtColor(hsv.astype(np.float32), cv2.COLOR_HSV2BGR)
#
#    cv2.imshow('flow', rgb)
#    k = cv2.waitKey(0)
    return flow

