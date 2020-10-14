import cv2
import numpy as np
import optical_flow, utils
import matplotlib.pyplot as plt

def blender_exr2flow(path):
    """
    Adapted from http://www.tobias-weis.de/groundtruth-data-for-computer-vision-with-blender/
    """
    exr = cv2.imread(path, cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
    exr = cv2.cvtColor(exr, cv2.COLOR_BGR2RGB)
    flow = np.dstack((exr[:,:,0], -exr[:,:,1]))
#    plt.imshow(exr[:,:,0])
#    plt.show()
#    plt.imshow(exr[:,:,1])
#    plt.show()
#    plt.imshow(exr[:,:,2])
#    plt.show()
    return flow

    """
    file = OpenEXR.InputFile(exr)

    # Compute the size
    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R,G,B) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B") ]

    img = np.zeros((h,w,3), np.float64)
    img[:,:,0] = np.array(R).reshape(img.shape[0],-1)
    img[:,:,1] = -np.array(G).reshape(img.shape[0],-1)

    hsv = np.zeros((h,w,3), np.uint8)
    hsv[...,1] = 255

    mag, ang = cv2.cartToPolar(img[...,0], img[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    return img, bgr, mag,ang

    """
