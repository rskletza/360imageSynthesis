import cv2
import numpy as np

from envmap import EnvironmentMap
from skimage.metrics import structural_similarity

import utils

def l1_error(gt, s):
    '''
    gt: ground truth
    s: synthesized image
    calculates the absolute error pixel-wise of two images in CIELAB color space 
    '''
    if not ( utils.is_cubemap(gt) and utils.is_cubemap(s) ):
        raise TypeError('The input must be in cubemap representation. Pass through prep_image first')

    if len(s.shape) > 2:
        vis = np.sum(np.abs(gt - s), -1)
    else:
        vis = np.abs(gt - s)
    #sum up complete error (since the black areas are 0, they have no impact)
    error = np.sum(vis)
    #calculate mean of the faces (not of the whole image, as the black areas would influence the result (because they have 0 error)
    error /= (vis.shape[0] / 4) * 6
    return (error, vis)

def l2_error(gt, s):
    '''
    calculates the squared error pixel-wise of two images in CIELAB color space
    '''
    if not ( utils.is_cubemap(gt) and utils.is_cubemap(s) ):
        raise TypeError('The input must be in cubemap representation. Pass through prep_image first')

    if len(s.shape) > 2:
        vis = np.sum(np.power(gt - s, 2), -1)
    else:
        vis = np.power(gt - s, 2)

    #sum up complete error (since the black areas are 0, they have no impact)
    error = np.sum(vis)
    #calculate mean of the faces (not of the whole image, as the black areas would influence the result (because they have 0 error)
    error /= (vis.shape[0] / 4) * 6
    return (error, vis)

def ssim(gt, s):
    if not ( utils.is_cubemap(gt) and utils.is_cubemap(s) ):
        raise TypeError('The input must be in cubemap representation. Pass through prep_image first')
    if len(gt.shape) > 2:
        mssim, vis = structural_similarity(gt, s, multichannel=True, full=True)
    else:
        mssim, vis = structural_similarity(gt, s, full=True)
    return(mssim, vis)

def prep_image(img):
    if not utils.is_cubemap(img):
        img = EnvironmentMap(img, "latlong").convertTo("cube").data.astype(np.float32)
    return img


