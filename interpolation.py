import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
import matplotlib.pyplot as plt

from envmap import EnvironmentMap, projections

import utils
from cubemapping import ExtendedCubeMap

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
    shifted_img = np.ones_like(img)
    for d in range(img.shape[2]):
        shifted_img[:,:,d] = np.reshape(map_coordinates(img[:,:,d], shifted_coords), (img.shape[0], img.shape[1]))

    return shifted_img

types = ["cube", "planar"]
class Interpolator:
    """
    creates an interpolator for either planar or cube (panoramic) images
    """
    def __init__(self, type):
        self.type = type

    def linear(self, A, B, alpha):
        out = (1-alpha) * self.get_image(A) + alpha * self.get_image(B)
        return out

    def flow(self, A, B, flow, alpha):
#        flow = np.full_like(flow, 0)
        shifted_A = shift_img(self.get_image(A), flow, alpha)
        shifted_B = shift_img(self.get_image(B), -flow, (1-alpha))

        out = (1 - alpha) * shifted_A + alpha * shifted_B

        if self.type is "cube":
            return ExtendedCubeMap(out, "Xcube", fov=A.fov, w_original=A.w_original)
        else:
            return out

    def get_image(self, A):
        if self.type is "cube":
            return  A.get_Xcube()
        elif self.type is "planar":
            return A
        else:
            raise NotImplementedError
        
def interpolate_nD(capture_set, indices, points):
    """
    WIP
    """
    dimension = capture_set.get_capture(indices[0]).img.shape[0]
    imgformat = "latlong" #TODO get from envmap directly
    for point in points:
        #calculate the intersections from the new point with the scene
        new = EnvironmentMap(dimension, imgformat)
        nx, ny, nz, _ = new.worldCoordinates()
        targets = np.dstack((nx, ny, nz))
        intersections = capture_set.calc_ray_intersection(point, targets)
        for viewpoint_i in indices:
            #get the rays from this viewpoint that hit the intersection points
            position = capture_set.get_position(viewpoint_i)
            theta, phi = calc_ray_angles(position, intersections)
            rays = calc_uvector_from_angle(theta, phi, capture_set.get_radius())

            capture_set.draw_scene(indices=[viewpoint_i], s_points=np.array([point]), points=[position + rays, intersections, point+targets], sphere=False, rays=[[position+rays, intersections]])
            #get the uv coordinates that correspond to the rays
            u,v = projections.world2latlong(rays[:,:,0], rays[:,:,1], rays[:,:,2])
#            redchannel = np.zeros_like(u)
#            utils.cvshow(np.dstack((redchannel,u,v)))

            #calculate the "flow" --> where which pixel has moved due to the new projection
            u_orig, v_orig = new.imageCoordinates()
            u_flow = (u_orig- u) * new.data.shape[1]
            v_flow = (v_orig- v) * new.data.shape[0]
            flow = np.dstack((u_flow,v_flow))
            synthesized = shift_img(capture_set.get_capture(viewpoint_i).img, flow, 1)
#            utils.cvshow(envmap.data, "01_ground_truth")
            utils.cvshow(synthesized, "02_synthesized_" + str(viewpoint_i))

################# interpolate using EnvironmentMap.interpolate function ############
#            envmap = EnvironmentMap(capture_set.get_capture(viewpoint_i).img, "latlong", copy=True)
#            envmap.interpolate(u,v)
#            utils.cvshow(envmap.data, "02_synthesized_" + str(viewpoint_i))


def calc_ray_angles(source, targets):
    """
    https://en.wikipedia.org/wiki/Spherical_coordinate_system#Coordinate_system_conversions
    """
    #move so source is 0,0,0 and normalize
    u_vectors = calc_unit_vectors(targets - source)
    theta = np.arccos(u_vectors[...,2]) #z
    phi = np.arctan2(u_vectors[...,1],u_vectors[...,0])
    return theta, phi

def calc_unit_vectors(vectors):
    mag = np.linalg.norm(vectors, axis=2)
    u_vectors = np.zeros_like(vectors)
    u_vectors[:,:,0] = vectors[:,:,0]/mag
    u_vectors[:,:,1] = vectors[:,:,1]/mag
    u_vectors[:,:,2] = vectors[:,:,2]/mag #TODO find a cleaner way to do this np.dot?
    return u_vectors

def calc_uvector_from_angle(theta, phi, radius=1):
    """
    https://en.wikipedia.org/wiki/Spherical_coordinate_system#Coordinate_system_conversions
    """
    x = np.sin(theta) * np.cos(phi) * radius
    y = np.sin(theta) * np.sin(phi) * radius
    z = np.cos(theta) * radius
    return calc_unit_vectors(np.dstack((x,y,z)))

