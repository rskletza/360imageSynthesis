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
        
def interpolate_nD(capture_set, indices, points, hack=None):
    """
    WIP
    """
    dimension = capture_set.get_capture(indices[0]).img.shape[0]
    imgformat = "latlong" #TODO get from envmap directly
    i = 0
    for point in points:
        print("synthesizing ", point)
        #calculate the intersections from the new point with the scene
        new = EnvironmentMap(dimension, imgformat)
        nx, nz, ny, _ = new.worldCoordinates() #switch y and z because EnvironmentMap has a different representation
        targets = np.dstack((nx, ny, nz))
        intersections = capture_set.calc_ray_intersection(point, targets)
        builder = ImageBuilder(new.data.shape[0], new.data.shape[1])
        for viewpoint_i in indices:
            #get the rays from this viewpoint that hit the intersection points
            position = capture_set.get_position(viewpoint_i)
#            capture_set.draw_scene(indices=[viewpoint_i], s_points=np.array([point]))

            #calculate the deviation angles
            theta, phi = calc_ray_angles(position, intersections)
            rays = calc_uvector_from_angle(theta, phi, capture_set.radius)
            #angle between the two vectors of each point is acos of dot product between rays (viewpoint vectors) and targets (synth point vectors)
            dot = np.sum((rays.flatten() * targets.flatten()).reshape(rays.shape), axis=2)
            dev_angles = np.arccos(np.around(dot, 5)) #round to avoid precision errors leading to values >|1|

#            capture_set.draw_scene(indices=[viewpoint_i], s_points=np.array([point]), points=[position + rays, intersections, point+targets], sphere=False, rays=[[position+rays, intersections]])

            #get the uv coordinates that correspond to the rays
            u,v = projections.world2latlong(rays[:,:,0], rays[:,:,2], rays[:,:,1])#switch y and z because EnvironmentMap has a different representation
#            redchannel = np.zeros_like(u)
#            utils.cvshow(np.dstack((redchannel,u,v)))

################# interpolate using flow and shift_img function ############
#seems like there are more artefacts using this than using interpolate
            #calculate the "flow" --> where which pixel has moved due to the new projection
#            u_orig, v_orig = new.imageCoordinates()
#            u_flow = (u_orig- u) * new.data.shape[1]
#            v_flow = (v_orig- v) * new.data.shape[0]
#            flow = np.dstack((u_flow,v_flow))
#            img = capture_set.get_capture(viewpoint_i).img
#            synthesized = shift_img(img, flow, 1)

################# interpolate using EnvironmentMap.interpolate function ############
            envmap = EnvironmentMap(capture_set.get_capture(viewpoint_i).img, "latlong", copy=True)
            envmap.interpolate(u,v)
            synthesized = envmap.data
#            utils.cvshow(builder.pixel_values, "out" + str(viewpoint_i))
################# end interpolation choice ################

#            utils.cvwrite(synthesized, str(i) + "reproj_" + str(point) + "_from_" + str(position) + ".jpg")
            i += 1
            builder.update(dev_angles, viewpoint_i, synthesized)
#        utils.cvshow(builder.indices_to_colors(len(indices)))
        if hack is not None:
            utils.cvwrite(builder.pixel_values, "out_" + str(hack) + ".jpg")
        else:
            utils.cvwrite(builder.pixel_values, "out.jpg")
#        utils.cvshow(builder.pixel_values, "out")

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

def rad2weight(angles, maxangle):
    """
    maxangle is the angle in rad that is weighted as zero
    """
    angles[angles > np.pi] -= 2 * np.pi
    angles = np.abs(angles)
    zero_indices = np.nonzero(angles >= maxangle)
    angles /= maxangle
    angles[zero_indices] = 1
    return 1-angles

class ImageBuilder:
    def __init__(self, height, width):
        self.min_dev_angles = np.full((height, width), np.pi)
        self.src_indices = np.zeros((height, width))
        self.pixel_values = np.zeros((height, width, 3))

    def update(self, dev_angles, index, image):
        update_indices = np.nonzero(dev_angles < self.min_dev_angles)

        self.min_dev_angles[update_indices] = dev_angles[update_indices]
        self.src_indices[update_indices] = index
        self.pixel_values[update_indices] = image[update_indices]
        test = np.zeros_like(self.pixel_values)
        test[update_indices] = image[update_indices]

    def indices_to_colors(self, numcolors=256):
        if np.amax(self.src_indices) == 0:
            colorvals = np.full_like(self.src_indices, 0)
        else:
            colorvals = (self.src_indices / (np.amax(self.src_indices))) * 256
        image = np.dstack((colorvals, np.full_like(colorvals, 255), np.full_like(colorvals, 255))).astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
