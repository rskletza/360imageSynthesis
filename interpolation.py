import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
import matplotlib.pyplot as plt

from envmap import EnvironmentMap, projections

import utils
from cubemapping import ExtendedCubeMap

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
#TODO make into Interpolator1D and create parent class
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
        
class Interpolator3D:
    def __init__(self, capture_set):
        self.dev_angles = None
        self.uvs = {}
        self.capture_set = capture_set
        self.indices = []
        self.best_indices = None
        self.point = None
        self.intersections = None

    def clear(self):
        self.dev_angles = None
        self.uvs = {}
        self.best_indices = None

    def interpolate(self, indices, point, knn=1):
        """
        WIP
        """
        self.clear()
        print("synthesizing ", point)
        print("using ", indices, " for synthesis")
        self.indices = indices
        self.point = point
        dimension = self.capture_set.get_capture(indices[0]).img.shape[0]
        imgformat = "latlong"

        #calculate the intersections from the new point with the scene
        #TODO don't use EnvironmentMap for this, just take the function that calculates the world coords
        new = EnvironmentMap(dimension, imgformat)
        #create structure to store uv coordinates of reprojected image for later use
        self.dev_angles = np.full((new.data.shape[0], new.data.shape[1], len(indices)), np.pi)
        nx, nz, ny, _ = new.worldCoordinates() #switch y and z because EnvironmentMap has a different representation
        targets = np.dstack((nx, ny, nz))
        self.intersections = self.capture_set.calc_ray_intersection(point, targets)
#        self.capture_set.draw_scene(indices=[], s_points=np.array([point]), points=[point+targets, self.intersections], sphere=True)

        #for each input viewpoint, determine and store the deviation angles
        for counter, viewpoint_i in enumerate(indices):
            position = self.capture_set.get_position(viewpoint_i)
#            self.capture_set.draw_scene(indices=[viewpoint_i], s_points=np.array([point]))

            #get the rays from this viewpoint that hit the intersection points
            theta, phi = calc_ray_angles(position, self.intersections)
            rays = calc_uvector_from_angle(theta, phi, self.capture_set.radius)
            #calculate the deviation angles
            #angle between the two vectors of each point is acos of dot product between rays (viewpoint vectors) and targets (synth point vectors)
            dot = np.sum((rays.flatten() * targets.flatten()).reshape(rays.shape), axis=2)
            dev_angles = np.arccos(np.around(dot, 5)) #round to avoid precision errors leading to values >|1|
            self.dev_angles[:,:,counter] = dev_angles

#            self.capture_set.draw_scene(indices=[viewpoint_i], s_points=np.array([point]), points=[position + rays, intersections, point+targets], sphere=False, rays=[[position+rays, intersections]])

            #get the uv coordinates that correspond to the rays
            u,v = projections.world2latlong(rays[:,:,0], rays[:,:,2], rays[:,:,1])#switch y and z because EnvironmentMap has a different representation
            self.uvs[viewpoint_i] = np.dstack((u,v))

        out = self.blend_image(knn)
        return out

    def blend_image(self, knn):
        sigma = 0.017
        #get the indices of the sorted deviation angles and get the knn first
        best_indices = np.argsort(self.dev_angles, axis=-1)[:,:,:knn]
        self.best_indices = best_indices
        #get the deviation angles of the knn first
        dev_angles = np.sort(self.dev_angles, axis=-1)[:,:,:knn] #TODO combine argsort and sort? or get values by index

        #use these deviation angles to calculate the weights
        #"image" of weights per rank in the knn order
        weights = 1 / (1 + np.exp(500*(dev_angles - sigma)))

        sums = np.sum(weights, axis=-1)
        weights /= sums[:,:,np.newaxis] #sum of weights should be 1
        #get "real" indices per rank in the knn order (instead of the location in the list

        #find all distinct viewpoint indices in the real indices image and reproject these viewpoints using self.uvs --> dict?
        used_indices = np.unique(best_indices)
        print("used indices: ", used_indices)
        reproj_imgs = np.zeros((used_indices.shape[0], self.dev_angles.shape[0], self.dev_angles.shape[1], 3))
#        for i in used_indices:
        for i in range(len(used_indices)):
            vp_i = used_indices[i] #viewpoint index
            reprojected = self.reproject(self.capture_set.get_capture(self.indices[vp_i]).img, self.uvs[self.indices[vp_i]])
            viewpoint_indices = np.nonzero(best_indices == i)
            # masks: for each index rank image, where index is reprojected image index -> 1 else 0 (or weight instead of 1)
            mask = np.zeros_like(weights)
            # multiply mask by weight for each index used, then multiply by reprojection
            mask[viewpoint_indices] = weights[viewpoint_indices]
            mask = np.add.reduce(mask, axis=2)
            reproj_imgs[i] = reprojected * mask[:,:,np.newaxis]
#            utils.cvshow(reproj_imgs[i])
        # sum up all masked & weighted reprojections
        image = np.add.reduce(reproj_imgs, axis=0)
        return image

    def reproject(self, image, uvs):
        envmap = EnvironmentMap(image, "latlong", copy=True)
        u, v = np.split(uvs, 2, axis=2)
        envmap.interpolate(u,v)
        return envmap.data

#    def show_point_influences(self, s_point, uv_index, intersection_point, show_inset=True, sphere=True, best_arrows=False):
    def show_point_influences(self, uv_height, uv_width, show_inset=True, sphere=True, best_arrows=False):
        uv_index = np.array([uv_height, uv_width])
        s_point = self.point
        intersection_point = self.intersections[uv_height, uv_width]
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        #best indices (actual viewpoint indices) for point uv
        best_real_indices = [ self.indices[b_i] for b_i in self.best_indices[uv_index[0], uv_index[1]] ]

        #all deviation angles for point uv
        dev_angles = np.rad2deg(self.dev_angles[uv_index[0], uv_index[1]])
        index2dev = {}
        for i in range(len(dev_angles)):
            index2dev[self.indices[i]] = dev_angles[i]

        #show the scene model (sphere)
        if sphere:
            u = np.linspace(0, np.pi, 15)
            v = np.linspace(0, 2 * np.pi, 15)
            x = np.outer(np.sin(u), np.sin(v)) * self.capture_set.radius
            y = np.outer(np.sin(u), np.cos(v)) * self.capture_set.radius
            z = np.outer(np.cos(u), np.ones_like(v)) * self.capture_set.radius
            ax.plot_wireframe(x, y, z, color='0.8')

        #show the synthesized point
        ax.scatter(s_point[0], s_point[1], s_point[2], color='orange')
        ax.text(s_point[0], s_point[1], s_point[2], 's')

        #show the scene intersection of the uv_index
        ax.scatter(intersection_point[0], intersection_point[1], intersection_point[2], color='green')
        ax.text(intersection_point[0], intersection_point[1], intersection_point[2], 'P')
        #show the ray from the s_point to the intersection
        i_point = intersection_point - s_point
        plt.quiver(s_point[0], s_point[1], s_point[2], i_point[0], i_point[1], i_point[2], arrow_length_ratio=0.1, color="orange")

        if best_arrows:
            starts = self.capture_set.get_positions(best_real_indices)
            ends = intersection_point - starts
            plt.quiver(starts[:,0], starts[:,1], starts[:,2], ends[:,0], ends[:,1], ends[:,2], arrow_length_ratio=0.1)

        #show the points used for this uv index
        for i,best_i in enumerate(best_real_indices):
            point = self.capture_set.get_position(best_i)
            ax.scatter(point[0], point[1], point[2], color='lime')
            ax.text(point[0], point[1], point[2], str(np.round(index2dev[best_i],2)) + "° (" + str(best_i) + ")")

        #show all the other points that are not the best indices
        if show_inset:
            inset_diff = np.setdiff1d(self.indices, best_real_indices, assume_unique=True)
            viewpoints = self.capture_set.get_positions(inset_diff)
            for i, vp in enumerate(viewpoints):
                ax.scatter(vp[0], vp[1], vp[2], color='mediumseagreen')
                #need the deviation angles in order of inset_diff
                #deviation angle argwhere indices == inset_diff[i]
                ax.text(vp[0], vp[1], vp[2], str(np.round(index2dev[inset_diff[i]], 2)) + "° (" + str(inset_diff[i]) + ")")


        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    def visualize_best_deviation(self, saveas=None):
        dev_angles = np.rad2deg(np.sort(self.dev_angles, axis=-1)[:,:,:1])
        avg = np.average(dev_angles, axis=-1)
        imgplot = plt.imshow(avg, cmap="RdYlGn_r")
        plt.colorbar()
        if saveas is None:
            plt.show()
        else:
            plt.savefig(saveas)
        plt.clf()

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
