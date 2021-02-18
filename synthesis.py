import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
import matplotlib.pyplot as plt
import matplotlib.colors
import time

from envmap import EnvironmentMap, projections

import utils
from optical_flow import farneback_of, visualize_flow, visualize_flow_arrows
from extendedcubemap import ExtendedCubeMap
from envmap import EnvironmentMap #TODO create interface for this dependency
        
class Synthesizer2DoF:
    """
    Creates a 2DoF synthesizer that can be used to synthesize new viewpoints in a scene within the convex hull of captured viewpoints. 
    """
    def __init__(self, capture_set):
        """
        Initializes all the instance variables required throughout the synthesis process 

        capture_set: An instance of the class CaptureSet, containing the image paths and, locations and rotations of the captured viewoints
        """
        self.dev_angles = None
        self.interpolation_distances = None
        self.distances = None
        self.uvs = {}
        self.capture_set = capture_set
        self.indices = []
        self.best_indices_flow = None
        self.best_indices_reg = None
        self.point = None
        self.intersections = None
        self.flow_blend = None
        self.reg_blend = None

    def clear(self):
        """
        Clears all the instance variables before a new synthesis pass
        """
        self.dev_angles = None
        self.interpolation_distances = None
        self.distances = None
        self.uvs = {}
        self.best_indices_flow = None
        self.best_indices_reg = None
        self.flow_blend = None
        self.reg_blend = None

    def synthesize(self, indices, point, flow=False, knn=2, flow_precision=2, visualize=(False, "out", utils.OUT)):
        """
        Synthesizes a point from the specified viewpoints using either flow-based blending or regular blending, saves the output images in latlong and cubemap format in utils.OUT if not otherwise specified.

        indices: viewpoints of the capture set to use, has to be a valid index from the CaptureSet
        point: point to be synthesized, given in format: [x,y,0]
        flow: if True, use flow-based blending, else use regular blending
        knn: if using regular blending, the k nearest neighbors (based on deviation angle) to be used
        visualize: (0,1,2)
            0: if True, saves the associated data visualizations
            1: id/name to identify the synthesized viewpoint
            2: path for saving the images

        returns: the synthesized image of the desired viewpoint
        """
        start = time.time()
        print("synthesizing ", point)
        print("using ", indices, " for synthesis")

        self.prepare_synthesis(indices, point)

        if flow:
            out = self.flow_blend_image(flow_precision)
        else:
            out = self.regular_blend_image(knn)

        self.visualize_synthesis(visualize[1], details=visualize[0], type=("flow" if flow else "reg"), path=visualize[2])
            
        end = time.time()
        print("Synthesis stats:\n \telapsed time: {0:4.2f}s\n \tnum viewpoints used: {1:d} ".format((end - start), len(indices)))

        return out

    def prepare_synthesis(self, indices, point):
        """
        Calculates the ray-sphere intersections and the deviation angles 
        which are required for the different blending methods

        Note: This function stores information in the class such as scene intersection points and deviation angles so that they can be visualized later on. This information is reset at each new synthesis operation
        """
        #reset and fill the interpolation information for later use
        self.clear()
        self.indices = indices
        self.point = point
        dimensions = self.capture_set.get_capture(indices[0]).img.shape
        self.dev_angles = np.full((dimensions[0], dimensions[1], len(indices)), np.pi)
        #get the rays from the new point that correspond to the uv coordinates of an equirectangular image at that point (points on the unit sphere in the point's local coordinate system)
        u,v = calc_uv_coordinates(dimensions[0])
        nx, nz, ny, _ = projections.latlong2world(u,v) #switch y and z because projections uses a different representation
        targets = np.dstack((nx, ny, nz))

        #calculate the intersections from the new point with the scene
        self.intersections = self.capture_set.calc_ray_intersection(point, targets)

        #for each input viewpoint, determine and store the difference between the ray angles of the viewpoint and the ray angles of the synthesized point (deviation angles)
        for counter, viewpoint_i in enumerate(indices):
            position = self.capture_set.get_position(viewpoint_i)

            #get the rays from this viewpoint that hit the intersection points
            theta, phi = calc_ray_angles(position, self.intersections)
            rays = calc_uvector_from_angle(theta, phi, self.capture_set.radius)
            #calculate the deviation angles (angle between the two vectors of each point is acos of dot product between rays (viewpoint vectors) and targets (synth point vectors))
            dot = np.sum((rays.flatten() * targets.flatten()).reshape(rays.shape), axis=2)
            dev_angles = np.arccos(np.around(dot, 5)) #round to avoid precision errors leading to values >|1|

            #calculate the sign of the angles, so that it is possible to determine whether two viewpoints are on the same "side" of the synthesized point or not
            rel_dev_angles = np.arctan2(rays[:,:,1], rays[:,:,0]) - np.arctan2(targets[:,:,1], targets[:,:,0])
            rel_dev_angles[rel_dev_angles>np.pi] -= 2*np.pi
            rel_dev_angles[rel_dev_angles<-np.pi] += 2*np.pi
            rel_dev_angles = np.around(rel_dev_angles, 5)

            angle_sign = np.ones_like(rel_dev_angles)
            angle_sign[rel_dev_angles < 0] = -1
            dev_angles = dev_angles * angle_sign
            self.dev_angles[:,:,counter] = dev_angles

            #get the uv coordinates that correspond to the rays
            u,v = projections.world2latlong(rays[:,:,0], rays[:,:,2], rays[:,:,1])#switch y and z because EnvironmentMap has a different representation
            self.uvs[viewpoint_i] = np.dstack((u,v))

    def regular_blend_image(self, knn):
        """
        "Regular blending"
        Blends the reprojected images of the k nearest neighbors according to a weighting scheme based on the deviation angles for each pixel
        knn: number of neighbors to incorporate in the result image

        returns: the synthesized image

        Note: This function uses information stored by the interpolation function and also adds information
        """
        #get the knn first indices and angles of the sorted deviation angles (these indices are in [0,cap_set.get_size()] and are not the actual viewpoint indices)
        best_indices = np.argsort(np.abs(self.dev_angles), axis=-1)[:,:,:knn]
        self.best_indices_reg = best_indices
        dev_angles = np.sort(np.abs(self.dev_angles), axis=-1)[:,:,:knn]

        #use these deviation angles to calculate the weights
        #weights matrix is an "image" of weights in order of best-worst angle
        weights = 1 / (1 + np.exp(500*(dev_angles - 0.017)))

        #modify the weights so that they sum up to 1
        sums = np.sum(weights, axis=-1)
        weights /= sums[:,:,np.newaxis]

        #find all of the viewpoints that are in the top knn and reproject these viewpoints using the corresponding uv coordinates in self.uvs
        used_indices = np.unique(best_indices)
        #prepare array to store the reprojected images
        reproj_imgs = np.zeros((used_indices.shape[0], self.dev_angles.shape[0], self.dev_angles.shape[1], 3))

        for i, vp_i in enumerate(used_indices):
            reprojected = self.reproject(self.capture_set.get_capture(self.indices[vp_i]).img, self.uvs[self.indices[vp_i]]) #use real viewpoint index instead of the location in the used_index list
            viewpoint_indices = np.nonzero(best_indices == vp_i)

            #masks will be filled with eigher weight or 0 depending on how much influence the pixel at each rank should have (rank as in best - k-best deviation angle)
            mask = np.zeros_like(weights)
            mask[viewpoint_indices] = weights[viewpoint_indices]
            mask = np.add.reduce(mask, axis=2)

            # multiply mask by weight for each index used, then multiply by reprojection
            reproj_imgs[i] = reprojected * mask[:,:,np.newaxis]

        # sum up all masked & weighted reprojections
        self.reg_blend = np.add.reduce(reproj_imgs, axis=0)
        return self.reg_blend

    def flow_blend_image(self, precision=2):
        """
        "Flow-based blending"
        Synthesize the image using flow-based blending:
            For each pixel, gets the two closest (by deviation angle)
            viewpoints A and B on _either_ side of the synthesized point,
            uses 1DoF interpolation to interpolate the viewpoint v_AB
            that is at the intersection of SP and AB 
            then reprojects v_AB to the position of the synthesized point

        precision: the precision with which to interpolate,
            i.e., the number of decimal points to round to when determining the interpolation distance.
            1: [0.0, .. 0.1, 1.0] (11 images),
            2: [0.00, 0.01, .. , 0.99, 1.00] (101 images),
            >2 will lead to extremely long computation times and should be avoided

        returns: the synthesized image

        Note: This function uses information stored by the interpolation function and also adds information
        """
        #get the deviation angles (which are in [-180,180]) and shift the angles <0 by + 360 degrees so that the largest negative angles (closest to 0) are now closest to 360
        mod_dev = np.copy(self.dev_angles)
        mod_dev[mod_dev < 0] += 2*np.pi

        #sort the modified angles and take the angles closest to 0 and closest to 360, which yields the two viewpoints with the smallest deviation angle _on either side_
        sorted_indices = np.argsort(mod_dev, axis=-1)
        best_indices = np.dstack((sorted_indices[:,:,0], sorted_indices[:,:,-1]))
        self.best_indices_flow = np.copy(best_indices)

        #get 2D vectors between best two indices for the best indices for each pixel
        unique_indices = np.unique(np.reshape(best_indices, (best_indices.shape[0]*best_indices.shape[1], best_indices.shape[2])), axis=0)
        best_indices = best_indices.reshape((best_indices.shape[0] * best_indices.shape[1], best_indices.shape[2]))
        vectors_A = np.zeros_like(best_indices).astype(np.float64)
        vectors_B = np.zeros_like(best_indices).astype(np.float64)
        for pair in unique_indices:
            pos = self.capture_set.get_positions([self.indices[pair[0]], self.indices[pair[1]]])
            vec_A = pos[0][:2]
            vec_B = pos[1][:2]

            mask = np.equal(best_indices, pair)
            mask = np.logical_and(mask[:,0], mask[:,1])
            mask = np.dstack((mask, mask))
            np.putmask(vectors_A, mask, vec_A)
            np.putmask(vectors_B, mask, vec_B)

        vectors_A = vectors_A.reshape(self.best_indices_flow.shape)
        vectors_B = vectors_B.reshape(self.best_indices_flow.shape)

        '''
        intersect line AB with line point+targets SP (in 2D)

        given:
            line AB: A + t * (B - A) | A is viewpoint A, B is viewpoint B (selected by best indices)
            line SP: S + u * (P - S) | S is synthesized point, P is target point 

        formula: (from https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line)
                (x_A - x_S)(y_S - y_P) - (y_A - y_S)(x_S - x_P)
            t = -----------------------------------------------
                (x_A - x_B)(y_S - y_P) - (y_A - y_B)(x_S - x_P)

                encoding:

                xAS * ySP - yAS * xSP
                ---------------------
                xAB * ySP - yAB * xSP

        if t < 0 or t > 1 --> no intersection
        else dist = t
        '''
        targets = np.repeat(self.intersections[np.newaxis, int(self.intersections.shape[0]/2)], self.intersections.shape[0], axis=0)[:,:,:2]

        AS = vectors_A - self.point[:2]
        SP = self.point[:2] - targets
        AB = vectors_A - vectors_B

        self.interpolation_distances = (AS[:,:,0] * SP[:,:,1] - AS[:,:,1] * SP[:,:,0]) / ( AB[:,:,0] * SP[:,:,1] - AB[:,:,1] * SP[:,:,0] )

        #for the points that for some reason are outside of the line segment AB, use the closer viewpoint
        self.interpolation_distances[self.interpolation_distances < 0] = 0
        self.interpolation_distances[self.interpolation_distances > 1] = 1

        #if the s_point is exactly on the line between A and B (resulting in t = nan), use t = |AS|\|AB|
        nan_indices = np.isnan(self.interpolation_distances)
        AS_len = np.sqrt(np.power(AS[nan_indices][:,0],2)+np.power(AS[nan_indices][:,1],2))
        AB_len = np.sqrt(np.power(AB[nan_indices][:,0],2)+np.power(AB[nan_indices][:,1],2))
        self.interpolation_distances[nan_indices] = AS_len / AB_len
        #in the case that AB_len was zero, there are still nan numbers, so filter again, this time replacing nan with 0
        self.interpolation_distances[np.isnan(self.interpolation_distances)] = 0

        #visualize where the points and lines etc are to debug
        #elevation has no impact, since only points on a plane are used
#        for l in range(0, self.interpolation_distances.shape[1], 60):
#            uvs = (10,l)
#            print(self.interpolation_distances[uvs])
#            self.show_lr_points(uvs, targets[uvs[0], uvs[1]], self.interpolation_distances[uvs], saveas=utils.OUT + "flow_pos" + str(l) + ".jpg")

        #round in order to reduce the number of different sets (reduces accuracy but also compute time)
        np.round(self.interpolation_distances, precision, self.interpolation_distances)

        #find the distinct pairs of best indices & interpolation distances that will be used so that the interpolated, reprojected images for these pixels only have to be calculated once
        sets = np.dstack((self.best_indices_flow, self.interpolation_distances))
        unique_pairs = np.unique(np.reshape(sets, (self.best_indices_flow.shape[0]*self.best_indices_flow.shape[1], sets.shape[2])), axis=0)

        masks = {}
        image = np.zeros((self.dev_angles.shape[0], self.dev_angles.shape[1], 3))
        for u_pair in unique_pairs:
            #get the actual indices
            pair = (self.indices[u_pair[0].astype(np.uint8)], self.indices[u_pair[1].astype(np.uint8)])
#            print("calculating image at ", u_pair[2], " between ", pair[0], "and", pair[1])
            dist = u_pair[2]
            #where best_indices and distance is u_pair -> 1 else 0
            mask = (sets == u_pair)
            mask = np.logical_and(np.logical_and(mask[:,:,0], mask[:,:,1]), mask[:,:,2])

            #retrieve the flow from the capture set for this image pair
            flow = self.capture_set.get_flow(pair)
            interpolator = Interpolator1DoF(self.capture_set.get_capture(pair[0]).img, self.capture_set.get_capture(pair[1]).img, flow=flow)
            shifted_cube = interpolator.interpolate(dist)
            shifted_latlong = EnvironmentMap(shifted_cube, "cube").convertTo("latlong").data
            positions = self.capture_set.get_positions(pair)
            new_pos = positions[0] + dist * (positions[1] - positions[0])

            #get the rays from this viewpoint that hit the intersection points
            #this step is the same as in self.interpolate
            theta, phi = calc_ray_angles(new_pos, self.intersections)
            rays = calc_uvector_from_angle(theta, phi, self.capture_set.radius)
            u,v = projections.world2latlong(rays[:,:,0], rays[:,:,2], rays[:,:,1])#switch y and z because EnvironmentMap has a different representation
            image += mask[:,:,np.newaxis] * self.reproject(shifted_latlong, np.dstack((u,v)))
        self.flow_blend = image #store for later visualization
        return self.flow_blend

    def reproject(self, image, uvs):
        """
        Reprojects an image using the EnvironmentMap.interpolate function
        
        image: the image to reproject in latlong format
        uvs: the uv coordinates to use for reprojection

        returns the reprojected image with the same dimensions as the input image
        """
        envmap = EnvironmentMap(image, "latlong")
        u, v = np.split(uvs, 2, axis=2)
        envmap.interpolate(u,v)
        return envmap.data

    def trivial_synthesis(self):
        '''
        Find the nearest neighbor according to euclidean distance

        returns: image of the nn viewpoint
        '''
        distance_vectors = self.capture_set.get_positions(self.indices) - self.point
        distances = np.sqrt(np.sum(np.power(distance_vectors, 2), axis=-1))
        nn = self.indices[np.argmin(distances)]
        return self.capture_set.get_capture(nn).img

######################### visualization functions #########################

    def show_lr_points(self, uvs, target, t, saveas=None):
        """
        Draws the choice of viewpoints A and B for 1DoF interpolation for a pixel at position uv, designed to be used during the calculation of the interpolation distances in flow_blend_image()

        uv: pixel coordinates
        target: is defined as the intersection points of the approximated ray with the proxy sphere
        t: calculated interpolation distance
        saveas: if None, image is displayed, else location to store image

        Usage example:
            uvs = (10,10)
            self.show_lr_points(, targets[uvs[0], uvs[1]], self.interpolation_distances[uvs], saveas=utils.OUT + "flow_pos" + str(l) + ".jpg")
        """
        s_point = self.point[:2]
        index_A = self.indices[self.best_indices_flow[uvs[0], uvs[1], 0]]
        index_B = self.indices[self.best_indices_flow[uvs[0], uvs[1], 1]]
        pos_A, pos_B = self.capture_set.get_positions([index_A, index_B])

        fig, ax = plt.subplots()
        fig.canvas.set_window_title(str(uvs[0]) + ", " + str(uvs[1]))
        ax.set_xlim((-self.capture_set.radius, +self.capture_set.radius))
        ax.set_ylim((-self.capture_set.radius, +self.capture_set.radius))
        ax.set_aspect('equal')

        scene_model = plt.Circle((0, 0), self.capture_set.radius, color='0.8', fill=False)
        #draw the scene model (circle)
        ax.add_artist(scene_model)

        #draw the synthesized point
        ax.scatter(s_point[0], s_point[1], color="orange")
        ax.annotate("S", xy=(s_point[0], s_point[1]), xytext=(-10, 2), textcoords='offset points')

        #draw the first point as per index position
        ax.scatter(pos_A[0], pos_A[1], color="green")
        ax.annotate("A", xy=(pos_A[0], pos_A[1]), xytext=(-10, 2), textcoords='offset points')

        #draw the second point as per index position
        ax.scatter(pos_B[0], pos_B[1], color="blue")
        ax.annotate("B", xy=(pos_B[0], pos_B[1]), xytext=(-10, 2), textcoords='offset points')

        #draw the target point
        ax.scatter(target[0], target[1], color="black")
        ax.annotate("target", xy=(target[0], target[1]), xytext=(4, 4), textcoords='offset points')

        #draw the line between target and synthesized point so that the intersection can be verified
        ax.axline((s_point[0], s_point[1]), (target[0], target[1]))

        #draw intersection
        intersect = pos_A[:2] + (pos_B[:2] - pos_A[:2]) * t
        ax.scatter(intersect[0], intersect[1], color="red")
        ax.annotate("δ = " + str(np.round(t,2)), xy=(intersect[0], intersect[1]), xytext=(4, 4), textcoords='offset points')

        if saveas is None:
            plt.show()
        else:
            plt.savefig(saveas, bbox_inches='tight', dpi=utils.DPI)
        plt.clf()

    def visualize_data(self, data, saveas=None, maxval=None):
        """
        Saves or shows an image of interpolation data, e.g. deviation angles or interpolation distances

        data: the data, in the shape of an image
        saveas: if None, image is displayed, else location to store image
        maxval: set a maximum color value (so that different images can be compared with the same scale)
        """
        if data is None:
            print("No data passed to Interpolator2DoF.visualize_data(). Returning.")
            return

        fig = plt.figure()
        ax = plt.axes()
        imgplot = ax.imshow(data, cmap="RdBu_r")
        if maxval is not None:
            imgplot.set_clim(0,maxval)
            extend = 'max'
        else:
            extend = 'neither'

        cax = fig.add_axes([ax.get_position().x1+0.02,ax.get_position().y0,0.02,ax.get_position().height])
        plt.colorbar(imgplot, extend=extend, cax=cax) # Similar to fig.colorbar(im, cax = cax)
#        plt.clim(0,15)

        if saveas is None:
            plt.show()
        else:
            plt.savefig(saveas, bbox_inches='tight', dpi=utils.DPI)
        plt.clf()

    def get_best_deviations(self):
        return np.rad2deg(np.sort(np.abs(self.dev_angles), axis=-1)[:,:,:1])

    def get_distances(self):
        """
        Returns an image of the distances of the image areas used for synthesis
        """
        distance_vectors = self.capture_set.get_positions(self.indices) - self.point
        distances = np.sqrt(np.sum(np.power(distance_vectors, 2), axis=-1))

        indices = self.best_indices_reg[:,:,0].flatten()
        distance_patches = distances[indices].reshape(self.dev_angles.shape[:2])
        return distance_patches

    def visualize_indices(self, saveas=None):
        """
        Returns an image of the indices used for the 1DoF interpolation at each pixel
        """
        #get the best two indices for each pixel (1DoF interpolation is between these)
        pairs = np.dstack((self.best_indices_flow[:,:,0], self.best_indices_flow[:,:,1]))
        #sort them so that 0-1 is equal to 1-0 (since it is semantically)
        pairs = np.sort(pairs, axis=-1)
        pairs = pairs.reshape(self.best_indices_flow.shape[0]*self.best_indices_flow.shape[1], 2)
        indices, img = np.unique(pairs, return_inverse=True, axis=0)
        img = img.reshape(self.best_indices_flow.shape[:2]).astype(np.float64)

        labels = []
        for i_set in indices:
            indexA = self.indices[i_set[0]]
            indexB = self.indices[i_set[1]]
            labels.append(str(indexA) + "-" + str(indexB))
        fig, ax = plt.subplots()

        colors = ['#f5793a', '#a95aa1','#85c0f9', '#0f2080', '#fad9ad', '#ffffff']
        cmap = matplotlib.colors.ListedColormap(colors[:len(labels)])
        #change the min and max values so that the colorscale ticks are centered on each color
        img[img == np.amax(img)] += 0.5
        img[img == np.amin(img)] -= 0.5
        cax = ax.imshow(img, cmap=cmap)

        caxes = fig.add_axes([ax.get_position().x1+0.02,ax.get_position().y0,0.02,ax.get_position().height])
        cbar = fig.colorbar(cax, ticks=list(range(len(labels))), cax=caxes)
        cbar.ax.set_yticklabels(labels)


        if saveas is None:
            plt.show()
        else:
            plt.savefig(saveas, bbox_inches='tight', dpi=utils.DPI)
        plt.clf()

    def visualize_synthesis(self, identifier, details=True, path=utils.OUT, type=""):
        '''
        Save all the visualization data from the last synthesis operation
        type: "" | "reg" | "flow", where "" signifies "both"
        '''
        print(details)
        if type == "reg" or type == "":
            utils.cvwrite(self.reg_blend, str(identifier) + "_out_latlong" +".jpg")
            utils.cvwrite(utils.latlong2cube(self.reg_blend), str(identifier) + "_out_cube" +".jpg")
            if details:
                self.visualize_data(self.get_best_deviations(), saveas=utils.OUT +str(identifier) + '_dev_angles_' + type + ".jpg", maxval=14)
                self.visualize_data(self.get_distances(), maxval=self.capture_set.radius, saveas=utils.OUT + str(identifier) + '_index_distances_' + type + ".jpg")
                utils.cvwrite(self.trivial_synthesis(), str(identifier) + "_baseline_latlong.jpg")
                utils.cvwrite(utils.latlong2cube(self.trivial_synthesis()), str(identifier) + "_baseline_cube.jpg")

        if type == "flow" or type == "":
            if details:
                self.visualize_data(self.interpolation_distances, saveas=utils.OUT + str(identifier) + '_interpolation_distances' + ".jpg")
                self.visualize_indices(utils.OUT + str(identifier) + "_visualized_indices" + ".jpg")
            utils.cvwrite(self.flow_blend, str(identifier) + "_out_flow_latlong.jpg")
            utils.cvwrite(utils.latlong2cube(self.flow_blend), str(identifier) + "_out_flow_cube.jpg")

        if details:
            self.capture_set.draw_spoints(np.array([self.point]), np.array([identifier]), self.indices, ilabel=True, saveas=path + str(identifier) + "_scene.jpg")


######################### 1DoF Interpolator class #########################

types = ["latlong", "cube", "planar"]
class Interpolator1DoF:
    """
    creates a 1DoF interpolator for either planar or panoramic images
    """
    def __init__(self, imgA, imgB, type="latlong", flowfunc=farneback_of, param_path=".", flow=None):
        """
        Initializes the interpolator with ExtendedCubeMaps and calculates the flow between the two images.

        imgA, imgB: the viewpoints (as RGB images) used in the interpolation
        type: one of [latlong | cube | planar]
        flowfunc: the flow function to be used (at the moment, only farneback_of is implemented)
        flow, invert_flow: flow from A to B and from B to A, respectively
        """
        self.type = type
        if type == "planar":
            self.A = imgA
            self.B = imgB
            if flow is None:
                self.flow = flowfunc((self.A*255).astype(np.uint8), (self.B*255).astype(np.uint8), param_path)
                self.inverse_flow = flowfunc((self.B*255).astype(np.uint8), (self.A*255).astype(np.uint8), param_path)
            else:
                self.flow = flow[0]
                self.inverse_flow = flow[1]
        else:
            self.A = ExtendedCubeMap(imgA, type)
            self.B = ExtendedCubeMap(imgB, type)
            if flow is None:
                self.flow = self.A.optical_flow(self.B, flowfunc, param_path)
                self.inverse_flow = self.B.optical_flow(self.A, flowfunc, param_path)
            else:
                self.flow = flow[0]
                self.inverse_flow = flow[1]
        self.out = None

    def clear(self):
        self.out = None

    def interpolate(self, alpha):
        """
        Uses the flow vectors to shift and blend the images to synthesize point alpha between viewpoint A and B

        alpha: distance along the vector between A and B

        returns: the blended image

        Note: self.out stores the blended image as an ExtendedCubeMap until the next time interpolate is called
        """
        self.clear()

        if self.type == "planar":
            shifted_A = shift_img(self.get_image(self.A), self.flow, alpha)
            shifted_B = shift_img(self.get_image(self.B), self.inverse_flow, (1-alpha))
        else:
            shifted_A = self.A.apply_facewise(shift_img, self.flow, alpha)
            shifted_B = self.B.apply_facewise(shift_img, self.inverse_flow, (1-alpha))

        out = (1 - alpha) * shifted_A + alpha * shifted_B

        if self.type is "planar":
            self.out = out
        else:
            self.out = ExtendedCubeMap(out, "Xcube", fov=self.A.fov, w_original=self.A.w_original)
            out = self.out.calc_clipped_cube()
        return out

    def trivial_interpolation(self, alpha):
        """
        Use linear blending to combine the images.
        """
        return (1 - alpha) * self.get_image(self.A) + alpha * self.get_image(self.B)

    def get_image(self, A):
        """
        hack to be able to differentiate between ExtendedCubeMaps and planar images
        this should become obsolete once there is a distinct class for the two cases
        """
        if self.type is "planar":
            return A
        else:
            return  A.get_Xcube()

    def get_flow_visualization(self):
        """
        returns the color wheel and arrow visualization of the flow and the inverse flow
        """
        return (visualize_flow(self.flow), visualize_flow_arrows(self.get_image(self.A), self.flow, 16))


######################### helper functions #########################
def shift_img(img, flow, alpha):
    """
    Shifts an image (in any color space) along the provided flow vectors

    img: the image to be shifted
    flow: the vectors per pixel that img should be shifted by
    alpha: the amount the image should be shifted along the flow vectors

    returns: the shifted image
    """
    xx, yy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))

    xx_shifted = (xx - (flow[:,:,0] * alpha)).astype(np.float32)
    yy_shifted = (yy - (flow[:,:,1] * alpha)).astype(np.float32)

    shifted_coords = np.array([yy_shifted.flatten(), xx_shifted.flatten()])
    shifted_img = np.ones_like(img)
    for d in range(img.shape[2]):
        shifted_img[:,:,d] = np.reshape(map_coordinates(img[:,:,d], shifted_coords), (img.shape[0], img.shape[1]))

    return shifted_img

def invert_flow(flow):
    """
    Inverts the flow vector field
    """
    inverted_flow = shift_img(flow, flow, 1)
    return -inverted_flow

def calc_ray_angles(source, targets):
    """
    Calculates the angles of the rays from a origin point
    https://en.wikipedia.org/wiki/Spherical_coordinate_system#Coordinate_system_conversions

    source: the origin point of all the rays
    targets: an array of vector end points

    returns: an array of inclinations (theta) and an array of azimuths (phi) of the vector angles
    """
    #move so source is 0,0,0 and normalize
    u_vectors = calc_unit_vectors(targets - source)
    theta = np.arccos(u_vectors[...,2]) #z
    phi = np.arctan2(u_vectors[...,1],u_vectors[...,0])
    return theta, phi

def calc_unit_vectors(vectors):
    """
    Calculates the unit vectors of given vectors
    """
    mag = np.linalg.norm(vectors, axis=2)
    u_vectors = np.zeros_like(vectors)
    u_vectors[:,:,0] = vectors[:,:,0]/mag
    u_vectors[:,:,1] = vectors[:,:,1]/mag
    u_vectors[:,:,2] = vectors[:,:,2]/mag
    return u_vectors

def calc_uvector_from_angle(theta, phi, radius=1):
    """
    Calculates the unit vectors from the given inclinations and azimuths
    https://en.wikipedia.org/wiki/Spherical_coordinate_system#Coordinate_system_conversions

    theta: array of inclinations
    phi: array of azimuths
    radius: TODO this shouldn't be necessary 
    """
    x = np.sin(theta) * np.cos(phi) * radius
    y = np.sin(theta) * np.sin(phi) * radius
    z = np.cos(theta) * radius
    return calc_unit_vectors(np.dstack((x,y,z)))

def calc_uv_coordinates(imgheight):
    """
    Gets the uv coordinates of a latlong image

    adapted from EnvironmentMap.imageCoordinates at https://github.com/soravux/skylibs/blob/master/envmap/environmentmap.py

    returns the uv coordinates
    """
    cols = np.linspace(0, 1, imgheight*2*2 + 1)
    rows = np.linspace(0, 1, imgheight*2 + 1)

    cols = cols[1::2]
    rows = rows[1::2]

    return [d.astype('float32') for d in np.meshgrid(cols, rows)]
