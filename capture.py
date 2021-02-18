from os import listdir, path, makedirs
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from envmap import EnvironmentMap

import utils
import preproc, supplementals
import optical_flow
#TODO get rid of this dependency
from extendedcubemap import ExtendedCubeMap

class CaptureSet:
    """
    Object that holds metadata and paths of all viewpoints and can retrieve pairs of both based on index
    also contains the model of the scene (as a sphere centered around 0,0,0)
    stores positional coordinates in x, y, z order, x/y being the plane parallel to the ground
    """
    def __init__(self, location, radius=None, in_place=False, blenderfile=None):
        """
        location directory must contain
            - a directory named images containing the images of the capture set in the same order as the metadata with names from 0.jpg to N.jpg (no leading 0s)
            - a file named metadata.txt containing the metadata (the format of the metadata is described in preproc.parse_metadata)
            - (optional) a file named ofparams.json containing the optical flow parameters (see utils.load_params / utils.build_params)
        in_place: if true, images are normalized in place (i.e. rotated)
        blenderfile: if the optical flow should be synthesized with Blender, blenderfile is the location of the file to use for this
        """
        self.location = location
        self.names = sorted(listdir(location + "images"))
        self.positions = np.zeros((len(self.names), 3))
        self.rotations = np.zeros((len(self.names), 4))
        self.blenderfile = blenderfile

        #try loading previously stored, normalized metadata
        try:
            with open(location + 'positions.npy', 'rb') as f:
                self.positions = np.load(f)

            with open(location + 'rotations.npy', 'rb') as f:
                self.rotations = np.load(f)
        except FileNotFoundError:
            print("No previously stored information found, loading and normalizing metadata")
            if not in_place:
                imgs = location + 'images_raw'
                if not path.exists(imgs):
                    print("no path " + imgs)
                    try:
                        makedirs(imgs)
                    except OSError as exc: # guard agains race condition
                        if exc.ernno != errno.EEXIST:
                            raise

                #copy images over as backup
                for i in range(len(self.names)):
                    print("backing up " + self.names[i])
                    img = cv2.cvtColor(self.get_capture(i).img, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(location + 'images_raw/' + str(i) + '.jpg', img)

            #get raw positions and rotations
            raw_pos, raw_rot = preproc.parse_metadata(location + "metadata.txt")

            #center points
            self.positions = preproc.center(raw_pos)
            self.store_positions()

            self.rotations = raw_rot
            preproc.normalize_rotation(self)

        #create a directory for optical flow value storage
        of = location + 'optical_flow'
        if not path.exists(of):
            try:
                makedirs(of)
            except OSError as exc: # guard agains race condition
                if exc.ernno != errno.EEXIST:
                        raise

        self.set_scene(radius)

    def set_scene(self, radius):
        """
        Sets the parameters of the scene
        """
        minima = np.amin(self.positions, axis=0)
        maxima = np.amax(self.positions, axis=0)
        self.center = minima + (maxima-minima) * 0.5
        if radius is not None:
            self.radius = radius
        else:
            self.radius = self.calc_radius()
        print("radius: ", self.radius)

    def get_size(self):
        """
        returns the number of captured viewpoints
        """
        return len(self.positions)

    def get_position(self, index):
        """
        retrieves the position of the capture at the specified index
        """
        return self.positions[index]

    def get_positions(self, indices):
        """
        retrieves the positions of the captures at the specified indices
        returns a 2D array containing these positions in order of argument input
        """
        pos = []
        for i in indices:
            pos.append(self.get_position(i))
        return np.array(pos)

    def get_rotation(self, index):
        """
        retrieves the rotation of the capture at the specified index
        """
        return self.rotations[index]

    def get_capture(self, index):
        """
        retrieves the entire capture at the specified index, containing the image, the position and the rotation
        """
        name = self.location + "images/" + str(index) + ".jpg"
        return Capture(name, self.get_position(index), self.get_rotation(index))

    def get_captures(self, indices):
        """
        gets a set of captures
        """
        captures = {}
        for i in indices:
            captures[i] = self.get_capture(i)
        return captures

    def store_rotations(self, location=None):
        if location is None:
            location = self.location
        with open(location + 'rotations.npy', 'wb') as f:
            np.save(f, self.rotations)

    def store_positions(self, location=None):
        if location is None:
            location = self.location
        with open(location + 'positions.npy', 'wb') as f:
            np.save(f, self.positions)

    def get_flow(self, indices):
        """
        Lazy calculation of optical flow between two viewpoints
        if flow and inverse flow have already been calculated, they will just be retrieved
        optical flow is stored in cubemap shape
        returns flow and inverse flow
        """
        path = self.location + 'optical_flow/' + str(indices[0]) + '-' + str(indices[1]) + '.npy'
        path_inverse = self.location + 'optical_flow/' + str(indices[1]) + '-' + str(indices[0]) + '.npy'
        try:
            with open(path, 'rb') as f:
                flow = np.load(f)
            with open(path_inverse, 'rb') as f:
                inverse_flow = np.load(f)

        except FileNotFoundError:
            if self.blenderfile is not None:
                #if location is None, remote calculation is used
                flow, inverse_flow = supplementals.render_of(indices, self.blenderfile, location=None)
            else:
                A = ExtendedCubeMap(self.get_capture(indices[0]).img, "latlong")
                B = ExtendedCubeMap(self.get_capture(indices[1]).img, "latlong")

                flow = A.optical_flow(B, optical_flow.farneback_of, params=self.location)
                inverse_flow = B.optical_flow(A, optical_flow.farneback_of, params=self.location)
            with open(path, 'wb') as f:
                np.save(f, flow.astype(np.float32))
            with open(path_inverse, 'wb') as f:
                np.save(f, inverse_flow.astype(np.float32))

        return (flow, inverse_flow)

    def calc_radius(self):
        """
        calculates the estimated radius of the scene
        at the moment this is a placeholder function that returns a radius that is slightly larger than the furthest point but in the end this should return a more accurate scene radius
        """
        buf = 0.1
        maxima = np.amax(np.abs(self.positions), axis=0)
        rad = np.sqrt(np.power(maxima[0], 2) + np.power(maxima[1], 2))
        return (rad) * (1 + buf)

    def get_vps_in_radius(self, point, radius, exclude=None):
        '''
        Find all the viewpoints within a certain radius around a point

        point: center of radius
        radius: radius to use
        exclude: testing: when synthesizing captured viewpoints, exclude a specific captured viewpoint at this index
        '''
        distance_vectors = self.positions - point
        distances = np.sqrt(np.sum(np.power(distance_vectors, 2), axis=-1))
        vps = np.argwhere(distances < radius).flatten()
        #if no viewpoints exist within the radius, use all
        if len(vps) == 0:
            vps = np.array(list(range(len(self.positions))))
        vps.tolist()
        if exclude is not None:
            vps.remove(exclude)
        return vps

    def get_closest_vps(self, point, n=2, exclude=None):
        '''
        Find the closest viewpoints surrounding the input point

        quadrants are: 0,1,2,3, starting at top right, going clockwise
        n: either 2 or 4
        exclude: for testing: when synthesizing captured viewpoints, exclude a specific captured viewpoint at this index
        '''
        shifted = self.positions - point

        #if any of these input points is at the exact location of the point, return it
        x_zero = np.nonzero(shifted[:,0] == 0)
        y_zero = np.nonzero(shifted[:,1] == 0)
        identical_index = np.intersect1d(x_zero, y_zero)
        if len(identical_index) > 0:
            return [identical_index[0]]

        quadrants = [{}, {}, {}, {}]
        for i in range(shifted.shape[0]):
            if exclude is not None:
                if i == exclude:
                    continue
            if shifted[i][0] >= 0 and shifted[i][1] > 0: #top right
                #directly calculate the distance
                quadrants[0][i] = np.sqrt(np.sum(np.power(shifted[i], 2), axis=-1))

            elif shifted[i][0] < 0 and shifted[i][1] >= 0: #top left
                quadrants[3][i] = np.sqrt(np.sum(np.power(shifted[i], 2), axis=-1))

            elif shifted[i][0] > 0 and shifted[i][1] <= 0: #bottom right
                quadrants[1][i] = np.sqrt(np.sum(np.power(shifted[i], 2), axis=-1))
            else: #bottom left
                quadrants[2][i] = np.sqrt(np.sum(np.power(shifted[i], 2), axis=-1))

        #just use one of them because we always have a regular grid (instead of checking which of the opposites has smaller distances) TODO future work
        # TODO also get convex hull instead of just 2. There must be a convex hull.
        if len(quadrants[0]) == 0 or len(quadrants[2]) == 0: #there are no viewpoints in quadrants 0,2
            if len(quadrants[1]) == 0 or len(quadrants[3]) == 0: #there are no viewpoints in opposing quadrants, so just take the two closest
                distance_vectors = self.positions - point
                distances = np.sqrt(np.sum(np.power(distance_vectors, 2), axis=-1))
                return np.argsort(distances)[:2]
            else: #there are viewpoints in quadrants 1,3 but none in 0,2, so take 2
                A = sorted(quadrant[1].items(), key=lambda x: x[1], reverse=False)[0]
                B = sorted(quadrant[3].items(), key=lambda x: x[1], reverse=False)[0]
                return [A[0], B[0]]

        else: #there are viewpoints in all quadrants
            if n == 2:
                A = sorted(quadrants[0].items(), key=lambda x: x[1], reverse=False)[0]
                B = sorted(quadrants[2].items(), key=lambda x: x[1], reverse=False)[0]
                return [A[0], B[0]]

            elif n == 4:
                A = sorted(quadrants[1].items(), key=lambda x: x[1], reverse=False)[0]
                B = sorted(quadrants[3].items(), key=lambda x: x[1], reverse=False)[0]
                C = sorted(quadrants[0].items(), key=lambda x: x[1], reverse=False)[0]
                D = sorted(quadrants[2].items(), key=lambda x: x[1], reverse=False)[0]
                return [A[0], B[0], C[0], D[0]]

            else:
                raise NotImplementedError("get_closest_vps is only defined for 2 or 4, not " + n)

    def calc_ray_intersection(self, point, vectors):
        """
        calculates the points at which the rays (point-vectors) intersect the scene
        https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
        http://viclw17.github.io/2018/07/16/raytracing-ray-sphere-intersection/

        point is origin, t is distance and D is unit vectors centered around 0,0,0
        x^2 + y^2 + z^2 = R^2 sphere function
        P^2 - R^2 = 0
        ...
        O^2 + D^2t^2 + 2ODt - R^O
        quadratic function with a=D^2, b= 2OD, c=O^2-R^2

        """
        #dot product of a unit vector with itself is 1
        a = np.ones(vectors.shape[:2])

        f_vecs = vectors.flatten()
        f_points = np.full((vectors.shape[0]*vectors.shape[1],vectors.shape[2]), point).flatten()
        dot = np.sum((f_vecs * f_points).reshape(vectors.shape), axis=2)
        b = 2 * dot
        c = np.full_like(a, np.dot(point, point)- np.power(self.radius, 2)) 

        discriminants = np.power(b, 2) - (4 * a * c)
        if np.amin(discriminants) < 0:
            sys.exit("Fatal error: No intersection found for some ray. Check if the synthesized viewpoint is actually within the scene boundaries!")

        t1 = (-b + np.sqrt(discriminants)) / (2 * a)
        t2 = (-b - np.sqrt(discriminants)) / (2 * a)
        #select the points with positive lengths
        lengths = t1 if np.amin(t1) >= 0 else t2
        intersections = point + (vectors * lengths[:,:,np.newaxis])
        
        return intersections

    def draw_scene(self, indices=None, s_points=None, sphere=True, points=None, rays=None, numpoints=200, twoD=False, saveas=None):
        """
        Draws the scene as a set of points and the containing sphere representing the scene
        indices:
            None -> all points are drawn
            [a,b,c] only points at indices a,b,c are drawn
        s_points: extra points (i.e. synthesized points) are drawn in a separate color
        sphere: if False, the sphere representing the scene is omitted
        points: a list of point arrays that can be drawn (e.g. intersections)
        rays: a list of rays to be drawn
        numpoints: how many points should be displayed
        twoD: draw the scene in 2D

        Example usage:
        capture_set.draw_scene(indices=[1,4], s_points=np.array([point]), points=[position + rays, intersections, point+targets], sphere=False, rays=[[position+rays, intersections]])
        """
        fig = plt.figure()
        if twoD:
            ax = plt.axes()
            ax.set_aspect('equal', 'box')
        else:
            ax = plt.axes(projection='3d')

        #draw captured viewpoints
        if indices is not None:
            viewpoints = self.positions[[tuple(indices)]]
        else:
            viewpoints = self.positions
            indices = list(range(self.get_size()))

        if twoD:
            ax.scatter(viewpoints[:,0], viewpoints[:,1], color='blue')
            for i in range(len(indices)):
                ax.text(viewpoints[i,0], viewpoints[i,1], indices[i])
        else:
            ax.scatter(viewpoints[:,0], viewpoints[:,1], viewpoints[:,2], color='blue')
            for i in range(len(indices)):
                ax.text(viewpoints[i,0], viewpoints[i,1], viewpoints[i,2], indices[i])

        if points is not None:
            colors = ['green', 'purple', 'magenta', 'cyan', 'yellow']
            for i in range(len(points)):
                p = utils.sample_points(points[i], numpoints)
                if twoD:
                    ax.scatter(p[:,:,0], p[:,:,1], color=colors[i%len(colors)])
                else:
                    ax.scatter(p[:,:,0], p[:,:,1], p[:,:,2], color=colors[i%len(colors)])

        if sphere:
            if twoD:
                ax.set_xlim((-self.radius, self.radius))
                ax.set_ylim((-self.radius, self.radius))
                circle = plt.Circle((0, 0), self.radius, color='0.8', fill=False)
                ax.add_artist(circle)
            else:
                u = np.linspace(0, np.pi, 15)
                v = np.linspace(0, 2 * np.pi, 15)

                x = np.outer(np.sin(u), np.sin(v)) * self.radius
                y = np.outer(np.sin(u), np.cos(v)) * self.radius
                z = np.outer(np.cos(u), np.ones_like(v)) * self.radius

                ax.plot_wireframe(x, y, z, color='0.8')

        #draw additional (synthesized) points
        if s_points is not None:
            if twoD:
                ax.scatter(s_points[:,0], s_points[:,1], color='orange')
            else:
                ax.scatter(s_points[:,0], s_points[:,1], s_points[:,2], color='orange')

        if rays is not None:
            for rayset in rays:
                origins = rayset[0]
                targets = rayset[1]
                if origins.shape != (3,):
                    o = utils.sample_points(origins, numpoints)
                else:
                    o = origins
                t = utils.sample_points(targets, numpoints)
                t = t.reshape(-1, t.shape[-1])
                if origins.shape != targets.shape:
                    if origins.shape[0] == 3 or origins.shape[0] == 2:
                        o = np.full_like(t, o)
                    else:
                        raise NotImplementedError
                else:
                    o = o.reshape(-1, o.shape[-1])
                diff = t - o

                if twoD:
                    pass
                else:
                    plt.quiver(o[:,0], o[:,1], o[:,2], diff[:,0], diff[:,1], diff[:,2], arrow_length_ratio=0.1)

        if saveas is None:
            plt.show()
        else:
            plt.savefig(saveas, bbox_inches='tight', dpi=utils.DPI)
        plt.clf()

    def draw_spoints(self, s_points, ids, indices=None, slabel=True, ilabel=False, sphere=True, saveas=None):
        """
        Draw a set of points and ids with labels
        s_points: array of points
        ids: labels of the points
        indices: if not None, draw the captured viewpoints at the given indices
        slabel: if true, label s_points
        ilabel: if true, label captured viewpoints
        sphere: if true, draw the proxy sphere
        """
        fig = plt.figure()
        ax = plt.axes()
        ax.set_aspect('equal', 'box')

        if sphere:
            ax.set_xlim((-self.radius, self.radius))
            ax.set_ylim((-self.radius, self.radius))
            circle = plt.Circle((0, 0), self.radius, color='0.8', fill=False)
            ax.add_artist(circle)

        try:
            image = utils.load_img(self.location + "../" + "top_gray.jpg")
        except FileNotFoundError:
            image = None

        gt_pos = s_points * np.array([-1,1,1])
        if indices is None:
            indices = list(range(len(self.positions)))
        vps = self.positions[[tuple(indices)]] * np.array([-1,1,1])

        if image is not None:
            with open(self.location + '../dims.txt', 'r') as f:
                data = f.readlines()
                dims0 = float(data[0].strip())
                dims1 = float(data[1].strip())
                dims = (dims0, dims1)
            ax.imshow(image, extent=(-dims[0]/2, dims[0]/2, -dims[1]/2, dims[1]/2))
            ax.axis('off')

        ax.scatter(gt_pos[:,0], gt_pos[:,1], color="orange", s=25, edgecolors="black")
        ax.scatter(vps[:,0], vps[:,1], color='blue', marker = 'x', edgecolors="black")
        if slabel:
            for j, p in enumerate(gt_pos):
                ax.annotate(ids[j], xy=(p[:2]), xytext=(2, 2), textcoords='offset points')
        if ilabel:
            for i, pos in enumerate(vps):
                ax.annotate(indices[i], xy=(pos[:2]), xytext=(2, 2), textcoords='offset points')

        if saveas is None:
            plt.show()
        else:
            plt.savefig(saveas, bbox_inches='tight', dpi=150)#utils.DPI)
        plt.clf()


class Capture:
    """
    Simple object storing the position, rotation and the image data of a capture
    """
    def __init__(self, imgpath, pos, rot):
        self.pos = pos
        self.rot = rot
        self.imgpath = imgpath
        self.img = utils.load_img(imgpath)

    def store_image(self, path=None):
        """
        stores the image at the imgpath if it does not already exist
        overwrites if it does
        """
        if path is None:
            path = self.imgpath

        utils.cvwrite(self.img, path)
        print("storing at " + path)

    def rotate(self, rotation):
        envmap = EnvironmentMap(self.img, 'latlong')
        envmap.rotate('DCM', rotation.as_matrix())
        self.img = envmap.data


