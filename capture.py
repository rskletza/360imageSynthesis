from os import listdir, path, makedirs
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from envmap import EnvironmentMap

import utils
import preproc

class Capture:
    """
    simple object storing the position, rotation and the image data of a capture
    """
    def __init__(self, imgpath, pos, rot):
        self.pos = pos
        self.rot = rot
        self.imgpath = imgpath
        self.img = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)

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

class CaptureSet:
    """
    set with all captures that holds metadata and paths and can retrieve pairs of both based on index
    also contains the model of the scene (as a sphere centered around 0,0,0)
    stores positional coordinates in x, y, z order, x/y being the plane parallel to the ground
    """
    def __init__(self, location):
        """
        location target must contain
            - a folder named images containing the images of the capture set in the same order as the metadata
            - a file named metadata.txt containing the metadata (the format of the metadata is described in preproc.parse_metadata)
        """
        self.location = location
        self.names = sorted(listdir(location + "/images"))
        self.positions, self.rotations = preproc.parse_metadata(location + "/metadata.txt")

        self.set_scene()

    def set_scene(self):
        minima = np.amin(self.positions, axis=0)
        maxima = np.amax(self.positions, axis=0)
        self.center = minima + (maxima-minima) * 0.5
        self.radius = self.get_radius()
        print("radius: ", self.radius)

    def get_size(self):
        return len(self.positions)

    def get_position(self, index):
        """
        retrieves the position of the capture at the specified index
        """
        return self.positions[index]

    def get_rotation(self, index):
        """
        retrieves the rotation of the capture at the specified index
        """
        return self.rotations[index]

    def get_capture(self, index):
        """
        retrieves the entire capture at the specified index, containing the image, the position and the rotation
        """
        name = self.location + "/images/" + str(index) + ".jpg"
        return Capture(name, self.get_position(index), self.get_rotation(index))

    def get_captures(self, indices):
        captures = {}
        for i in indices:
            captures[i] = self.get_capture(i)
        return captures

#    def update_captures(self, captures):
#        for k, v in captures.items():
#            self.position[int(k)] = v.pos
#            self.rotation[int(k)] = v.rot
#            v.store_image()

    def store_rotations(self, location=None):
        if location is None:
            location = self.location
        with open(location + '/rotations.npy', 'wb') as f:
            np.save(f, self.rotations)

    def store_positions(self, location=None):
        if location is None:
            location = self.location
        with open(location + '/positions.npy', 'wb') as f:
            np.save(f, self.positions)

#    def store(self, field):
#        if field is "positions":
#            pass
#        elif field is "rotations":
#            pass
#        elif field is "image":
#            pass
#        else:
#

    def get_radius(self):
        """
        gets or calculates the (estimated) radius of the scene
        at the moment this is a placeholder function that returns a radius that is slightly larger than the furthest point but in the end this should return a more accurate scene radius
        """
        buf = 0.5
        maxima = np.amax(np.abs(self.positions), axis=0)
        rad = np.sqrt(np.power(maxima[0], 2) + np.power(maxima[1], 2))
        return rad * (1 + buf)

    def calc_ray_intersection(self, point, vectors):
        """
        calculates the point at which the ray intersects the scene
        http://viclw17.github.io/2018/07/16/raytracing-ray-sphere-intersection/
        https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection

        point is origin, t is distance and D is unit vectors
        x^2 + y^2 + z^2 = R^2
        P^2 - R^2 = 0
        ...
        O^2 + D^2t^2 + 2ODt - R^O
        quadratic function with a=D^2, b= 2OD, c=O^2-R^2

        """
        #dot product of a vector with itself is 1
        a = np.ones(vectors.shape[:2])

        f_vecs = vectors.flatten()
        f_points = np.full(f_vecs.shape, point).flatten()
        dot = np.sum((f_vecs * f_points).reshape(vectors.shape), axis=2)
        b = 2 * dot
        c = np.full_like(a, 1 - np.power(self.radius, 2)) #again, dot product of identity is 1

        discriminants = np.power(b, 2) - (4 * a * c)
        if np.amin(discriminants) < 0:
            #TODO specify and throw error
            print("no intersection found at some point")
            return
        t1 = (-b + np.sqrt(discriminants)) / (2 * a)
        t2 = (-b - np.sqrt(discriminants)) / (2 * a)
        lengths = t1 if np.amin(t1) >= 0 else t2
        lengths = np.dstack((lengths, lengths, lengths))
        intersections = vectors * lengths
        
        return intersections

    def draw_scene(self, indices=None, s_points=None, sphere=True):
        """
        draws the scene as a set of points and the containing sphere representing the scene
        indices=None -> all points are drawn
        indices=[a,b,c] only points at indices a,b,c are drawn
        s_points -> extra points (i.e. synthesized points) are drawn in a separate color
        if sphere=False, the sphere representing the scene is omitted
        this is a member function instead of a free function, so it can ensure correct handling of the axes (may change this later)
        """
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        #draw captured viewpoints
        if indices is not None:
            points = self.positions[indices]
        else:
            points = self.positions
        ax.scatter(points[:,0], points[:,1], points[:,2], color='blue')

        if sphere:
            u = np.linspace(0, np.pi, 15)
            v = np.linspace(0, 2 * np.pi, 15)

            x = np.outer(np.sin(u), np.sin(v)) * self.radius
            y = np.outer(np.sin(u), np.cos(v)) * self.radius
            z = np.outer(np.cos(u), np.ones_like(v)) * self.radius

            ax.plot_wireframe(x, y, z, color='0.8')

        #draw additional (synthesized) points
        if s_points is not None:
            ax.scatter(s_points[:,0], s_points[:,1], s_points[:,2], color='orange')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

#TODO remove class and write into function
class NormalizedCaptureSet(CaptureSet):
    def __init__(self, location, raw_capture_set=None):
        """
        NormalizedCaptureSet("../../data/captures/meetingRoom_test/normalized")
        or
        NormalizedCaptureSet("../../data/captures/meetingRoom_test/normalized", raw_capture_set)
        """
        self.location = location

        if raw_capture_set is None:
            with open(location + '/positions.npy', 'rb') as f:
                self.positions = np.load(f)

            with open(location + '/rotations.npy', 'rb') as f:
                self.rotations = np.load(f)

        else:
            imgs = location + '/images'
            if not path.exists(imgs):
                try:
                    makedirs(imgs)
                except OSError as exc: # guard agains race condition
                    if exc.ernno != errno.EEXIST:
                        raise

            #copy images over for modification
            for i in range(raw_capture_set.get_size()):
                print(i)
                img = cv2.cvtColor(raw_capture_set.get_capture(i).img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(location + '/images/' + str(i) + '.jpg', img)

            #center points
            self.positions = preproc.center(raw_capture_set.positions)
            self.store_positions()

            self.rotations = raw_capture_set.rotations
            preproc.normalize_rotation(self)

        self.set_scene()