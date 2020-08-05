from os import listdir
import numpy as np
import cv2
import matplotlib.pyplot as plt

import utils

class Capture:
    """
    simple object storing the position, rotation and the image data of a capture
    """
    def __init__(self, img, pos, rot):
        self.pos = pos
        self.rot = rot
        self.img = img

class CaptureSet:
    """
    set with all captures that holds metadata and paths and can retrieve pairs of both based on index
    also contains the model of the scene (as a sphere centered around 0,0,0)
    stores positional coordinates in x, y, z order, x/z being the plane parallel to the ground
    """
    def __init__(self, imgpath):
        """
        imgpath target must contain
            - a folder named images containing the images of the capture set in the same order as the metadata
            - a file named metadata.txt containing the metadata (the format of the metadata is described in utils.parse_metadata)
            """
        self.imgpath = imgpath
        self.names = sorted(listdir(imgpath + "/images"))
        self.positions, self.rotations = utils.parse_metadata(imgpath + "/metadata.txt")
        #center points
        self.positions = utils.center(self.positions)

        minima = np.amin(self.positions, axis=0)
        maxima = np.amax(self.positions, axis=0)
        self.center = minima + (maxima-minima) * 0.5
        self.radius = self.get_radius()

        self.images = [None] * len(self.names)

    def get_image(self, index):
        """
        retrieves an image from the capture set at the specified index
        lazy evaluation of the images: no images are loaded from the beginning
        if an image is requested, it is loaded and stored
        """
        if self.images[index] is None:
            name = self.imgpath + "/images/" + str(index) + ".jpg"
            self.images[index] = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
        return self.images[index]

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
        return Capture(self.get_image(index), self.get_position(index), self.get_rotation(index))

    def get_radius(self):
        """
        gets or calculates the (estimated) radius of the scene
        at the moment this is a placeholder function that returns a radius that is slightly larger than the furthest point but in the end this should return a more accurate scene radius
        """
        buf = 0.3
        maxima = np.amax(np.abs(self.positions), axis=0)
        rad = np.sqrt(np.power(maxima[0], 2) + np.power(maxima[2], 2))
        return rad * (1 + buf)

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
        ax.scatter(points[:,0], points[:,2], points[:,1], color='blue')

        if sphere:
            u = np.linspace(0, np.pi, 15)
            v = np.linspace(0, 2 * np.pi, 15)

            x = np.outer(np.sin(u), np.sin(v)) * self.radius
            y = np.outer(np.sin(u), np.cos(v)) * self.radius
            z = np.outer(np.cos(u), np.ones_like(v)) * self.radius

            ax.plot_wireframe(x, y, z, color='0.8')

        #draw additional (synthesized) points
        if s_points is not None:
            ax.scatter(s_points[:,0], s_points[:,2], s_points[:,1], color='orange')

        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')

        plt.show()
