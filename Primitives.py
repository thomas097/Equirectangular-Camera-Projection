"""
Project:      Equirectangular Camera Mapping
Name:         Primitives.py
Date:         November 25th, 2018.
Author:       Thomas Bellucci
Description:  All primitives used to construct the 3D environment.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

from Transformations import rotation_matrix, translation_matrix, scaling_matrix
from Projections import equirectangular_projection


class Cuboid():
    def __init__(self):
        """ The Cuboid class is used to construct the environment bounding box
            and to represent box-like objects in the environment.
        """
        # Construct initial transformation matrices.
        self.M_scl = scaling_matrix(1, 1, 1)
        self.M_rot = rotation_matrix(0, 0, 0)
        self.M_loc = translation_matrix(0, 0, 0)

        # Initialize cube geometry.
        self.verts = np.array([[-1, -1, -1, -1,  1,  1,  1,  1],
                               [-1, -1,  1,  1, -1, -1,  1,  1],
                               [-1,  1, -1,  1, -1,  1, -1,  1],
                               [ 1,  1,  1,  1,  1,  1,  1,  1]])

        self.edges = np.array([[0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6],
                               [1, 2, 4, 3, 5, 3, 6, 7, 5, 6, 7, 7]]).T


    def set_location(self, px, py, pz):
        """ Updates location of the cube w.r.t. to the camera.
        :param p*: Location along some axis.
        :return:  None
        """
        self.M_loc = translation_matrix(px, py, pz)


    def set_rotation(self, rx, ry, rz):
        """ Updates rotation of the cube w.r.t. to its center.

        :param r*: Rotation around some axis.
        :return:   None
        """
        self.M_rot = rotation_matrix(rx, ry, rz)


    def set_scale(self, sx, sy, sz):
        """ Updates the scaling of the cube w.r.t. its center.

        :param s*: Scaling along some axis.
        :return:   None
        """
        self.M_scl = scaling_matrix(sx, sy, sz)


    def equirectangular_render(self, width, steps = 16, thickness = 1):
        """ Renders the cube as a wireframe object in equirectangular format.

        :param width:     Width of the view plane (height being width/2).
        :param steps:     Number of vertices to interpolate on edges.
        :param thickness: Integer denoting thickness of the wireframe.
        :return:          Numpy array of shape (width/2, width, 1).
        """
        # Transform vertices to the desired location, scale and orientation.
        M = self.M_loc.dot(self.M_rot).dot(self.M_scl)
        verts = M.dot(self.verts)[:3].T

        # Initialize frame buffer with 2:1 aspect ratio.
        buffer = np.zeros((width // 2, width, 1), dtype = np.uint8)

        # For each edge of the cube...
        for i, j in self.edges:
            # Interpolate additional points between corner points.
            v1, v2 = verts[i], verts[j]
            pts3D = [s * v1 + (1 - s) * v2 for s in np.linspace(0, 1, steps)]

            # Project each point onto unwrapped equirectangular view cylinder.
            pts2D = [equirectangular_projection(p, width) for p in pts3D]

            # Draw the line segments between the projected edge points.
            for k in range(steps-1):
                pt1, pt2 = tuple(pts2D[k]), tuple(pts2D[k + 1])

                # Prevent lines from being drawn all the way across the image.
                if abs(pt2[0] - pt1[0]) < (width // 2):
                    cv2.line(buffer, pt1, pt2, 255, thickness)

        return buffer
