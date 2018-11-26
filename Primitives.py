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

from Transformations import *


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


    def orthographic_render(self, width, height, camera_view = "birds eye", thickness = 1):
        """ Renders the cube as a wireframe object using perspective projection.

        :param width:       Width of the view plane.
        :param height:      Height of the view plane.
        :param camera_view: String denoting view (either "birds eye" or "top down").
        :param thickness:   Integer denoting thickness of the wireframe.
        :return:            Numpy array of shape (height, width, 1).
        """

        # Select camera position and orientation given 'camera_view' parameter.
        e, p, t = None, None, None
        if camera_view == "birds eye":
            e = np.array([-0.5, 0.7, 1])
            p = np.array([0, 0, 0])
            t = np.array([0, 1, 0])
        elif camera_view == "top down":
            e = np.array([1, 0, 0])
            p = np.array([0, 0, 0])
            t = np.array([0, 1, 0])
        else:
            raise Exception("ERROR: No such view available.")

        # Construct matrices for orthographic transformation.
        Mc = camera_matrix(e, p, t)
        Mo = orthographic_projection_matrix(width, height)
        M = Mo.dot(Mc)

        # Initialize frame buffer of (height, width, 1).
        buffer = np.zeros((height, width, 1), dtype=np.uint8)

        # For each edge of the cube...
        for i, j in self.edges:
            pt1, pt2 = self.verts[:, i], self.verts[:, j]

            # Project vertices onto view plane.
            pt1, pt2 = M.dot(pt1), M.dot(pt2)
            pt1 = np.int16(pt1[:2] / pt1[3])
            pt2 = np.int16(pt2[:2] / pt2[3])

            # Rasterize line in frame buffer.
            if np.min(pt1) >= 0 and np.min(pt2) >= 0:
                cv2.line(buffer, tuple(pt1), tuple(pt2), 255, thickness)

        return buffer


# TODO change to returning a list of line segments to draw to allow a z-buffer to be implemented