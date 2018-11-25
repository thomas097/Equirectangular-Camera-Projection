import numpy as np
import cv2
import matplotlib.pyplot as plt

from Transformations import rotation_matrix, translation_matrix, scaling_matrix
from Projections import equirectangular_projection


"""
Project:      Equirectangular Camera Mapping
Date:         25 nov 2018
Name:         Thomas Bellucci
Description:  This file contains all primitives used to construct the 3D 
              environment.
"""


class Cuboid():
    def __init__(self):
        # Construct transformation matrices for scale, rotation and location.
        self.M_scale = scaling_matrix(1, 2, 1)
        self.M_rot = rotation_matrix(0, 0, 0)
        self.M_loc = translation_matrix(0, 0, 0)
        self.M = self.M_loc.dot(self.M_rot).dot(self.M_scale)

        # Vertices (as rows) of cube centered around camera origin.
        self.verts = np.array([[-1, -1, -1, -1,  1,  1,  1,  1],
                               [-1, -1,  1,  1, -1, -1,  1,  1],
                               [-1,  1, -1,  1, -1,  1, -1,  1],
                               [ 1,  1,  1,  1,  1,  1,  1,  1]])

        # Define edges as being interconnected vertices.
        self.edges = np.array([[0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6],
                               [1, 2, 4, 3, 5, 3, 6, 7, 5, 6, 7, 7]]).T


    def equirectangular_render(self, width, steps = 16, color = (255, 255, 255), thickness = 1):
        """ Renders the cube as a wireframe in 2:1 equirectangular format.

        :param width:     Width of the viewplane.
        :param steps:     Number of vertices to interpolate on edges.
        :param color:     RGB tuple with values between 0 and 255.
        :param thickness: Integer denoting thickness of the wireframe.
        :return:          Numpy array of shape (width/2, width, 3).
        """
        # Transform vertices to desired position and orientation.
        verts = self.M.dot(self.verts)[:3].T

        # Initialize frame buffer.
        height = width // 2
        buffer = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw lines when they do not cross the entire image.
        thres = width // 4

        # For each line...
        for i, j in self.edges:
            # Interpolate between end points (corners).
            v1, v2 = verts[i], verts[j]

            pts3D = [s * v1 + (1 - s) * v2 for s in np.linspace(0, 1, steps)]

            # Project each 3D point onto view cylinder.
            pts2D = [equirectangular_projection(p, width) for p in pts3D]

            # Draw line segments.
            for k in range(steps-1):
                x0, y0 = pts2D[k]
                x1, y1 = pts2D[k + 1]
                if abs(x1-x0) < thres:
                    cv2.line(buffer, (x0, y0), (x1, y1), color, thickness)

        cv2.imshow('', buffer)
        cv2.waitKey(0)



c = Cuboid()
c.equirectangular_render(1200, color=(155, 0, 0), thickness=1)

