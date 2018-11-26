"""
Project:      Equirectangular Camera Mapping
Name:         Primitives.py
Date:         November 26th, 2018.
Author:       Thomas Bellucci
Description:  All primitives used to construct the 3D environment including
              room bounding box and objects in the room.
"""

import numpy as np
from Transformations import *


class Cuboid():
    def __init__(self):
        """ The Cuboid class is used to construct the environment bounding box
            and to represent box-like objects in the environment.

            :var M_*:   Transformation matrices used to position cube in space.
            :var verts: Vertices of the cube centered at the origin.
            :var edges: Pairs (i, j) denoting an edge from corner i to j.
            :return:    None
        """
        # Construct initial transformation matrices.
        self.M_scl = scaling_matrix(1, 1, 1)
        self.M_rot = rotation_matrix(0, 0, 0)
        self.M_loc = translation_matrix(0, 0, 0)

        # Initialize cube's vertices and edges.
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


    def equirectangular_fragments(self, width, height, steps = 16):
        """ Returns a list of ((x0, y0, z0), (x1, y1, z1)) pairs representing line
            segments to be drawn by the rendering pipeline.

        :param width:  Width of the view cylinder.
        :param height: Height of the view cylinder.
        :param steps:  Number of vertices to interpolate on edges.
        :return:       A list of line segments represented as pairs of (x, y, z) tuples.
        """
        # Transform vertices to the desired location, scale and orientation.
        M = self.M_loc.dot(self.M_rot).dot(self.M_scl)
        verts = M.dot(self.verts)[:3].T

        # Maintain a list of line segments.
        line_segments = []

        # For each edge of the cube...
        for i, j in self.edges:
            # Interpolate additional points between corner points.
            v1, v2 = verts[i], verts[j]
            pts3D = [s * v1 + (1 - s) * v2 for s in np.linspace(0, 1, steps)]

            # Project each point onto unwrapped equirectangular view cylinder.
            pts3D = [equirectangular_transformation(p, width, height) for p in pts3D]

            # Draw the line segments between the projected edge points.
            for k in range(steps-1):
                pt1, pt2 = pts3D[k], pts3D[k + 1]

                # Prevent line segments from being drawn all the way across the image.
                if abs(pt2[0] - pt1[0]) < (width // 2):
                    line_segments.append((tuple(pt1), tuple(pt2)))

        return line_segments


    def orthographic_fragments(self, width, height, camera_view = "birdseye"):
        """ Renders the cube as a wireframe object using orthographic projection.

        :param width:       Width of the view plane.
        :param height:      Height of the view plane.
        :param camera_view: String denoting view (either "birdseye" or "top down").
        :return:            A list of lines (edges) represented as pairs of (x, y, z) tuples.
        """
        # Select camera position and orientation given 'camera_view' parameter.
        if camera_view == "birdseye":
            e = np.array([0.4, 0.9, 1])
            p = np.array([0, 0, 0])
            t = np.array([0, 1, 0])
        elif camera_view == "top down":
            e = np.array([0, 0, 1])
            p = np.array([0, 0, 0])
            t = np.array([0, 1, 0])
        else:
            raise Exception("{}: No such view available.".format(camera_view))

        # Construct matrices for orthographic transformation.
        M_cam = camera_matrix(e, p, t)
        M_orth = orthographic_projection_matrix(width, height)

        # Transform vertices to the desired location, scale and orientation and
        # position relative to the camera.
        M = M_orth.dot(M_cam).dot(self.M_loc).dot(self.M_rot).dot(self.M_scl)
        verts = M.dot(self.verts)[:3].T

        # Initialize lst to maintain segments.
        line_segments = []

        # For each edge of the cube...
        for i, j in self.edges:
            pt1, pt2 = verts[i], verts[j]

            # Add edge to list of line segments.
            line_segments.append((tuple(pt1), tuple(pt2)))

        return line_segments