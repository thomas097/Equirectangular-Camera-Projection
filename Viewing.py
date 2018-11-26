"""
Project:      Equirectangular Camera Mapping
Name:         Viewing.py
Date:         November 26th, 2018.
Author:       Thomas Bellucci
Description:  Contains a class Viewer, which is used to render/rasterize
              geometry on screen.
"""

import numpy as np
from Primitives import *


class Viewer():
    def __init__(self):
        """ The Viewer class is used to render the objects described in
            Primitives.py.

            :var frame_buffer: Buffer containing the RGB values of a render.
            :var depth_buffer: Buffer containing the depth values of a render.
        """
        self.frame_buffer = None
        self.depth_buffer = None


    def __draw_line(self, pt0, pt1, color):
        """ Draws a line in the frame buffer making sure to draw near pixel last
            using the depth buffer.

        :param pt0:       First point.
        :param pt1:       Second point.
        :param color:     The RGB color to use for the line.
        :return:          None
        """
        # determine size of buffer
        h, w = self.depth_buffer.shape[:2]

        # Define parametric line P + t*D
        P = np.array(pt0)
        D = np.array(pt1) - P

        # Draw lines that are more horizontal than vertical (dx > dy).
        if abs(D[0]) > abs(D[1]):
            D = D / abs(D[0])

            x0, x1 = int(min(pt1[0], pt0[0])), int(max(pt1[0], pt0[0]))
            for _ in range(x0, x1):
                x, y, z = P.astype(int)
                z = P[2]
                if 0 <= y < h and 0 <= x < w and z > self.depth_buffer[y, x]:
                    self.frame_buffer[y, x] = color
                    self.depth_buffer[y, x] = z
                P += D
                x, y, z = P

        # Draw the remaining (vertical?) lines, where dx < dy.
        elif abs(D[0]) < abs(D[1]):
            D = D / abs(D[1])

            y0, y1 = int(min(pt1[1], pt0[1])), int(max(pt1[1], pt0[1]))
            for _ in range(y0, y1):
                x, y, _ = P.astype(int)
                z = P[2]
                if 0 <= y < h and 0 <= x < w and z > self.depth_buffer[y, x]:
                    self.frame_buffer[y, x] = color
                    self.depth_buffer[y, x] = z
                P += D
                x, y, z = P


    def wireframe_render(self, width, height, objects, colors, camera_view, steps = 32):
        """ Takes in an unordered list of objects and renders their wireframes
            using the orthographic or equirectangular projection method.

        :param width:       Width of the output image.
        :param height:      Height of the output image.
        :param objects:     List of primitive objects (e.g. Cubes)
        :param colors:      List of tuples representing RGB-colors (e.g. (255, 0, 0))
        :param camera_view: The camera view to use ("equirectangular", "birdseye" or "top down").
        :param steps:       Number of segments to draw an edge with (for equirectangular projection)
        :return:            Numpy array with shape (height, width, 3).
        """
        # Initialize frame-buffer with zeros.
        self.frame_buffer = np.zeros((height, width, 3))

        # Initialize the depth-buffer and set all entries to infinity.
        self.depth_buffer = -np.inf * np.ones((height, width, 1))

        # Iterate through objects...
        for obj, color in zip(objects, colors):

            # Select type of projection.
            if camera_view == "equirectangular":
                pts = obj.equirectangular_fragments(width, height, steps)
            else:
                pts = obj.orthographic_fragments(width, height, camera_view)

            # Iterate through the line segments required to draw the object.
            for pt0, pt1 in pts:
                self.__draw_line(pt0, pt1, color)

        return self.frame_buffer



import cv2

cube1 = Cuboid()
cube1.set_location(0, -0.7, 0)
cube1.set_scale(1.5, 0.3, 0.2)

cube2 = Cuboid()
cube2.set_scale(0.25, 0.25, 0.25)
cube2.set_location(1.75, 0.3, 0)

cube3 = Cuboid()
cube3.set_scale(2, 1, 1)


view = Viewer()
im = view.wireframe_render(600, 600, [cube1, cube2, cube3], [(0, 255, 0), (255, 0, 0), (0, 0, 255)], "top down")

cv2.imshow('', im)
cv2.waitKey(0)



