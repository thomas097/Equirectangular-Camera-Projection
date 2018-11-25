"""
Project:      Equirectangular Camera Mapping
Name:         Projections.py
Date:         November 25th, 2018.
Author:       Thomas Bellucci
Description:  The projection method(s) used for rendering objects / primitives
              in the viewport.
"""

import numpy as np


def equirectangular_projection(point, width, dtype=np.int16):
    """ Projects a point in camera coordinates using equirectangular projection
        onto the 2:1 view plane defined by longitude and latitude coordinates.

    :param point:  Numpy array of point representing (x, y, z)^T.
    :param width:  Width of the view plane in pixels.
    :param dtype:  Dtype of the output array (default: 16-bit integer).
    :return:       Numpy array of point as (u, v) pixel coordinates.
    """
    x, y, z = point

    # Convert point to spherical coordinates (radius, longitude, latitude)^T.
    rad = np.linalg.norm(point)
    lon = np.arctan2(y, x)
    lat = np.arcsin(z / rad)

    # Apply window transformation to correct value ranges.
    u = width * (lon + np.pi) / (2 * np.pi)
    v = (width / 2) * (lat + (np.pi / 2)) / np.pi

    return np.array([u, v], dtype=dtype)


def perspective_projection(point, width, height, dtype=np.uint16):
    """ Projects a point in world coordinates onto the view plane using a
        perspective projection.

    :param point:  Numpy array of point representing (x, y, z)^T.
    :param width:  Width of the view plane in pixels.
    :param height: Height of the view plane in pixels.
    :param dtype:  Dtype of the output array (default: 16-bit integer).
    :return:       Numpy array of point as (u, v) pixel coordinates.
    """
    # TODO: implement!
    pass