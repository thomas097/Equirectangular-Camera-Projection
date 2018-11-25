import numpy as np


"""
Project:      Equirectangular Camera Mapping
Date:         25 nov 2018
Name:         Thomas Bellucci
Description:  This file contains the projection method(s) used in the project.
"""


def equirectangular_projection(point, width, dtype=np.int16):
    """ Projects a point in camera coordinates using equirectangular projection
        onto the 2:1 view plane defined by longitude and latitude coordinates.

    :param point:  Numpy array of point representing (x, y, z)^T.
    :param width:  Width of the view plane in pixels.
    :return:       Numpy array of point in spherical coordinates.
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