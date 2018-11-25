"""
Project:      Equirectangular Camera Mapping
Name:         Transformations.py
Date:         November 25th, 2018.
Author:       Thomas Bellucci
Description:  Transformation matrices required for object / primitive
              manipulation.
"""

import numpy as np


def rotation_matrix(rx, ry, rz):
    """ Constructs 4D homogeneous rotation matrix given the rotation angles
        (degrees) around the x, y and z-axis. Rotation is implemented in
        XYZ order.

    :param rx: Rotation around the x-axis in degrees.
    :param ry: Rotation around the y-axis in degrees.
    :param rz: Rotation around the z-axis in degrees.
    :return:   4x4 matrix rotation matrix.
    """
    # Convert from degrees to radians.
    rx = np.pi * rx / 180
    ry = np.pi * ry / 180
    rz = np.pi * rz / 180

    # Pre-compute sine and cosine of angles.
    cx, cy, cz = np.cos([rx, ry, rz])
    sx, sy, sz = np.sin([rx, ry, rz])

    # Set up euler rotations.
    Rx = np.array([[1, 0,  0,   0],
                   [0, cx, -sx, 0],
                   [0, sx, cx,  0],
                   [0, 0,  0,   1]])

    Ry = np.array([[cy,  0, sy, 0],
                   [0,   1, 0,  0],
                   [-sy, 0, cy, 0],
                   [0,   0, 0,  1]])

    Rz = np.array([[cz, -sz, 0, 0],
                   [sz, cz,  0, 0],
                   [0,  0,   1, 0],
                   [0,  0,   0, 1]])
    return Rz.dot(Ry.dot(Rx))


def translation_matrix(tx, ty, tz):
    """ Constructs a 4D homogeneous translation matrix given translations
        along each axis in 3D.

    :param tx: Translation along x-axis.
    :param ty: Translation along y-axis.
    :param tz: Translation along z-axis.
    :return:   4x4 matrix translation matrix.
    """
    T = np.array([[1, 0, 0, tx],
                  [0, 1, 0, ty],
                  [0, 0, 1, tz],
                  [0, 0, 0, 1 ]])
    return T


def scaling_matrix(sx, sy, sz):
    """ Constructs a 4D homogeneous scaling matrix given coefficients for
        along each axis in 3D.

    :param sx: Scaling factor along x-axis.
    :param sy: Scaling factor along y-axis.
    :param sz: Scaling factor along z-axis.
    :return:   4x4 matrix scaling matrix.
    """
    S = np.array([[sx, 0,  0,  0],
                  [0,  sy, 0,  0],
                  [0,  0,  sz, 0],
                  [0,  0,  0,  1]])
    return S