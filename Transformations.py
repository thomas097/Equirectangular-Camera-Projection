"""
Project:      Equirectangular Camera Mapping
Name:         Transformations.py
Date:         November 25th, 2018.
Author:       Thomas Bellucci
Description:  Transformation matrices / projection functions required for the
              manipulation and viewing of objects in the 3D environment and
              viewport.
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


def camera_matrix(e, p, t):
    """ Constructs a 4D homogeneous camera matrix used to convert points
        defined in XYZ world coordinates into UVW camera coordinates.

    :param e: Position of the camera in world coordinates.
    :param p: Point in space the camera is looking at (lookat).
    :param t: Up-vector.
    :return:  4x4 camera matrix.
    """
    # Translates all points such that the camera is centered at the origin.
    T = np.array([[1, 0, 0, -e[0]],
                  [0, 1, 0, -e[1]],
                  [0, 0, 1, -e[2]],
                  [0, 0, 0,     1]])

    # Set up orthonormal basis.
    w = e - p
    w = w / np.linalg.norm(w)
    u = np.cross(t, w)
    u = u / np.linalg.norm(u)
    v = np.cross(w, u)

    # Rotate points such that camera is aligned with UVW-axes (g -> -z-axis).
    R = np.array([[u[0], u[1], u[2], 0],
                  [v[0], v[1], v[2], 0],
                  [w[0], w[1], w[2], 0],
                  [   0,    0,    0, 1]])

    return R.dot(T)


def orthographic_projection_matrix(width, height, w = 5, h = 5, n = 0.01, f = 5):
    """ Constructs a 4D homogeneous orthographic projection which takes the
        orthographic view volume in camera coordinates and projects points in
        the z-direction onto the view plane.

    :param width:  Width of the viewport image in pixels.
    :param height: Height of the viewport image in pixels.
    :param w:      Width of the near plane (along x-axis).
    :param h:      Height of the near plane (along y-axis).
    :param n:      Position of near plane along z-axis.
    :param f:      Position of far plane along z-axis.
    :return:       4x4 perspective transformation matrix.
    """
    # Transformation that maps from orthographic vv to canonical vv.
    Morth = np.array([[2/w,   0,       0,            0],
                      [  0, 2/h,       0,            0],
                      [  0,   0, 2/(n-f), -(n+f)/(n-f)],
                      [  0,   0,       0,            1]])

    # Transform canonical vv to viewport.
    Mvp = np.array([[width/2,        0, 0,  (width-1)/2],
                    [      0, height/2, 0, (height-1)/2],
                    [      0,        0, 1,            0],
                    [      0,        0, 0,            1]])

    return Mvp.dot(Morth)


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

