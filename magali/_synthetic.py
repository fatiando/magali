# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Functions to generate synthetic data
"""

import harmonica as hm
import numpy as np
import scipy.io
import verde as vd


def random_directions(
    inclination, declination, dispersion_angle, size, random_state=None
):
    r"""
    Creates random directions around a preferred direction.

    Parameters
    ----------
    inclination : float
        Inclination of the preferred direction in degrees.
    declination : float
        Declination of the preferred direction in degrees
    dispersion_angle : float
        Dispersion angle that defines a region on the surface of a sphere.
    size : int
        Number of random directions to be generated.
    random_state : numpy.random.RandomState or an int seed
        A random number generator used to define the state of the random
        permutations. Use a fixed seed to make sure computations are
        reproducible. Use ``None`` to choose a seed automatically (resulting in
        different numbers with each run).

    Returns
    -------
    inclination : numpy.array
        Inclination of the generated directions in degrees.
    declination : numpy.array
        Declination of the generated directions in degre

    Notes
    -----

    Explain everything
    - Generate random directions
    - Rotation
    >- R_z(phi)*R_y(theta)
    
    This function returns random directions to a preferred direction in a
    spherographic projection. That is achieved in two steps: the creation of 
    random vectors representing the projetction of a magnetic vector and the
    rotation of specific inclination and declination.

    Creation of random vectors:
    
    First, a colection of values is obtained in an uniform distribution ranging
    from 0 to 360 degrees to represent the directions of the projection of the 
    vectors.

    From this distribution, the verctors and its cartesian coordinates are 
    calculated 

    Rotation of vectors: 
    
    
    .. math::

        R_z(\phi)R_y(\theta) =      \begin{bmatrix}
        \cos(\theta) & 0 & \sin(\theta) \\
        0 & 1 & 0 \\
        -\sin(\theta) & 0 & \cos(\cos)
        \end{bmatrix} \begin{bmatrix}
         \cos(\phi) & -\sin(\phi) & 0 \\
         \sin(\phi) & \cos(\phi) & 0 \\
         0 & 0 & 1
        \end{bmatrix}  \begin{bmatrix}
         x \\ y \\ z   
        \end{bmatrix} \\
    
    .. math::

        R_z(\phi)R_y(\theta) =   \begin{bmatrix}
        \cos(\phi)(X\cos(\theta)+Z\sin(\theta))-Y\sin(\phi) \\
        \sin(\phi)(X\cos(\theta)+Z\sin(\theta))+Y\cos(\phi) \\
        -X\sin(\theta)+Z\cos(\theta)
        \end{bmatrix}
    """
    alpha = np.deg2rad(np.random.uniform(0, 360, size))
    r = np.random.normal(0, dispersion_angle, size)
    x = np.sin(r) * np.cos(alpha)
    y = np.sin(r) * np.sin(alpha)
    z = np.cos(r)

    x_r, y_r, z_r = _rotate_vector(x, y, z, inclination, declination)

    _, inc, dec = hm.magnetic_vec_to_angles(x_r, y_r, z_r)
    return inc, dec


def _rotate_vector(x, y, z, inclination, declination):
    """
    Creates a random unitary vector from a defined dispersion angle.



    Parameters
    ----------
    r_vector : :class:'numpy.ndarray'
        A random unitary vector.
    inclination : float or array
        Inclination of the magnetic vector in degrees.
    declination : float or array
        Declination of the magnetic vector in degrees

    Returns
    -------
    rotatated_vector : :class:'numpy.ndarray'
        The rotated vector.
    """

    theta = np.deg2rad(90 + inclination)
    phi = np.deg2rad(90 - declination)
    # R_z(phi)*R_y(theta)
    x_r = np.cos(phi) * (x * np.cos(theta) + z * np.sin(theta)) - y * np.sin(phi)
    y_r = np.sin(phi) * (x * np.cos(theta) + z * np.sin(theta)) + y * np.cos(phi)
    z_r = -x * np.sin(theta) + z * np.cos(theta)
    return x_r, y_r, z_r
