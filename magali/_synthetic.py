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
import verde as vd

from ._utils import _convert_micrometer_to_meter


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
        Declination of the preferred direction in degrees.
    dispersion_angle : float
        Dispersion angle in degrees that defines a region on the surface of
        a sphere. Corresponds to the variance of the generated directions.
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
        Declination of the generated directions in degrees.

    Notes
    -----
    We calculate the azimuth (:math:`\alpha`) via a random uniform distribution
    ranging from 0° to 360°, to represent the equal probability for each
    value in a stereographic polar projection.
    The distance to the pole (:math:`\psi`) for each vector is obtained with a
    normal distribution.
    Given the values of azimuth (:math:`\alpha`) and distance to the pole
    (:math:`\psi`), we calculate the cartesian coordinates

    .. math::
        x = \sin(\psi) * cos(\alpha)
    .. math::
        y = np.sin(\psi) * sin(\alpha)
    .. math::
        z = np.cos(\psi)

    The rotation of the pole is performed to the preferred direction using the
    specified values of inclination and declination to obtain the rotation
    vector. With :math:`\theta=90°+inclination` and
    :math:`\phi=90°-declination` we calculate the rotation
    vector by

    .. math::
        \mathbf{u}=\begin{bmatrix}
        x\\y\\z
        \end{bmatrix}\\
    .. math::
        \mathbf{R}_z(\phi)\mathbf{R}_y(\theta)\mathbf{u}=\begin{bmatrix}
        \cos(\theta)&0&\sin(\theta)\\
        0&1&0\\
        -\sin(\theta)&0&\cos(\cos)
        \end{bmatrix}\begin{bmatrix}
        \cos(\phi)&-\sin(\phi)&0\\
        \sin(\phi)&\cos(\phi)&0\\
        0&0&1
        \end{bmatrix}\begin{bmatrix}
        x\\y\\z
        \end{bmatrix}\\
    .. math::

        \mathbf{R}_z(\phi) \mathbf{R}_y(\theta)\mathbf{u}  =   \begin{bmatrix}
        \cos(\phi)(x\cos(\theta)+z\sin(\theta))-y\sin(\phi) \\
        \sin(\phi)(x\cos(\theta)+z\sin(\theta))+y\cos(\phi) \\
        -x\sin(\theta)+z\cos(\theta)
        \end{bmatrix}

    in which :math:`\mathbf{R}_z(\phi)` and :math:`\mathbf{R}_z(\phi)` are the
    rotation matrices on the z and y axes, respectively.
    """
    # Set random number generator
    rng = np.random.default_rng(random_state)

    azimuth = np.deg2rad(rng.uniform(0, 360, size))
    distance = rng.normal(0, np.deg2rad(dispersion_angle), size)
    x = np.sin(distance) * np.cos(azimuth)
    y = np.sin(distance) * np.sin(azimuth)
    z = np.cos(distance)

    x_r, y_r, z_r = _rotate_vector(x, y, z, inclination, declination)

    _, directions_inclination, directions_declination = hm.magnetic_vec_to_angles(
        x_r, y_r, z_r
    )
    return directions_inclination, directions_declination


def _rotate_vector(x, y, z, inclination, declination):
    """
    Rotates vectors using its cartesian coordinates towards specific direction

    Parameters
    ----------
    x : numpy.array
        x coordinates of vectors to be rotated.
    y : numpy.array
        y coordinates of vectors to be rotated.
    z : numpy.array
        z coordinates of vectors to be rotated.
    inclination : float
        Inclination of the preferred direction in degrees.
    declination : float
        Declination of the preferred direction in degrees.

    Returns
    -------
    x_r : numpy.array
        x coordinates of rotated vectors.
    y_r : numpy.array
        y coordinates of rotated vectors.
    z_r : numpy.array
        z coordinates of rotated vectors.
    """

    theta = np.deg2rad(90 + inclination)
    phi = np.deg2rad(90 - declination)
    # R_z(phi)*R_y(theta)
    x_r = np.cos(phi) * (x * np.cos(theta) + z * np.sin(theta)) - y * np.sin(phi)
    y_r = np.sin(phi) * (x * np.cos(theta) + z * np.sin(theta)) + y * np.cos(phi)
    z_r = -x * np.sin(theta) + z * np.cos(theta)
    return x_r, y_r, z_r


def dipole_bz(coordinates, dipole_coordinates, dipole_moments):
    """
    Computes the vertical component of the magnetic field (Bz) produced by a magnetic dipole.

    Parameters
    ----------
    coordinates : tuple of float
        Observation point coordinates in micrometers (μm).
    dipole_coordinates : tuple of float
        Dipole location coordinates in micrometers (μm).
    dipole_moments : tuple of float
        Dipole moment components (Am²).

    Returns
    -------
    bz : float
        The vertical component of the magnetic field (Bz) at the given observation point.
    """
    coordinates_m = _convert_micrometer_to_meter(coordinates)
    dipole_coordinates_m = _convert_micrometer_to_meter(dipole_coordinates)
    return hm.dipole_magnetic(
        coordinates_m, dipole_coordinates_m, dipole_moments, field="b_u"
    )


def dipole_bz_grid(coordinates, dipole_coordinates, dipole_moments):
    """
    Computes the vertical component of the magnetic field (Bz) produced by a magnetic dipole
    and returns it as a gridded dataset.

    Parameters
    ----------
    coordinates : tuple of numpy.array
        Grid coordinates in micrometers (μm) with shape (3, N), where N is the number of points.
    dipole_coordinates : tuple of float
        Dipole location coordinates in micrometers (μm).
    dipole_moments : tuple of float
        Dipole moment components (Am²).

    Returns
    -------
    data : xarray.Dataset
        Gridded dataset containing the vertical component of the magnetic field (Bz).
        The dataset includes:
        - "bz" : vertical magnetic field (nT)
        - "x" and "y" coordinates with units in micrometers (μm)
    """
    bz = dipole_bz(coordinates, dipole_coordinates, dipole_moments)
    data = vd.make_xarray_grid(
        coordinates, bz, data_names=["bz"], dims=("y", "x"), extra_coords_names="z"
    )
    data.x.attrs = {"units": "µm"}
    data.y.attrs = {"units": "µm"}
    data.bz.attrs = {"long_name": "vertical magnetic field", "units": "nT"}
    return data
